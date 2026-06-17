"""
run_probe.py — Experiment harness for probing LatentMAS.

This is a standalone clone of the LatentMAS run path, instrumented for the
"latent thoughts: reasoning vs conditioning?" investigation. It does NOT modify
run.py / run_reproduce.py / methods/* — it reuses ModelWrapper's latent and text
generation primitives and adds experiment hooks.

Key controllable axes (each can be swept independently):
  - latent_steps (m)             : --latent_steps 0 10 20 40 80   (sweepable list)
  - judger reasoning mode (T/1C) : --judger_mode {think,nothink,direct}
        think   : enable_thinking=True + CoT prompt (paper default)
        nothink : enable_thinking=False, but CoT prompt still elicits reasoning
        direct  : no CoT, answer immediately -> forces latent KV to carry reasoning (1C)
  - agent thinking mode          : --agent_thinking {on,off}
  - judger answer budget         : --judger_max_new_tokens N   (separate from agents)
  - KV isolation                 : --kv_mode {full,sequential_info_only,latent_only}
  - latent source decoupling     : --latent_source {self,transplant,generic,none}
        self       : latent agents run on the target question (paper default)
        transplant : latent agents run on a DIFFERENT question (cross-question 2A)
        generic    : latent agents run on a fixed content-free prompt (2D)
        none       : skip latent agents entirely (judger only; ~baseline+MAS prompt)
  - realignment                  : --latent_space_realign / (default off)

Metrics logged per (task, condition):
  - accuracy
  - judger thinking tokens vs answer tokens (split on </think>)
  - latent forward passes (compute proxy)
  - wall-clock time

Usage examples:
  # Trục T x m sweep on the decisive experiment:
  python run_probe.py --model_name Qwen/Qwen3-4B --tasks gsm8k arc_easy \
      --max_samples 100 --latent_steps 0 10 20 40 --judger_mode think direct

  # Cross-question transplant:
  python run_probe.py --model_name Qwen/Qwen3-4B --tasks gsm8k \
      --max_samples 100 --latent_steps 40 --latent_source transplant
"""

import argparse
import json
import os
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from data import (
    load_gsm8k,
    load_arc_easy,
    load_arc_challenge,
    load_aime2024,
    load_aime2025,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa,
)
from methods import default_agents, Agent
from models import ModelWrapper, _past_length
from prompts import (
    build_agent_message_sequential_latent_mas,
    build_agent_message_hierarchical_latent_mas,
)
from utils import (
    auto_device,
    set_seed,
    extract_gsm8k_answer,
    normalize_answer,
    extract_markdown_python_block,
    run_with_timeout,
)

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


# ---------------------------------------------------------------------------
# Per-task max output length (from paper Implementation Details)
# ---------------------------------------------------------------------------

TASK_MAX_TOKENS = {
    "gsm8k": 2048,
    "arc_easy": 2048,
    "arc_challenge": 2048,
    "medqa": 4096,
    "mbppplus": 4096,
    "humanevalplus": 4096,
    "gpqa": 8192,
    "aime2024": 20000,
    "aime2025": 20000,
}

TASK_LOADERS = {
    "gsm8k": lambda: load_gsm8k(split="test"),
    "arc_easy": lambda: load_arc_easy(split="test"),
    "arc_challenge": lambda: load_arc_challenge(split="test"),
    "aime2024": lambda: load_aime2024(split="train"),
    "aime2025": lambda: load_aime2025(split="train"),
    "gpqa": lambda: load_gpqa_diamond(split="test"),
    "mbppplus": lambda: load_mbppplus(split="test"),
    "humanevalplus": lambda: load_humanevalplus(split="test"),
    "medqa": lambda: load_medqa(split="test"),
}

ALL_TASKS = list(TASK_LOADERS.keys())

# A fixed content-free prompt for the `generic` latent-source ablation (2D).
# It should be answerable/encodable but carry NO information about the target task.
GENERIC_QUESTION = (
    "Think carefully and reason step by step before producing a careful, "
    "well-structured answer to the user's request."
)


def load_task(task: str, max_samples: int, seed: int) -> List[Dict]:
    if task not in TASK_LOADERS:
        raise ValueError(f"Unsupported task: {task}")
    data = list(TASK_LOADERS[task]())
    if max_samples is not None and max_samples > 0:
        data = data[:max_samples]
    return data


# ---------------------------------------------------------------------------
# Answer evaluation (mirrors methods/latent_mas.py run_batch logic)
# ---------------------------------------------------------------------------

def evaluate_prediction(task: str, final_text: str, item: Dict) -> Tuple[str, str, bool]:
    if task in ["mbppplus", "humanevalplus"]:
        pred = extract_markdown_python_block(final_text)
        gold = item.get("gold", "")
        if pred is None:
            return ("", gold, False)
        ok, _ = run_with_timeout(pred + "\n" + gold, timeout=10)
        return (pred, gold, bool(ok))

    if task in ["aime2024", "aime2025"]:
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = str(item.get("gold", "")).strip()
        try:
            return (pred, gold, int(pred) == int(gold))
        except (ValueError, TypeError):
            return (pred or "", gold, False)

    pred = normalize_answer(extract_gsm8k_answer(final_text))
    gold = item.get("gold", "")
    ok = (pred == gold) if (pred and gold) else False
    return (pred or "", gold, ok)


# ---------------------------------------------------------------------------
# Thinking / answer token split (Qwen3 emits <think> ... </think> then answer)
# ---------------------------------------------------------------------------

def split_think(raw_text: str) -> Tuple[str, str]:
    """Return (thinking_text, answer_text) by splitting on the last </think>."""
    marker = "</think>"
    if marker in raw_text:
        idx = raw_text.rfind(marker)
        return raw_text[:idx], raw_text[idx + len(marker):]
    # No closing think tag: treat everything as answer (model answered directly),
    # unless an opening tag exists and was never closed (then it's all thinking).
    if "<think>" in raw_text:
        return raw_text, ""
    return "", raw_text


# ---------------------------------------------------------------------------
# Chat rendering with explicit thinking control (does not touch models.py)
# ---------------------------------------------------------------------------

def render_messages(model: ModelWrapper, messages: List[Dict], enable_thinking: bool) -> str:
    """Render a chat message list to a prompt string, controlling Qwen3 thinking mode.

    Qwen3's chat template accepts `enable_thinking`. We pass it explicitly so the
    judger's token-space reasoning budget becomes a controllable experiment axis
    instead of silently defaulting to True (as in the original run path).
    """
    tok = model.tokenizer
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Template does not support enable_thinking; fall back to default render.
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def build_messages(model_name: str, prompt_arch: str, role: str, question: str, task: str):
    """Wrapper over the latent_mas prompt builders. Builds a throwaway args-like
    namespace carrying only what the prompt builders read (model_name, task)."""
    ns = argparse.Namespace(model_name=model_name, task=task)
    if prompt_arch == "sequential":
        return build_agent_message_sequential_latent_mas(
            role=role, question=question, context="", method="latent_mas", args=ns
        )
    else:
        return build_agent_message_hierarchical_latent_mas(
            role=role, question=question, context="", method="latent_mas", args=ns
        )


# Direct-answer judger prompt for the 1C experiment: forbids chain-of-thought so
# that any reasoning must be carried by the accumulated latent KV, not by the
# judger's own token-space deliberation.
def build_judger_direct_messages(task: str, question: str):
    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    if task in ["gsm8k", "aime2024", "aime2025"]:
        fmt = ("Output ONLY the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
               "Do NOT explain. Do NOT show any reasoning or steps.")
    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        fmt = ("Output ONLY the final answer letter inside \\boxed{} (A, B, C, or D). "
               "Do NOT explain. Do NOT show any reasoning or steps.")
    elif task in ["mbppplus", "humanevalplus"]:
        fmt = ("Output ONLY a self-contained Python function in a single ```python code block. "
               "Do NOT explain. Do NOT show any reasoning.")
    elif task in ["winogrande"]:
        fmt = ("Output ONLY the final answer inside \\boxed{} (1 or 2). "
               "Do NOT explain. Do NOT show any reasoning or steps.")
    else:
        fmt = "Output ONLY the final answer. Do NOT explain."
    user_prompt = (
        f"Target Question: {question}\n\n"
        f"You are provided with latent information for reference. "
        f"Ignore it if it is not helpful.\n{fmt}\n"
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def build_judger_messages(model_name: str, prompt_arch: str, question: str, task: str, judger_mode: str):
    if judger_mode == "direct":
        return build_judger_direct_messages(task, question)
    return build_messages(model_name, prompt_arch, "judger", question, task)


# Latent (non-judger) agent presets. Used for the agent-ablation experiment
# (drop one agent at a time) and for varying the pipeline size.
AGENT_SETS = {
    "full":       ["planner", "critic", "refiner"],
    "no_planner": ["critic", "refiner"],
    "no_critic":  ["planner", "refiner"],
    "no_refiner": ["planner", "critic"],
    "planner":    ["planner"],
    "critic":     ["critic"],
    "refiner":    ["refiner"],
    "none":       [],
}
_ROLE_NAME = {"planner": "Planner", "critic": "Critic", "refiner": "Refiner"}


def build_agents(agent_set: str) -> List[Agent]:
    return [Agent(name=_ROLE_NAME[r], role=r) for r in AGENT_SETS[agent_set]]


def tokenize_prompts(model: ModelWrapper, prompts: List[str]):
    enc = model.tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    )
    return enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device)


# ---------------------------------------------------------------------------
# KV-cache truncation (mirrors LatentMASMethod._slice_tensor / _truncate_past)
# ---------------------------------------------------------------------------

def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    if tokens_to_keep <= 0:
        return tensor[..., 0:0, :].contiguous()
    keep = min(tokens_to_keep, tensor.shape[-2])
    start = tensor.shape[-2] - keep
    return tensor[..., start:, :].contiguous()


def truncate_past(past_kv, tokens_to_keep: int):
    if past_kv is None or tokens_to_keep <= 0:
        return None
    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
        trimmed = tuple(
            tuple(_slice_tensor(t, tokens_to_keep) for t in layer) for layer in legacy
        )
        return past_kv.__class__.from_legacy_cache(trimmed)
    trimmed_layers = []
    for layer in past_kv:
        if isinstance(layer, tuple):
            trimmed_layers.append(tuple(_slice_tensor(t, tokens_to_keep) for t in layer))
        elif torch.is_tensor(layer):
            trimmed_layers.append(_slice_tensor(layer, tokens_to_keep))
        else:
            trimmed_layers.append(layer)
    return tuple(trimmed_layers)


# ---------------------------------------------------------------------------
# Experiment condition spec
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    latent_steps: int
    judger_mode: str        # think | nothink | direct
    agent_thinking: bool
    judger_max_new_tokens: int
    kv_mode: str            # full | sequential_info_only | latent_only
    latent_source: str      # self | transplant | generic | none
    agent_set: str          # which latent (non-judger) agents are active
    final_latent: bool      # judger also thinks in latent, then a decoder verbalizes
    prompt_arch: str        # sequential | hierarchical
    realign: bool

    def tag(self) -> str:
        return (
            f"m{self.latent_steps}"
            f"_j-{self.judger_mode}"
            f"_at{'1' if self.agent_thinking else '0'}"
            f"_jb{self.judger_max_new_tokens}"
            f"_kv-{self.kv_mode}"
            f"_src-{self.latent_source}"
            f"_as-{self.agent_set}"
            f"_fl{'1' if self.final_latent else '0'}"
            f"_{self.prompt_arch}"
            f"_ra{'1' if self.realign else '0'}"
        )


# ---------------------------------------------------------------------------
# Core: run the LatentMAS pipeline for one batch under one Condition
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_condition_batch(
    model: ModelWrapper,
    agents: List[Agent],
    target_items: List[Dict],
    latent_source_items: List[Dict],
    task: str,
    cond: Condition,
    temperature: float,
    top_p: float,
) -> List[Dict]:
    """
    target_items        : questions the JUDGER must answer (the eval question).
    latent_source_items : questions the LATENT agents (planner/critic/refiner) run on.
                          For `self` this equals target_items; for `transplant` it is a
                          shuffled/other-problem batch; for `generic` it is the fixed
                          content-free prompt; for `none` latent agents are skipped.
    """
    batch_size = len(target_items)
    past_kv = None
    running_mask = None
    latent_forward_passes = 0

    # --- Latent agents (planner / critic / refiner) -----------------------
    # m==0 (or source "none") means no latent collaboration: skip the agents
    # entirely so the judger runs on the target question alone. This keeps the
    # m=0 control clean and reports latent_forward_passes=0 honestly.
    if cond.latent_source != "none" and cond.latent_steps > 0:
        for agent in agents:
            if agent.role == "judger":
                continue

            # Build the prompt for the latent source question(s)
            src_questions = [it["question"] for it in latent_source_items]
            prompts = [
                render_messages(
                    model,
                    build_messages(model.model_name, cond.prompt_arch, agent.role, q, task),
                    enable_thinking=cond.agent_thinking,
                )
                for q in src_questions
            ]
            input_ids, attn = tokenize_prompts(model, prompts)

            prev_len = _past_length(past_kv)
            past_kv, running_mask = model.generate_latent_batch(
                input_ids,
                attention_mask=attn,
                latent_steps=cond.latent_steps,
                past_key_values=past_kv,
                past_attention_mask=running_mask,
                return_mask=True,
            )
            # one forward pass for the prompt + one per latent step
            latent_forward_passes += 1 + cond.latent_steps

            if cond.kv_mode in ("sequential_info_only", "latent_only"):
                new_len = _past_length(past_kv)
                tokens_added = new_len - prev_len
                tokens_to_keep = (
                    cond.latent_steps if cond.kv_mode == "latent_only" else tokens_added
                )
                past_kv = truncate_past(past_kv, tokens_to_keep)
                if running_mask is not None and tokens_to_keep > 0:
                    running_mask = running_mask[:, -tokens_to_keep:]

    # --- direct-v2: let the solver (judger) also think in latent ----------
    # The judger runs m latent steps on the TARGET question (contributing KV),
    # then a separate text decode verbalizes the answer. Tests whether pushing
    # the final reasoning into latent (instead of text CoT) is decodable.
    if cond.final_latent and cond.latent_steps > 0:
        jl_prompts = [
            render_messages(
                model,
                build_judger_messages(model.model_name, cond.prompt_arch, it["question"], task, cond.judger_mode),
                enable_thinking=cond.agent_thinking,
            )
            for it in target_items
        ]
        jl_ids, jl_attn = tokenize_prompts(model, jl_prompts)
        past_kv, running_mask = model.generate_latent_batch(
            jl_ids,
            attention_mask=jl_attn,
            latent_steps=cond.latent_steps,
            past_key_values=past_kv,
            past_attention_mask=running_mask,
            return_mask=True,
        )
        latent_forward_passes += 1 + cond.latent_steps

    # --- Judger / decoder (text decode, conditioned on accumulated KV) -----
    judger_enable_thinking = (cond.judger_mode == "think")
    judger_prompts = [
        render_messages(
            model,
            build_judger_messages(model.model_name, cond.prompt_arch, it["question"], task, cond.judger_mode),
            enable_thinking=judger_enable_thinking,
        )
        for it in target_items
    ]
    judger_ids, judger_mask = tokenize_prompts(model, judger_prompts)

    use_past = past_kv is not None and _past_length(past_kv) > 0
    past_for_decoding = past_kv if use_past else None

    generated, _ = model.generate_text_batch(
        judger_ids,
        judger_mask,
        max_new_tokens=cond.judger_max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        past_key_values=past_for_decoding,
        past_attention_mask=(running_mask if past_for_decoding is not None else None),
        skip_special_tokens=False,  # keep <think> tags so we can split budget
    )

    results = []
    for idx, item in enumerate(target_items):
        raw = generated[idx].strip()
        thinking_text, answer_text = split_think(raw)
        # Answer extraction runs on the full raw text (extractors take the last match)
        pred, gold, ok = evaluate_prediction(task, raw, item)

        n_think = len(model.tokenizer(thinking_text, add_special_tokens=False)["input_ids"]) if thinking_text else 0
        n_answer = len(model.tokenizer(answer_text, add_special_tokens=False)["input_ids"]) if answer_text else 0

        results.append({
            "question": item["question"],
            "gold": gold,
            "prediction": pred,
            "correct": ok,
            "judger_thinking_tokens": n_think,
            "judger_answer_tokens": n_answer,
            "latent_forward_passes": latent_forward_passes,
            "raw_output_preview": raw[:600],
        })
    return results


# ---------------------------------------------------------------------------
# Build the latent-source batch for a given target batch + source mode
# ---------------------------------------------------------------------------

def make_latent_source(
    target_batch: List[Dict],
    full_dataset: List[Dict],
    global_offset: int,
    latent_source: str,
    rng: random.Random,
) -> List[Dict]:
    """Produce the per-item latent-source questions aligned to target_batch.

    transplant: pair each target with a DIFFERENT problem (deterministic shift by
                len//2 across the full dataset, guaranteeing question_src != question_tgt).
    generic   : the fixed content-free prompt for every item.
    self/none : the target batch itself (for `none`, latent agents are skipped anyway).
    """
    if latent_source in ("self", "none"):
        return target_batch
    if latent_source == "generic":
        return [{"question": GENERIC_QUESTION} for _ in target_batch]
    if latent_source == "transplant":
        n = len(full_dataset)
        shift = max(1, n // 2)
        src = []
        for i in range(len(target_batch)):
            tgt_global = global_offset + i
            src_global = (tgt_global + shift) % n
            if src_global == tgt_global:
                src_global = (src_global + 1) % n
            src.append(full_dataset[src_global])
        return src
    raise ValueError(f"Unknown latent_source: {latent_source}")


# ---------------------------------------------------------------------------
# Run a single condition over a full task dataset
# ---------------------------------------------------------------------------

def run_task_condition(
    model: ModelWrapper,
    task: str,
    dataset: List[Dict],
    cond: Condition,
    generate_bs: int,
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> Dict:
    agents = build_agents(cond.agent_set)
    all_results: List[Dict] = []
    t0 = time.time()

    for start in tqdm(range(0, len(dataset), generate_bs),
                      desc=f"[{task}|{cond.tag()}]", leave=False):
        target_batch = dataset[start:start + generate_bs]
        source_batch = make_latent_source(
            target_batch, dataset, start, cond.latent_source, rng
        )
        batch_results = run_condition_batch(
            model, agents, target_batch, source_batch, task, cond, temperature, top_p
        )
        all_results.extend(batch_results)

    elapsed = time.time() - t0
    n = len(all_results)
    correct = sum(1 for r in all_results if r["correct"])
    acc = correct / n if n else 0.0

    def _mean(key):
        return round(sum(r[key] for r in all_results) / n, 1) if n else 0.0

    summary = {
        "task": task,
        "condition": cond.tag(),
        "latent_steps": cond.latent_steps,
        "judger_mode": cond.judger_mode,
        "agent_thinking": cond.agent_thinking,
        "judger_max_new_tokens": cond.judger_max_new_tokens,
        "kv_mode": cond.kv_mode,
        "latent_source": cond.latent_source,
        "agent_set": cond.agent_set,
        "final_latent": cond.final_latent,
        "prompt_arch": cond.prompt_arch,
        "realign": cond.realign,
        "n": n,
        "correct": correct,
        "accuracy": round(acc, 4),
        "avg_judger_thinking_tokens": _mean("judger_thinking_tokens"),
        "avg_judger_answer_tokens": _mean("judger_answer_tokens"),
        "avg_latent_forward_passes": _mean("latent_forward_passes"),
        "total_time_sec": round(elapsed, 2),
        "time_per_sample_sec": round(elapsed / n, 3) if n else 0.0,
    }
    return {"summary": summary, "results": all_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="LatentMAS probing harness")

    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                   help="Qwen3-family HF model id (e.g. Qwen/Qwen3-4B, Qwen/Qwen3-8B, Qwen/Qwen3-14B).")
    p.add_argument("--tasks", nargs="+", default=["gsm8k"],
                   choices=ALL_TASKS, help="Tasks to evaluate.")
    p.add_argument("--max_samples", type=int, default=100,
                   help="Samples per task (-1 for all).")
    p.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"],
                   default="sequential")

    # --- swept axes (accept lists) ---
    p.add_argument("--latent_steps", type=int, nargs="+", default=[40],
                   help="List of m values to sweep, e.g. 0 10 20 40 80.")
    p.add_argument("--judger_mode", nargs="+", choices=["think", "nothink", "direct"],
                   default=["think"],
                   help="Judger reasoning mode(s) to sweep (trục T / experiment 1C). "
                        "think=CoT+<think>; nothink=CoT only; direct=no CoT, answer immediately.")
    p.add_argument("--agent_thinking", choices=["on", "off"], default="off",
                   help="Latent agents' thinking mode (paper default: not manually opened).")
    p.add_argument("--kv_mode", nargs="+",
                   choices=["full", "sequential_info_only", "latent_only"], default=["full"],
                   help="KV isolation mode(s) to sweep.")
    p.add_argument("--latent_source", nargs="+",
                   choices=["self", "transplant", "generic", "none"], default=["self"],
                   help="Where latent agents get their question from.")
    p.add_argument("--agent_set", nargs="+", choices=list(AGENT_SETS.keys()), default=["full"],
                   help="Latent (non-judger) agent pipeline(s) to sweep. Use for ablation: "
                        "full / no_planner / no_critic / no_refiner / single roles / none.")
    p.add_argument("--final_latent", action="store_true",
                   help="direct-v2: judger also runs m latent steps before a final text decode.")

    # --- fixed knobs ---
    p.add_argument("--judger_max_new_tokens", type=int, default=None,
                   help="Judger answer budget. Default: per-task paper value.")
    p.add_argument("--latent_space_realign", action="store_true",
                   help="Build true realignment matrix (else identity+renorm). Run-level.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--generate_bs", type=int, default=20)
    p.add_argument("--output_dir", type=str, default=None)

    args = p.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = auto_device(args.device)

    # ModelWrapper reads these off args at construction time.
    args.method = "latent_mas"
    args.use_vllm = False
    args.task = args.tasks[0]
    args.think = False

    if args.output_dir is None:
        model_short = args.model_name.replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("output_probe", model_short, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model_name} on {device} (realign={args.latent_space_realign}) ...")

    # Save the exact command and args for reproducibility
    import sys
    meta = {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(args.output_dir, "run_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    print("Model loaded.\n")

    jt_list = list(args.judger_mode)
    agent_thinking = args.agent_thinking == "on"

    all_summaries: List[Dict] = []

    for task in args.tasks:
        dataset = load_task(task, args.max_samples, args.seed)
        default_budget = TASK_MAX_TOKENS.get(task, 2048)
        judger_budget = args.judger_max_new_tokens or default_budget

        for m in args.latent_steps:
            for jt in jt_list:
                for kv_mode in args.kv_mode:
                    for src in args.latent_source:
                      for aset in args.agent_set:
                        cond = Condition(
                            latent_steps=m,
                            judger_mode=jt,
                            agent_thinking=agent_thinking,
                            judger_max_new_tokens=judger_budget,
                            kv_mode=kv_mode,
                            latent_source=src,
                            agent_set=aset,
                            final_latent=args.final_latent,
                            prompt_arch=args.prompt,
                            realign=args.latent_space_realign,
                        )
                        out = run_task_condition(
                            model, task, dataset, cond,
                            args.generate_bs, args.temperature, args.top_p, rng,
                        )
                        s = out["summary"]
                        all_summaries.append(s)
                        print(
                            f"[{task}] {cond.tag()} -> acc={s['accuracy']:.4f} "
                            f"({s['correct']}/{s['n']}) | "
                            f"think_tok={s['avg_judger_thinking_tokens']} "
                            f"ans_tok={s['avg_judger_answer_tokens']} "
                            f"lat_fwd={s['avg_latent_forward_passes']} "
                            f"| {s['time_per_sample_sec']}s/it"
                        )

                        # per-condition detailed dump
                        cond_path = os.path.join(
                            args.output_dir, f"{task}__{cond.tag()}.json"
                        )
                        with open(cond_path, "w", encoding="utf-8") as f:
                            json.dump(out, f, ensure_ascii=False, indent=2)

    # combined summary table
    combined = os.path.join(args.output_dir, "summaries.json")
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    print("\n===== SUMMARY =====")
    for s in all_summaries:
        print(
            f"{s['task']:14s} {s['condition']:48s} "
            f"acc={s['accuracy']:.4f} think={s['avg_judger_thinking_tokens']:>6} "
            f"ans={s['avg_judger_answer_tokens']:>5} lat_fwd={s['avg_latent_forward_passes']:>5}"
        )
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
