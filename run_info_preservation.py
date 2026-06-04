"""
run_info_preservation.py — Information preservation probe for latent communication.

Measures how much information survives through latent KV-cache communication
between two agents. A minimal 2-agent setup:

  Agent 1 (Encoder): Receives a prompt containing target information (question,
      secret string, etc.) and performs m latent reasoning steps. Its KV-cache
      is the "message" sent to Agent 2.

  Agent 2 (Decoder): Receives Agent 1's KV-cache (full or partial) and must
      reproduce/recall the original information without ever seeing it in text.

Probe tasks (--probe_task):
  - question_recall : Agent 1 sees a GSM8K/ARC question; Agent 2 must repeat it verbatim.
  - secret_key      : Agent 1 sees a random alphanumeric string; Agent 2 must reproduce it.
  - question_answer : Agent 1 sees the question; Agent 2 must answer it (never sees question text).
  - reasoning_error : Agent 1 is given a WRONG solution; Agent 2 must detect the error.

KV pass modes (--kv_pass_mode):
  - full           : Agent 2 gets the entire KV from Agent 1 (prompt + latent steps).
  - latent_only    : Agent 2 only gets the KV entries from the m latent steps.
  - prompt_only    : Agent 2 only gets Agent 1's prompt KV (no latent steps).
  - none           : Agent 2 gets no KV; pure baseline (decoder must guess).

Usage:
  python run_info_preservation.py --model_name Qwen/Qwen3-4B \\
      --probe_task question_recall secret_key \\
      --latent_steps 0 10 20 40 80 \\
      --kv_pass_mode full latent_only prompt_only none \\
      --max_samples 50 --generate_bs 8 --latent_space_realign --temperature 0
"""

import argparse
import json
import os
import random
import string
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from data import load_gsm8k, load_arc_challenge
from models import ModelWrapper, _past_length
from utils import auto_device, set_seed, extract_gsm8k_answer, normalize_answer

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


# ---------------------------------------------------------------------------
# KV-cache utilities
# ---------------------------------------------------------------------------

def _slice_tensor(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice along the sequence dimension (dim -2)."""
    return tensor[..., start:end, :].contiguous()


def slice_past_kv(past_kv, start: int, end: int):
    """Extract KV entries from position [start, end) across all layers."""
    if past_kv is None:
        return None
    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
        sliced = tuple(
            tuple(_slice_tensor(t, start, end) for t in layer) for layer in legacy
        )
        return past_kv.__class__.from_legacy_cache(sliced)
    sliced_layers = []
    for layer in past_kv:
        if isinstance(layer, tuple):
            sliced_layers.append(tuple(_slice_tensor(t, start, end) for t in layer))
        elif torch.is_tensor(layer):
            sliced_layers.append(_slice_tensor(layer, start, end))
        else:
            sliced_layers.append(layer)
    return tuple(sliced_layers)


# ---------------------------------------------------------------------------
# Secret key generation
# ---------------------------------------------------------------------------

def generate_secret_key(rng: random.Random, length: int = 16) -> str:
    """Generate a random alphanumeric string of given length (hash-like, no spaces)."""
    chars = string.ascii_letters + string.digits
    return ''.join(rng.choice(chars) for _ in range(length))


# A pool of common English words for generating nonsense phrases.
_WORD_POOL = [
    "apple", "bridge", "castle", "dragon", "eagle", "forest", "guitar", "hammer",
    "island", "jungle", "kettle", "lemon", "marble", "needle", "ocean", "pencil",
    "quartz", "ribbon", "silver", "tunnel", "umbrella", "violet", "window", "yellow",
    "anchor", "basket", "candle", "desert", "engine", "falcon", "garden", "harbor",
    "insect", "jacket", "kitten", "ladder", "mirror", "napkin", "orange", "pillow",
    "rabbit", "saddle", "temple", "unique", "velvet", "walnut", "zigzag", "bottle",
    "copper", "donkey", "finger", "global", "helmet", "jigsaw", "lantern", "magnet",
    "notion", "oyster", "planet", "rocket", "socket", "tiger", "vacuum", "wisdom",
]


def generate_secret_string(rng: random.Random, num_words: int = 5) -> str:
    """Generate a random sequence of English words (non-meaningful phrase)."""
    return ' '.join(rng.choice(_WORD_POOL) for _ in range(num_words))


# ---------------------------------------------------------------------------
# Prompt builders for the 2-agent setup
# ---------------------------------------------------------------------------

def build_encoder_prompt_question_recall(question: str) -> List[Dict]:
    """Encoder agent receives the question and is told to reason about it."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            f"Read and carefully memorize the following question. Think about how to solve it.\n\n"
            f"Question: {question}\n\n"
            f"Reason step by step about this question."
        )},
    ]


def build_encoder_prompt_secret_key(secret: str) -> List[Dict]:
    """Encoder agent receives a secret key (hash-like) and is told to memorize it."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            f"IMPORTANT: Memorize the following secret key exactly. You will need to recall it later.\n\n"
            f"SECRET KEY: {secret}\n\n"
            f"Think carefully about this key and commit it to memory."
        )},
    ]


def build_encoder_prompt_secret_string(secret: str) -> List[Dict]:
    """Encoder agent receives a random word sequence and is told to memorize it."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            f"IMPORTANT: Memorize the following secret phrase exactly. You will need to recall it later.\n\n"
            f"SECRET PHRASE: {secret}\n\n"
            f"Think carefully about this phrase and commit every word to memory in order."
        )},
    ]


def build_encoder_prompt_question_answer(question: str) -> List[Dict]:
    """Encoder sees the question; same as recall but decoder will need to answer."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            f"Read the following question carefully and reason step by step to solve it.\n\n"
            f"Question: {question}\n\n"
            f"Think through the solution step by step."
        )},
    ]


def build_encoder_prompt_reasoning_error(question: str, wrong_solution: str) -> List[Dict]:
    """Encoder sees a question with a wrong solution baked in."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            f"A student attempted to solve the following question but made errors.\n\n"
            f"Question: {question}\n\n"
            f"Student's (WRONG) Solution:\n{wrong_solution}\n\n"
            f"Analyze this solution carefully and identify the errors."
        )},
    ]


def build_decoder_prompt_question_recall() -> List[Dict]:
    """Decoder must reproduce the question from latent KV only."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            "You have been provided with latent information from a previous agent. "
            "That agent was given a specific question to read.\n\n"
            "Your task: reproduce the EXACT question that the previous agent was given. "
            "Output ONLY the question text, nothing else. Do not add any explanation or reasoning."
        )},
    ]


def build_decoder_prompt_secret_key() -> List[Dict]:
    """Decoder must reproduce the secret key from latent KV only."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            "You have been provided with latent information from a previous agent. "
            "That agent was given a secret key to memorize.\n\n"
            "Your task: reproduce the EXACT secret key that the previous agent memorized. "
            "Output ONLY the secret key string, nothing else. No explanation, no reasoning, "
            "just the key."
        )},
    ]


def build_decoder_prompt_secret_string() -> List[Dict]:
    """Decoder must reproduce the random word phrase from latent KV only."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            "You have been provided with latent information from a previous agent. "
            "That agent was given a secret phrase (a sequence of random words) to memorize.\n\n"
            "Your task: reproduce the EXACT secret phrase that the previous agent memorized. "
            "Output ONLY the phrase (the words in order, separated by spaces), nothing else. "
            "No explanation, no reasoning, just the phrase."
        )},
    ]


def build_decoder_prompt_question_answer(task: str) -> List[Dict]:
    """Decoder must answer a question it never sees in text, only through latent KV."""
    if task in ["gsm8k"]:
        fmt = "Output your step-by-step reasoning and final answer inside \\boxed{YOUR_ANSWER}."
    elif task in ["arc_challenge", "arc_easy"]:
        fmt = "Output your reasoning and final answer (A, B, C, or D) inside \\boxed{YOUR_ANSWER}."
    else:
        fmt = "Output your reasoning and final answer inside \\boxed{YOUR_ANSWER}."
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            "You have been provided with latent information from a previous agent. "
            "That agent was reasoning about a specific question.\n\n"
            "Your task: based ONLY on the latent information, determine what the question is "
            "and provide the correct answer.\n\n"
            f"{fmt}"
        )},
    ]


def build_decoder_prompt_reasoning_error() -> List[Dict]:
    """Decoder must identify errors detected by the encoder."""
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": (
            "You have been provided with latent information from a previous agent. "
            "That agent analyzed a student's wrong solution to a math problem.\n\n"
            "Your task: based on the latent information, describe:\n"
            "1. What the original question was about\n"
            "2. What error(s) the student made\n"
            "3. What the correct answer should be\n\n"
            "Output your analysis clearly."
        )},
    ]


# ---------------------------------------------------------------------------
# Rendering & tokenization helpers
# ---------------------------------------------------------------------------

def render_messages(model: ModelWrapper, messages: List[Dict], enable_thinking: bool) -> str:
    tok = model.tokenizer
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def tokenize_prompts(model: ModelWrapper, prompts: List[str]):
    enc = model.tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    )
    return enc["input_ids"].to(model.device), enc["attention_mask"].to(model.device)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_token_overlap(prediction: str, reference: str) -> Dict[str, float]:
    """Compute token-level precision, recall, F1 (unigram overlap)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_set = set(pred_tokens)
    ref_set = set(ref_tokens)
    overlap = pred_set & ref_set

    precision = len(overlap) / len(pred_set) if pred_set else 0.0
    recall = len(overlap) / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def longest_common_substring_ratio(prediction: str, reference: str) -> float:
    """Ratio of longest common substring length to reference length."""
    if not reference:
        return 0.0
    s1 = prediction.lower()
    s2 = reference.lower()
    m, n = len(s1), len(s2)
    # Use rolling array for memory efficiency
    prev = [0] * (n + 1)
    longest = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                longest = max(longest, curr[j])
        prev = curr
    return round(longest / len(s2), 4)


def exact_match(prediction: str, reference: str) -> bool:
    """Checks if prediction contains the reference (case-insensitive)."""
    return reference.lower().strip() in prediction.lower().strip()


def evaluate_recall(prediction: str, reference: str) -> Dict[str, float]:
    """Full evaluation suite for recall tasks."""
    overlap = compute_token_overlap(prediction, reference)
    lcs_ratio = longest_common_substring_ratio(prediction, reference)
    em = exact_match(prediction, reference)
    return {
        "exact_match": em,
        "token_f1": overlap["f1"],
        "token_precision": overlap["precision"],
        "token_recall": overlap["recall"],
        "lcs_ratio": lcs_ratio,
    }


# ---------------------------------------------------------------------------
# Condition dataclass
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    probe_task: str          # question_recall | secret_key | secret_string | question_answer | reasoning_error
    latent_steps: int        # m
    kv_pass_mode: str        # full | latent_only
    secret_key_length: int   # only for secret_key task
    secret_word_count: int   # only for secret_string task
    decoder_thinking: bool   # whether decoder gets <think> mode
    realign: bool

    def tag(self) -> str:
        parts = [
            f"probe-{self.probe_task}",
            f"m{self.latent_steps}",
            f"kv-{self.kv_pass_mode}",
        ]
        if self.probe_task == "secret_key":
            parts.append(f"sklen{self.secret_key_length}")
        elif self.probe_task == "secret_string":
            parts.append(f"sw{self.secret_word_count}")
        parts.append(f"dec-{'think' if self.decoder_thinking else 'direct'}")
        parts.append(f"ra{'1' if self.realign else '0'}")
        return "_".join(parts)


# ---------------------------------------------------------------------------
# Core: 2-agent information preservation pipeline
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_preservation_batch(
    model: ModelWrapper,
    items: List[Dict],
    cond: Condition,
    task: str,
    temperature: float,
    top_p: float,
    rng: random.Random,
    max_new_tokens: int = 1024,
) -> List[Dict]:
    """
    Run the 2-agent information preservation probe on a batch.

    Steps:
      1. Build encoder prompts (Agent 1) based on probe_task
      2. Run encoder forward + latent steps → get full KV-cache
      3. Slice KV according to kv_pass_mode
      4. Build decoder prompts (Agent 2)
      5. Run decoder text generation conditioned on sliced KV
      6. Evaluate how much information was preserved
    """
    batch_size = len(items)

    # --- Step 1: Build encoder prompts ---
    encoder_messages_list = []
    references = []  # what the decoder should reproduce

    for item in items:
        if cond.probe_task == "question_recall":
            msgs = build_encoder_prompt_question_recall(item["question"])
            references.append(item["question"])
        elif cond.probe_task == "secret_key":
            secret = generate_secret_key(rng, cond.secret_key_length)
            msgs = build_encoder_prompt_secret_key(secret)
            references.append(secret)
            item["_secret"] = secret  # stash for logging
        elif cond.probe_task == "secret_string":
            secret = generate_secret_string(rng, cond.secret_word_count)
            msgs = build_encoder_prompt_secret_string(secret)
            references.append(secret)
            item["_secret"] = secret
        elif cond.probe_task == "question_answer":
            msgs = build_encoder_prompt_question_answer(item["question"])
            references.append(item.get("gold", ""))
        elif cond.probe_task == "reasoning_error":
            # Generate a plausible wrong answer for the encoder
            wrong = item.get("_wrong_solution", f"The answer is 999.")
            msgs = build_encoder_prompt_reasoning_error(item["question"], wrong)
            references.append(item.get("gold", ""))
        else:
            raise ValueError(f"Unknown probe_task: {cond.probe_task}")
        encoder_messages_list.append(msgs)

    # Render encoder prompts
    encoder_prompts = [
        render_messages(model, msgs, enable_thinking=False)
        for msgs in encoder_messages_list
    ]
    enc_ids, enc_mask = tokenize_prompts(model, encoder_prompts)

    # --- Step 2: Encoder forward + latent steps ---
    prompt_len = enc_ids.shape[1]  # after padding, all have same length

    if cond.latent_steps > 0:
        past_kv = model.generate_latent_batch(
            enc_ids,
            attention_mask=enc_mask,
            latent_steps=cond.latent_steps,
            past_key_values=None,
        )
    else:
        # m=0: just run a forward pass to get the prompt KV
        outputs = model.model(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True,
        )
        past_kv = outputs.past_key_values
        del outputs

    total_kv_len = _past_length(past_kv)

    # --- Step 3: Slice KV based on kv_pass_mode ---
    if cond.kv_pass_mode == "full":
        decoder_kv = past_kv
    elif cond.kv_pass_mode == "latent_only":
        # Only the latent step entries (last m positions)
        if cond.latent_steps > 0:
            start = total_kv_len - cond.latent_steps
            decoder_kv = slice_past_kv(past_kv, start, total_kv_len)
        else:
            decoder_kv = None
    elif cond.kv_pass_mode == "prompt_only":
        # Only the original prompt KV (first prompt_len positions)
        decoder_kv = slice_past_kv(past_kv, 0, prompt_len)
    elif cond.kv_pass_mode == "none":
        decoder_kv = None
    else:
        raise ValueError(f"Unknown kv_pass_mode: {cond.kv_pass_mode}")

    # Free the full KV if we sliced a subset
    if cond.kv_pass_mode not in ("full",):
        del past_kv

    # --- Step 4: Build decoder prompts ---
    if cond.probe_task == "question_recall":
        decoder_msgs = build_decoder_prompt_question_recall()
    elif cond.probe_task == "secret_key":
        decoder_msgs = build_decoder_prompt_secret_key()
    elif cond.probe_task == "secret_string":
        decoder_msgs = build_decoder_prompt_secret_string()
    elif cond.probe_task == "question_answer":
        decoder_msgs = build_decoder_prompt_question_answer(task)
    elif cond.probe_task == "reasoning_error":
        decoder_msgs = build_decoder_prompt_reasoning_error()
    else:
        raise ValueError(f"Unknown probe_task: {cond.probe_task}")

    # Same decoder prompt for all items in batch
    decoder_prompt = render_messages(model, decoder_msgs, enable_thinking=cond.decoder_thinking)
    decoder_prompts = [decoder_prompt] * batch_size
    dec_ids, dec_mask = tokenize_prompts(model, decoder_prompts)

    # --- Step 5: Decoder generates conditioned on encoder KV ---
    generated, _ = model.generate_text_batch(
        dec_ids,
        dec_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        past_key_values=decoder_kv,
        skip_special_tokens=False,
    )

    # Free decoder KV
    if decoder_kv is not None:
        del decoder_kv
    torch.cuda.empty_cache()

    # --- Step 6: Evaluate ---
    results = []
    for idx in range(batch_size):
        raw_output = generated[idx].strip()

        # Strip thinking tags if present
        if "</think>" in raw_output:
            raw_output_eval = raw_output[raw_output.rfind("</think>") + len("</think>"):].strip()
        else:
            raw_output_eval = raw_output

        ref = references[idx]

        if cond.probe_task in ("question_recall", "secret_key", "secret_string"):
            metrics = evaluate_recall(raw_output_eval, ref)
        elif cond.probe_task == "question_answer":
            pred = normalize_answer(extract_gsm8k_answer(raw_output))
            gold = ref
            metrics = {
                "exact_match": (pred == gold) if (pred and gold) else False,
                "prediction": pred,
            }
        elif cond.probe_task == "reasoning_error":
            # For error detection, check if gold answer appears in output
            gold = ref
            pred = normalize_answer(extract_gsm8k_answer(raw_output))
            metrics = {
                "correct_answer_found": (pred == gold) if (pred and gold) else False,
                "prediction": pred,
            }
        else:
            metrics = {}

        result = {
            "question": items[idx]["question"],
            "reference": ref,
            "raw_output": raw_output[:800],
            "output_eval": raw_output_eval[:500],
            "kv_total_len": total_kv_len,
            "kv_passed_mode": cond.kv_pass_mode,
            **metrics,
        }
        if cond.probe_task in ("secret_key", "secret_string"):
            result["secret_key"] = items[idx].get("_secret", "")

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Run one condition over full dataset
# ---------------------------------------------------------------------------

def run_condition(
    model: ModelWrapper,
    task: str,
    dataset: List[Dict],
    cond: Condition,
    generate_bs: int,
    temperature: float,
    top_p: float,
    rng: random.Random,
    max_new_tokens: int = 1024,
) -> Dict:
    all_results: List[Dict] = []
    t0 = time.time()

    for start in tqdm(range(0, len(dataset), generate_bs),
                      desc=f"[{task}|{cond.tag()}]", leave=False):
        batch = dataset[start:start + generate_bs]
        batch_results = run_preservation_batch(
            model, batch, cond, task, temperature, top_p, rng, max_new_tokens
        )
        all_results.extend(batch_results)

    elapsed = time.time() - t0
    n = len(all_results)

    # Aggregate metrics
    if cond.probe_task in ("question_recall", "secret_key", "secret_string"):
        em_count = sum(1 for r in all_results if r.get("exact_match", False))
        avg_f1 = sum(r.get("token_f1", 0.0) for r in all_results) / n if n else 0.0
        avg_lcs = sum(r.get("lcs_ratio", 0.0) for r in all_results) / n if n else 0.0
        avg_recall = sum(r.get("token_recall", 0.0) for r in all_results) / n if n else 0.0
        summary = {
            "exact_match_rate": round(em_count / n, 4) if n else 0.0,
            "avg_token_f1": round(avg_f1, 4),
            "avg_token_recall": round(avg_recall, 4),
            "avg_lcs_ratio": round(avg_lcs, 4),
        }
    elif cond.probe_task == "question_answer":
        correct = sum(1 for r in all_results if r.get("exact_match", False))
        summary = {"accuracy": round(correct / n, 4) if n else 0.0, "correct": correct}
    elif cond.probe_task == "reasoning_error":
        correct = sum(1 for r in all_results if r.get("correct_answer_found", False))
        summary = {"error_detection_rate": round(correct / n, 4) if n else 0.0, "correct": correct}
    else:
        summary = {}

    summary.update({
        "task": task,
        "condition": cond.tag(),
        "probe_task": cond.probe_task,
        "latent_steps": cond.latent_steps,
        "kv_pass_mode": cond.kv_pass_mode,
        "decoder_thinking": cond.decoder_thinking,
        "realign": cond.realign,
        "n": n,
        "total_time_sec": round(elapsed, 2),
        "time_per_sample_sec": round(elapsed / n, 3) if n else 0.0,
    })
    if cond.probe_task == "secret_key":
        summary["secret_key_length"] = cond.secret_key_length
    if cond.probe_task == "secret_string":
        summary["secret_word_count"] = cond.secret_word_count

    return {"summary": summary, "results": all_results}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_for_probe(task: str, max_samples: int) -> List[Dict]:
    if task == "gsm8k":
        data = list(load_gsm8k(split="test"))
    elif task == "arc_challenge":
        data = list(load_arc_challenge(split="test"))
    else:
        raise ValueError(f"Unsupported task for info preservation probe: {task}")
    if max_samples > 0:
        data = data[:max_samples]
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Information Preservation Probe for Latent Communication")

    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B")
    p.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "arc_challenge"])
    p.add_argument("--max_samples", type=int, default=50)

    # Swept axes
    p.add_argument("--probe_task", nargs="+",
                   choices=["question_recall", "secret_key", "secret_string", "question_answer", "reasoning_error"],
                   default=["question_recall", "secret_key", "secret_string"])
    p.add_argument("--latent_steps", type=int, nargs="+", default=[0, 10, 20, 40, 80])
    p.add_argument("--kv_pass_mode", nargs="+",
                   choices=["full", "latent_only"],
                   default=["full", "latent_only"])
    p.add_argument("--secret_key_length", type=int, nargs="+", default=[8, 16, 32],
                   help="Length of random secret keys in chars (only for secret_key probe)")
    p.add_argument("--secret_word_count", type=int, nargs="+", default=[3, 5, 8],
                   help="Number of random words in secret phrase (only for secret_string probe)")
    p.add_argument("--decoder_thinking", choices=["on", "off"], default="on",
                   help="Whether decoder agent gets thinking mode (CoT)")

    # Fixed knobs
    p.add_argument("--latent_space_realign", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Default 0 (greedy) for deterministic recall evaluation")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--generate_bs", type=int, default=8)
    p.add_argument("--output_dir", type=str, default=None)

    args = p.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = auto_device(args.device)

    # ModelWrapper construction args
    args.method = "latent_mas"
    args.use_vllm = False
    args.think = False

    if args.output_dir is None:
        model_short = args.model_name.replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("output_info_preservation", model_short, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metadata
    import sys
    meta = {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Loading {args.model_name} on {device} (realign={args.latent_space_realign}) ...")
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    print("Model loaded.\n")

    dataset = load_dataset_for_probe(args.task, args.max_samples)
    print(f"Dataset: {args.task}, {len(dataset)} samples\n")

    decoder_thinking = (args.decoder_thinking == "on")
    all_summaries: List[Dict] = []

    for probe_task in args.probe_task:
        # For secret_key, sweep key lengths; for secret_string, sweep word counts
        if probe_task == "secret_key":
            length_sweep = args.secret_key_length
        elif probe_task == "secret_string":
            length_sweep = args.secret_word_count
        else:
            length_sweep = [0]  # dummy, single pass

        for length_val in length_sweep:
            for m in args.latent_steps:
                for kv_mode in args.kv_pass_mode:
                    # Skip nonsensical combinations
                    if m == 0 and kv_mode == "latent_only":
                        continue  # no latent steps → nothing to pass

                    cond = Condition(
                        probe_task=probe_task,
                        latent_steps=m,
                        kv_pass_mode=kv_mode,
                        secret_key_length=length_val if probe_task == "secret_key" else 16,
                        secret_word_count=length_val if probe_task == "secret_string" else 5,
                        decoder_thinking=decoder_thinking,
                        realign=args.latent_space_realign,
                    )

                    out = run_condition(
                        model, args.task, dataset, cond,
                        args.generate_bs, args.temperature, args.top_p,
                        rng, args.max_new_tokens,
                    )
                    s = out["summary"]
                    all_summaries.append(s)

                    # Print headline
                    headline_parts = [f"[{args.task}] {cond.tag()}"]
                    if probe_task in ("question_recall", "secret_key", "secret_string"):
                        headline_parts.append(
                            f"EM={s['exact_match_rate']:.3f} F1={s['avg_token_f1']:.3f} "
                            f"Recall={s['avg_token_recall']:.3f} LCS={s['avg_lcs_ratio']:.3f}"
                        )
                    elif probe_task == "question_answer":
                        headline_parts.append(f"Acc={s['accuracy']:.4f}")
                    elif probe_task == "reasoning_error":
                        headline_parts.append(f"ErrDet={s['error_detection_rate']:.4f}")
                    headline_parts.append(f"{s['time_per_sample_sec']}s/it")
                    print(" | ".join(headline_parts))

                    # Save per-condition results
                    cond_path = os.path.join(args.output_dir, f"{args.task}__{cond.tag()}.json")
                    with open(cond_path, "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False, indent=2)

            # Break out of length_sweep loop for tasks that don't sweep lengths
            if probe_task not in ("secret_key", "secret_string"):
                break

    # Save combined summaries
    with open(os.path.join(args.output_dir, "summaries.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    for s in all_summaries:
        line = f"{s['condition']:60s}"
        if s["probe_task"] in ("question_recall", "secret_key", "secret_string"):
            line += f" EM={s['exact_match_rate']:.3f} F1={s['avg_token_f1']:.3f} LCS={s['avg_lcs_ratio']:.3f}"
        elif s["probe_task"] == "question_answer":
            line += f" Acc={s['accuracy']:.4f}"
        elif s["probe_task"] == "reasoning_error":
            line += f" ErrDet={s['error_detection_rate']:.4f}"
        print(line)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
