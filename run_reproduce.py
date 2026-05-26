"""
Reproduce LatentMAS results with Qwen3-8B on lightweight tasks.
Saves detailed logs and per-sample KV-cache to disk.

Follows paper implementation details:
- Temperature 0.6, top_p 0.95
- Per-task max output length: 2048 (gsm8k, arc), 4096 (medqa, mbpp+, humaneval+),
  8192 (gpqa), 20000 (aime24/25)
- Realignment matrix computed once per run

Usage:
    python run_reproduce.py --max_samples 30
    python run_reproduce.py --max_samples 10 --tasks gsm8k arc_easy
    python run_reproduce.py --max_samples 5 --save_kv_cache --latent_steps 10

Outputs saved to: ./reproduce_outputs/<model_short>/ls<latent_steps>/
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple, Optional

import torch
from tqdm import tqdm

from data import load_gsm8k, load_arc_easy, load_arc_challenge
from methods import default_agents, Agent
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas
from utils import (
    auto_device,
    set_seed,
    extract_gsm8k_answer,
    normalize_answer,
)


# ---------------------------------------------------------------------------
# Per-task max_new_tokens (from paper)
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


# ---------------------------------------------------------------------------
# KV-cache saving utility
# ---------------------------------------------------------------------------

def save_kv_cache(past_kv, save_path: str):
    """Save KV-cache to disk. Handles both DynamicCache and legacy tuple format."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert DynamicCache to serializable format
    try:
        from transformers.cache_utils import Cache
        if isinstance(past_kv, Cache):
            # Use legacy format for serialization
            if hasattr(past_kv, "to_legacy_cache"):
                past_kv = past_kv.to_legacy_cache()
            elif hasattr(past_kv, "key_cache"):
                past_kv = tuple(
                    (k.cpu(), v.cpu())
                    for k, v in zip(past_kv.key_cache, past_kv.value_cache)
                )
    except ImportError:
        pass

    # Convert to CPU tensors for saving
    if isinstance(past_kv, (tuple, list)):
        serializable = tuple(
            tuple(t.cpu().half() for t in layer) if isinstance(layer, (tuple, list)) else layer
            for layer in past_kv
        )
    else:
        serializable = past_kv

    torch.save(serializable, save_path)


# ---------------------------------------------------------------------------
# Single-sample LatentMAS runner with logging
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_latent_mas_single(
    model: ModelWrapper,
    item: Dict,
    agents: List[Agent],
    latent_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    args,
) -> Tuple[Dict, Optional[object]]:
    """Run LatentMAS pipeline on a single item. Returns (result_dict, final_kv_cache)."""

    past_kv = None
    agent_logs = []

    for agent in agents:
        messages = build_agent_message_sequential_latent_mas(
            role=agent.role,
            question=item["question"],
            context="",
            method="latent_mas",
            args=args,
        )
        prompts, input_ids, attention_mask, tokens_batch = model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        prompt_text = prompts[0]

        if agent.role != "judger":
            # Latent generation
            past_kv = model.generate_latent_batch(
                input_ids,
                attention_mask=attention_mask,
                latent_steps=latent_steps,
                past_key_values=past_kv,
            )
            kv_len = _past_length(past_kv)
            agent_logs.append({
                "name": agent.name,
                "role": agent.role,
                "prompt": prompt_text,
                "latent_steps": latent_steps,
                "kv_cache_length_after": kv_len,
                "output": "(latent — no text output)",
            })
        else:
            # Judger: text generation conditioned on accumulated KV-cache
            past_for_decoding = past_kv if latent_steps > 0 else None
            generated_batch, _ = model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                past_key_values=past_for_decoding,
            )
            output_text = generated_batch[0].strip()
            agent_logs.append({
                "name": agent.name,
                "role": agent.role,
                "prompt": prompt_text,
                "output": output_text,
            })

    # Extract answer
    final_text = agent_logs[-1]["output"]
    pred = normalize_answer(extract_gsm8k_answer(final_text))
    gold = item.get("gold", "")
    ok = (pred == gold) if (pred and gold) else False

    result = {
        "question": item["question"],
        "gold": gold,
        "solution": item.get("solution", ""),
        "prediction": pred,
        "raw_output": final_text,
        "correct": ok,
        "agents": agent_logs,
    }
    return result, past_kv


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    task_name: str,
    model: ModelWrapper,
    args,
    output_dir: str,
):
    """Run LatentMAS on a single task, save logs and optionally KV-cache."""

    # Load dataset
    if task_name == "gsm8k":
        dataset_iter = load_gsm8k(split="test")
    elif task_name == "arc_easy":
        dataset_iter = load_arc_easy(split="test")
    elif task_name == "arc_challenge":
        dataset_iter = load_arc_challenge(split="test")
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    dataset = list(dataset_iter)
    if args.max_samples > 0:
        dataset = dataset[: args.max_samples]

    agents = default_agents()
    task_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    if args.save_kv_cache:
        kv_dir = os.path.join(task_dir, "kv_cache")
        os.makedirs(kv_dir, exist_ok=True)

    results = []
    total_time = 0.0

    print(f"\n{'='*60}")
    print(f"  Task: {task_name} | Samples: {len(dataset)} | Latent Steps: {args.latent_steps}")
    print(f"{'='*60}")

    # Per-task max_new_tokens from paper (can be overridden by CLI)
    task_max_tokens = TASK_MAX_TOKENS.get(task_name, 2048)
    if args.max_new_tokens_override:
        task_max_tokens = args.max_new_tokens_override
    print(f"  max_new_tokens: {task_max_tokens}")

    for idx, item in enumerate(tqdm(dataset, desc=f"[{task_name}]")):
        t0 = time.time()
        result, final_kv = run_latent_mas_single(
            model=model,
            item=item,
            agents=agents,
            latent_steps=args.latent_steps,
            max_new_tokens=task_max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            args=args,
        )
        elapsed = time.time() - t0
        total_time += elapsed
        result["time_sec"] = round(elapsed, 2)
        result["sample_idx"] = idx
        results.append(result)

        # Save KV-cache per sample
        if args.save_kv_cache and final_kv is not None:
            kv_path = os.path.join(kv_dir, f"sample_{idx:04d}.pt")
            save_kv_cache(final_kv, kv_path)

        # Print progress
        status = "✓" if result["correct"] else "✗"
        print(f"  [{status}] #{idx} | Pred: {result['prediction']} | Gold: {result['gold']} | {elapsed:.1f}s")

    # Compute metrics
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / len(results) if results else 0.0

    summary = {
        "task": task_name,
        "model": args.model_name,
        "method": "latent_mas",
        "prompt": "sequential",
        "latent_steps": args.latent_steps,
        "latent_space_realign": args.latent_space_realign,
        "max_new_tokens": task_max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "total_samples": len(results),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "total_time_sec": round(total_time, 2),
        "avg_time_per_sample_sec": round(total_time / len(results), 2) if results else 0,
    }

    # Save results
    results_path = os.path.join(task_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(task_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save readable log
    log_path = os.path.join(task_dir, "log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Task: {task_name}\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Latent Steps: {args.latent_steps}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({correct}/{len(results)})\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"--- Sample #{r['sample_idx']} ---\n")
            f.write(f"Question: {r['question'][:200]}...\n")
            f.write(f"Gold: {r['gold']}\n")
            f.write(f"Pred: {r['prediction']}\n")
            f.write(f"Correct: {r['correct']}\n")
            f.write(f"Time: {r['time_sec']}s\n")
            f.write(f"\n--- Agent Traces ---\n")
            for agent_log in r["agents"]:
                f.write(f"  [{agent_log['role']}]\n")
                if agent_log["role"] != "judger":
                    f.write(f"    Latent Steps: {agent_log.get('latent_steps', 0)}\n")
                    f.write(f"    KV Length After: {agent_log.get('kv_cache_length_after', '?')}\n")
                else:
                    output_preview = agent_log["output"][:500]
                    f.write(f"    Output: {output_preview}\n")
            f.write("\n")

    print(f"\n  Summary: {accuracy:.2%} ({correct}/{len(results)})")
    print(f"  Saved to: {task_dir}/")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reproduce LatentMAS with Qwen3-8B")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model ID")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "arc_easy"],
                        choices=["gsm8k", "arc_easy", "arc_challenge"],
                        help="Tasks to evaluate")
    parser.add_argument("--max_samples", type=int, default=30,
                        help="Max samples per task (-1 for all)")
    parser.add_argument("--latent_steps", type=int, default=10,
                        help="Number of latent reasoning steps per agent")
    parser.add_argument("--latent_space_realign", action="store_true",
                        help="Enable latent-space realignment matrix (paper default)")
    parser.add_argument("--max_new_tokens", type=int, default=None, dest="max_new_tokens_override",
                        help="Override per-task max_new_tokens (default: use paper values)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (paper: 0.6)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling (paper: 0.95)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_kv_cache", action="store_true",
                        help="Save KV-cache tensors per sample (warning: large files)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ./reproduce_outputs/<model>/)")

    args = parser.parse_args()

    # Add fields expected by prompts.py and models.py
    args.task = None  # set per-task
    args.prompt = "sequential"
    args.method = "latent_mas"
    args.think = False

    set_seed(args.seed)
    device = auto_device(args.device)

    # Determine output directory
    if args.output_dir is None:
        model_short = args.model_name.replace("/", "_")
        realign_tag = "_realign" if args.latent_space_realign else ""
        args.output_dir = os.path.join(
            "reproduce_outputs", model_short, f"ls{args.latent_steps}{realign_tag}"
        )

    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Tasks: {args.tasks}")
    print(f"Max samples: {args.max_samples}")
    print(f"Latent steps: {args.latent_steps}")
    print(f"Realign: {args.latent_space_realign}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output_dir}")
    print()

    # Load model
    print("Loading model...")
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=args)
    print(f"Model loaded. Device: {model.device}")
    print()

    # Run tasks
    all_summaries = []
    for task_name in args.tasks:
        args.task = task_name
        summary = run_task(task_name, model, args, args.output_dir)
        all_summaries.append(summary)

    # Final report
    print(f"\n{'='*60}")
    print("  FINAL REPORT")
    print(f"{'='*60}")
    for s in all_summaries:
        print(f"  {s['task']:15s} | Acc: {s['accuracy']:.4f} ({s['correct']}/{s['total_samples']}) | {s['avg_time_per_sample_sec']:.1f}s/sample")
    print(f"{'='*60}")

    # Save combined summary
    combined_path = os.path.join(args.output_dir, "all_summaries.json")
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\n  Combined summary: {combined_path}")


if __name__ == "__main__":
    main()
