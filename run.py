import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
from crossrare_data import load_crossrare
from methods.baseline import BaselineMethod
from methods.latent_mas import LatentMASMethod
from methods.text_mas import TextMASMethod
from methods.raredisease_mas import RarediseaseMASMethod
from methods.medlatentdx_h import MedLatentDxHMethod
from models import ModelWrapper
from utils import auto_device, set_seed
import time

_CROSSRARE_SOURCES = ("rarebench", "zenodo", "phenopackets26")


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def _normalize_label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _macro_f1(preds: List[Dict]) -> float:
    """Macro-F1 over disease classes, preserving alias-aware correct matches."""
    if not preds:
        return 0.0

    true_labels = [_normalize_label(p.get("gold")) for p in preds]
    labels = sorted(set(true_labels))
    alias_to_label = {}
    for pred, true_label in zip(preds, true_labels):
        for alias in pred.get("gold_aliases", [pred.get("gold", "")]):
            alias_to_label.setdefault(_normalize_label(alias), true_label)

    predicted_labels = []
    for pred, true_label in zip(preds, true_labels):
        if pred.get("correct", False):
            predicted_labels.append(true_label)
        else:
            predicted_labels.append(
                alias_to_label.get(_normalize_label(pred.get("prediction")), "__unknown__")
            )

    f1_scores = []
    for label in labels:
        tp = sum(y_true == label and y_pred == label for y_true, y_pred in zip(true_labels, predicted_labels))
        fp = sum(y_true != label and y_pred == label for y_true, y_pred in zip(true_labels, predicted_labels))
        fn = sum(y_true == label and y_pred != label for y_true, y_pred in zip(true_labels, predicted_labels))
        denominator = 2 * tp + fp + fn
        f1_scores.append((2 * tp / denominator) if denominator else 0.0)
    return sum(f1_scores) / len(f1_scores)


def evaluate_crossrare(preds: List[Dict]) -> Dict[str, Dict[str, float | int]]:
    """Calculate metrics overall and for each fixed CrossRare source cohort."""
    cohorts = {"overall": preds}
    for source in _CROSSRARE_SOURCES:
        cohorts[source] = [p for p in preds if p.get("source") == source]

    metrics = {}
    for name, cohort in cohorts.items():
        accuracy, correct = evaluate(cohort)
        metrics[name] = {
            "accuracy": accuracy,
            "macro_f1": _macro_f1(cohort),
            "correct": correct,
            "total": len(cohort),
        }
    return metrics


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "result"


def save_results(preds: List[Dict], args: argparse.Namespace, metrics: Dict[str, Dict[str, float | int]], total_time: float) -> None:
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    model_name = _safe_name(args.model_name)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"{args.method}_{model_name}_{ts}_results.json")
    summary_path = os.path.join(output_dir, f"{args.method}_{model_name}_{ts}_summary.json")

    rows = []
    for item in preds:
        rows.append(
            {
                "id": item.get("id", ""),
                "source": item.get("source", ""),
                "ground_truth": item.get("gold", ""),
                "predicted_text": item.get("prediction", ""),
                "correct": bool(item.get("correct", False)),
                "raw_prediction": item.get("raw_prediction", ""),
                "question": item.get("question", ""),
            }
        )

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    summary = {
        "method": args.method,
        "model": args.model_name,
        "task": args.task,
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "metrics": metrics,
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": round(total_time / max(len(preds), 1), 4),
        "output_dir": output_dir,
        "results_json": results_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Results] Saved {len(rows)} rows to {results_path}")
    print(f"[Results] Summary saved to {summary_path}")


# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument("--method", choices=["baseline", "text_mas", "latent_mas", "raredisease_mas", "medlatentdx_h"], required=True,
                        help="Which multi-agent method to run.")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-4B-Instruct-2507"],
                        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa', 'crossrare'], default="gsm8k",
                        help="Dataset/task to evaluate.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/run_results", help="Directory to save JSON evaluation outputs")

    # CrossRare / raredisease_mas specific args
    parser.add_argument("--num_hospitals", type=int, default=5, help="Number of simulated hospital databases for CrossRare")
    parser.add_argument("--hospital_agents", type=int, default=3, help="Hospital agents selected per CrossRare query")
    parser.add_argument("--agent_hospital_ids", type=int, nargs=3, default=[1, 2, 3],
                        help="One-based IDs of the three CrossRare retrieval hospitals")
    parser.add_argument("--retrieval_top_k", type=int, default=1, help="Top-k cases retrieved per hospital")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Fraction of CrossRare data held out for test")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of CrossRare data held out for validation")
    parser.add_argument("--partition_strategy", type=str, default="random",
                        choices=["random", "round_robin"],
                        help="How to distribute train cases across hospitals")
    parser.add_argument("--distiller_checkpoint", type=str, default=None,
                        help="Trained MedLatentDx-H distiller checkpoint.")
    parser.add_argument("--max_prompt_length", type=int, default=320,
                        help="Local and host prompt token limit for MedLatentDx-H.")

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")

    args = parser.parse_args()
    
    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True 
        args.enable_prefix_caching = True
    
    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    
    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        method = TextMASMethod(
            model,
            max_new_tokens_each=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )
    elif args.method == 'raredisease_mas':
        method = RarediseaseMASMethod(
            model,
            latent_steps=args.latent_steps,
            host_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == 'medlatentdx_h':
        if args.task != "crossrare" or not args.distiller_checkpoint:
            parser.error("medlatentdx_h requires --task crossrare and --distiller_checkpoint")
        if (args.num_hospitals, args.hospital_agents, args.retrieval_top_k) != (5, 3, 1):
            parser.error("medlatentdx_h reproduction fixes five hospital partitions, three agents/query, and top-1 retrieval")
        method = MedLatentDxHMethod.from_checkpoint(
            model, args.distiller_checkpoint, latent_steps=args.latent_steps,
            max_prompt_length=args.max_prompt_length,
            host_max_new_tokens=args.max_new_tokens, **common_kwargs,
            generate_bs=args.generate_bs, args=args,
        )

    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []
    
    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    elif args.task == "crossrare":
        dataset_iter = load_crossrare(
            num_hospitals=args.num_hospitals,
            num_active_hospitals=args.hospital_agents,
            agent_hospital_ids=args.agent_hospital_ids,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            top_k=args.retrieval_top_k,
            seed=args.seed,
            partition_strategy=args.partition_strategy,
        )
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    progress = tqdm(total=args.max_samples)

    for item in dataset_iter:
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                args.max_samples,
                args,
            )
            batch = []
            if processed >= args.max_samples:
                break

    if batch and processed < args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=args.max_samples,
            args=args,
        )
    progress.close()
    
    total_time = time.time() - start_time

    if args.task == "crossrare":
        metrics = evaluate_crossrare(preds)
    else:
        acc, correct = evaluate(preds)
        metrics = {"overall": {"accuracy": acc, "correct": correct, "total": len(preds)}}
    save_results(preds, args, metrics, total_time)
    
    # Load results in JSON format
    print(
        json.dumps(
            {
                "method": args.method,
                "model": args.model_name,
                "split": args.split,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "metrics": metrics,
                "total_time_sec": round(total_time,4),
                "time_per_sample_sec": round(total_time / max(args.max_samples, 1), 4),
            },
            ensure_ascii=False,
        )
    )



if __name__ == "__main__":
    main()
