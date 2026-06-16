import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

from data import (
    load_arc_challenge,
    load_arc_easy,
    load_aime2024,
    load_aime2025,
    load_gsm8k,
)
from models import ModelWrapper, _positions_from_mask
from prompts import build_agent_message_sequential_latent_mas
from utils import auto_device, set_seed


@dataclass
class StepStat:
    step: int
    mode: str
    feed_norm: float
    hidden_norm: float
    cos_to_prev_hidden: float
    cos_to_init_hidden: float
    cos_feed_to_next_hidden: float
    top1_token: str
    top1_logit: float
    top_tokens: List[Dict]


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return 0.0
    return torch.dot(a, b).div(denom).item()


def _topk_tokens(model: ModelWrapper, logits: torch.Tensor, top_k: int) -> List[Dict]:
    k = min(top_k, logits.shape[-1])
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    top_logits = torch.gather(logits.float(), dim=-1, index=top_ids)
    rows: List[Dict] = []
    for token_id, prob, logit in zip(top_ids[0], top_probs[0], top_logits[0]):
        tid = int(token_id.item())
        rows.append(
            {
                "token_id": tid,
                "token": model.tokenizer.decode([tid]).replace("\n", "\\n"),
                "prob": float(prob.item()),
                "logit": float(logit.item()),
            }
        )
    return rows


def _load_task_item(task: str, split: str, index: int) -> Dict:
    if task == "gsm8k":
        items = list(load_gsm8k(split=split))
    elif task == "aime2024":
        items = list(load_aime2024(split=split))
    elif task == "aime2025":
        items = list(load_aime2025(split=split))
    elif task == "arc_easy":
        items = list(load_arc_easy(split=split))
    elif task == "arc_challenge":
        items = list(load_arc_challenge(split=split))
    else:
        raise ValueError(f"Unsupported task: {task}")

    if not items:
        raise ValueError(f"No items found for task={task}, split={split}")
    return items[index % len(items)]


def _build_raw_realign_matrix(model: torch.nn.Module, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output_embeds is None:
        output_embeds = getattr(model, "lm_head", None)
    if (
        input_embeds is None
        or output_embeds is None
        or not hasattr(input_embeds, "weight")
        or not hasattr(output_embeds, "weight")
    ):
        raise RuntimeError("Cannot build realign matrix: embedding weights not accessible.")

    input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
    output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

    gram = output_weight.T @ output_weight
    reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    gram = gram + reg
    rhs = output_weight.T @ input_weight
    matrix = torch.linalg.solve(gram, rhs)
    target_norm = input_weight.norm(dim=1).mean().detach()
    return matrix, target_norm


def _matrix_stats(matrix: torch.Tensor, *, include_spectrum: bool = False) -> Dict:
    matrix_f32 = matrix.detach().float().cpu()
    sv = torch.linalg.svdvals(matrix_f32)
    identity = torch.eye(matrix_f32.shape[0], dtype=matrix_f32.dtype)
    diff = matrix_f32 - identity
    out = {
        "shape": list(matrix_f32.shape),
        "fro_norm": matrix_f32.norm().item(),
        "fro_norm_to_identity": diff.norm().item(),
        "spectral_norm": sv.max().item(),
        "min_singular": sv.min().item(),
        "median_singular": sv.median().item(),
        "max_singular": sv.max().item(),
        "condition_number": (sv.max() / sv.min().clamp_min(1e-12)).item(),
    }
    if include_spectrum:
        out["singular_values"] = sv.tolist()
    return out


def _short_token(token: str) -> str:
    token = token.encode("unicode_escape").decode("ascii")
    token = token.replace(r"\n", "\\n").replace(r"\t", "\\t")
    if len(token) > 16:
        return token[:13] + "..."
    return token


def _display_mode_name(mode_name: str, rollout_name: str) -> str:
    if mode_name == "normal":
        return "Greedy decoding"
    if mode_name == "latent_on":
        return "Latent reasoning, realignment ON"
    if mode_name == "latent_off":
        return "Latent reasoning, realignment OFF"
    return rollout_name


def _plot_mode_chart(mode_name: str, rollout_name: str, rows: List[Dict], out_path: Path, *, plot_steps: int) -> Optional[Path]:
    if plt is None:
        print("matplotlib is not available; skipping plots.")
        return None

    if not rows:
        return None

    base = out_path.with_suffix("")
    chart_path = base.with_name(f"{base.name}_{mode_name}.png")

    subset = rows[: min(plot_steps, len(rows))]

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.15, 1.8], hspace=0.45)

    ax_norm = fig.add_subplot(gs[0, 0])
    ax_norm_r = ax_norm.twinx()
    ax_cos = fig.add_subplot(gs[1, 0])
    ax_tokens = fig.add_subplot(gs[2, 0])

    xs = [row["step"] for row in subset]
    color_hidden = "#2a6fdb"
    color_feed = "#d95f02"
    color_prev = "#6a3d9a"
    color_next = "#1b9e77"
    display_name = _display_mode_name(mode_name, rollout_name)

    ax_norm.plot(xs, [row["hidden_norm"] for row in subset], color=color_hidden, linestyle="-", linewidth=2.2, marker="o", markersize=3, label="output hidden-state norm")
    ax_norm_r.plot(xs, [row["feed_norm"] for row in subset], color=color_feed, linestyle="-", linewidth=1.9, alpha=0.9, label="input embedding norm")
    ax_cos.plot(xs, [row["cos_to_prev_hidden"] for row in subset], color=color_prev, linestyle="-", linewidth=1.8, alpha=0.9, label="cos(input embedding, previous hidden state)")
    ax_cos.plot(xs, [row["cos_feed_to_next_hidden"] for row in subset], color=color_next, linestyle="-", linewidth=1.8, alpha=0.9, label="cos(input embedding, next hidden state)")

    ax_norm.set_title(f"{display_name} | norms")
    ax_norm.set_xlabel("Step")
    ax_norm.set_ylabel("Output hidden-state norm")
    ax_norm_r.set_ylabel("Input embedding norm")
    ax_norm.grid(True, alpha=0.25)
    ax_norm.set_xlim(1, max(1, len(subset)))

    handles1, labels1 = ax_norm.get_legend_handles_labels()
    handles2, labels2 = ax_norm_r.get_legend_handles_labels()
    ax_norm.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8, ncol=2)

    ax_cos.set_title(f"{display_name} | cosine trends")
    ax_cos.set_xlabel("Step")
    ax_cos.set_ylabel("Cosine similarity")
    ax_cos.grid(True, alpha=0.25)
    ax_cos.set_xlim(1, max(1, len(subset)))
    handles1, labels1 = ax_cos.get_legend_handles_labels()
    if handles1:
        ax_cos.legend(handles1, labels1, loc="upper right", fontsize=8, ncol=2)

    probs = torch.tensor([[token["prob"] for token in row["top_tokens"][:3]] for row in subset], dtype=torch.float32).T
    probs_np = probs.numpy()
    im = ax_tokens.imshow(probs_np, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.05, float(probs_np.max())))
    ax_tokens.set_title(f"{display_name} | top-3 decoded tokens by step")
    ax_tokens.set_ylabel("Rank")
    ax_tokens.set_yticks([0, 1, 2])
    ax_tokens.set_yticklabels(["1", "2", "3"])
    ax_tokens.set_xticks(range(len(subset)))
    ax_tokens.set_xticklabels([str(row["step"]) for row in subset], rotation=0, fontsize=8)
    ax_tokens.set_xlabel("Step")

    for col, row in enumerate(subset):
        for rank in range(min(3, len(row["top_tokens"]))):
            tok = row["top_tokens"][rank]
            ax_tokens.text(
                col,
                rank,
                f"{_short_token(tok['token'])}\n{tok['prob']:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if tok["prob"] < 0.5 else "black",
            )

    cbar = fig.colorbar(im, ax=ax_tokens, fraction=0.03, pad=0.02)
    cbar.set_label("Probability")

    fig.suptitle(
        f"{display_name} | {out_path.stem}",
        fontsize=13,
    )
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def _plot_report(report: Dict, out_path: Path) -> List[Path]:
    if plt is None:
        print("matplotlib is not available; skipping plots.")
        return []

    saved_paths: List[Path] = []
    rollouts = report.get("rollouts", {})
    if not rollouts:
        return saved_paths

    base = out_path.with_suffix("")

    for key, rows in rollouts.items():
        if key.startswith("latent_"):
            mode_name = "latent_on" if key.endswith("_on") else "latent_off"
            p = _plot_mode_chart(mode_name, key, rows, out_path, plot_steps=report.get("plot_steps", 20))
            if p is not None:
                saved_paths.append(p)
        elif key.startswith("normal_"):
            p = _plot_mode_chart("normal", key, rows, out_path, plot_steps=report.get("plot_steps", 20))
            if p is not None:
                saved_paths.append(p)

    matrix_stats = report.get("matrix_stats")
    if matrix_stats and matrix_stats.get("shape") and matrix_stats["shape"][0] > 0:
        spectrum_path = base.with_name(base.name + "_spectrum.png")
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        sv = matrix_stats.get("singular_values")
        if sv:
            ax.plot(range(1, len(sv) + 1), sv, linewidth=1.5)
            ax.set_yscale("log")
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Singular value (log scale)")
            ax.set_title("Realignment matrix spectrum")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(spectrum_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(spectrum_path)

    return saved_paths


@torch.no_grad()
def _rollout_latent(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    latent_steps: int,
    apply_realign: bool,
    matrix: torch.Tensor,
    target_norm: torch.Tensor,
    top_k: int,
) -> List[StepStat]:
    hf_model = model.model
    device = input_ids.device

    outputs = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=_positions_from_mask(attention_mask, input_ids.shape[1]),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    init_hidden = outputs.hidden_states[-1][:, -1, :]
    prev_hidden = init_hidden
    full_mask = attention_mask
    stats: List[StepStat] = []

    for step in range(1, latent_steps + 1):
        if apply_realign:
            latent = prev_hidden.float() @ matrix
            latent_norm = latent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            latent = latent * (target_norm.to(device=device, dtype=latent.dtype) / latent_norm)
        else:
            latent = prev_hidden.float()
        latent_embed = latent.to(dtype=prev_hidden.dtype)

        full_mask = torch.cat(
            [
                full_mask,
                torch.ones((full_mask.shape[0], 1), dtype=full_mask.dtype, device=device),
            ],
            dim=-1,
        )

        outputs = hf_model(
            inputs_embeds=latent_embed.unsqueeze(1),
            attention_mask=full_mask,
            position_ids=_positions_from_mask(full_mask, 1),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        next_hidden = outputs.hidden_states[-1][:, -1, :]
        logits = outputs.logits[:, -1, :]
        top_tokens = _topk_tokens(model, logits, top_k)
        top1 = top_tokens[0]

        stats.append(
            StepStat(
                step=step,
                mode="latent_realign_on" if apply_realign else "latent_realign_off",
                feed_norm=float(latent_embed.norm(dim=-1).item()),
                hidden_norm=float(next_hidden.norm(dim=-1).item()),
                cos_to_prev_hidden=_cosine(latent_embed[0], prev_hidden[0]),
                cos_to_init_hidden=_cosine(latent_embed[0], init_hidden[0]),
                cos_feed_to_next_hidden=_cosine(latent_embed[0], next_hidden[0]),
                top1_token=top1["token"],
                top1_logit=top1["logit"],
                top_tokens=top_tokens,
            )
        )
        prev_hidden = next_hidden

    return stats


@torch.no_grad()
def _rollout_normal(
    model: ModelWrapper,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    steps: int,
    top_k: int,
) -> List[StepStat]:
    hf_model = model.model
    device = input_ids.device
    input_embed_layer = hf_model.get_input_embeddings()

    outputs = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=_positions_from_mask(attention_mask, input_ids.shape[1]),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    init_hidden = outputs.hidden_states[-1][:, -1, :]
    prev_hidden = init_hidden
    logits = outputs.logits[:, -1, :]
    full_mask = attention_mask
    stats: List[StepStat] = []

    for step in range(1, steps + 1):
        top_tokens = _topk_tokens(model, logits, top_k)
        next_token_id = torch.tensor([[top_tokens[0]["token_id"]]], dtype=torch.long, device=device)
        feed_embed = input_embed_layer(next_token_id).squeeze(1)

        full_mask = torch.cat(
            [
                full_mask,
                torch.ones((full_mask.shape[0], 1), dtype=full_mask.dtype, device=device),
            ],
            dim=-1,
        )

        outputs = hf_model(
            input_ids=next_token_id,
            attention_mask=full_mask,
            position_ids=_positions_from_mask(full_mask, 1),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        next_hidden = outputs.hidden_states[-1][:, -1, :]

        stats.append(
            StepStat(
                step=step,
                mode="normal_greedy",
                feed_norm=float(feed_embed.norm(dim=-1).item()),
                hidden_norm=float(next_hidden.norm(dim=-1).item()),
                cos_to_prev_hidden=_cosine(feed_embed[0], prev_hidden[0]),
                cos_to_init_hidden=_cosine(feed_embed[0], init_hidden[0]),
                cos_feed_to_next_hidden=_cosine(feed_embed[0], next_hidden[0]),
                top1_token=top_tokens[0]["token"],
                top1_logit=top_tokens[0]["logit"],
                top_tokens=top_tokens,
            )
        )
        prev_hidden = next_hidden
        logits = outputs.logits[:, -1, :]

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "aime2024", "aime2025", "arc_easy", "arc_challenge"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--role", type=str, default="planner", choices=["planner", "critic", "refiner", "judger"])
    parser.add_argument("--latent_steps", type=int, default=80)
    parser.add_argument("--mode", type=str, default="latent", choices=["latent", "normal", "both"], help="Which rollout to analyze.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of decoded token candidates to report per step.")
    parser.add_argument("--plot_steps", type=int, default=20, help="How many initial steps to include in the plot.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="latent_drift_report.json")
    parser.add_argument("--compare_realign", action="store_true", help="Run both realign ON and OFF in one pass.")
    parser.add_argument("--report_svd", action="store_true", help="Compute singular values for the raw least-squares matrix.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)

    wrapper_args = argparse.Namespace(
        latent_space_realign=False,
        model_name=args.model_name,
        task=args.task,
    )
    model = ModelWrapper(args.model_name, device, use_vllm=False, args=wrapper_args)
    item = _load_task_item(args.task, args.split, args.sample_index)

    prompt_messages = build_agent_message_sequential_latent_mas(
        role=args.role,
        question=item["question"],
        context="",
        method="latent_mas",
        args=wrapper_args,
    )
    prompt_text, input_ids, attention_mask, tokens = model.prepare_chat_input(
        prompt_messages,
        add_generation_prompt=True,
    )

    need_matrix = args.mode in {"latent", "both"} or args.report_svd
    raw_matrix: Optional[torch.Tensor] = None
    target_norm: Optional[torch.Tensor] = None
    if need_matrix:
        raw_matrix, target_norm = _build_raw_realign_matrix(model.model, device)

    report: Dict = {
        "model_name": args.model_name,
        "task": args.task,
        "split": args.split,
        "sample_index": args.sample_index,
        "role": args.role,
        "latent_steps": args.latent_steps,
        "mode": args.mode,
        "top_k": args.top_k,
        "plot_steps": args.plot_steps,
        "prompt_text": prompt_text,
        "prompt_tokens": tokens,
        "target_norm": float(target_norm.item()) if target_norm is not None else None,
        "matrix_stats": _matrix_stats(raw_matrix, include_spectrum=args.report_svd) if raw_matrix is not None and (args.report_svd or args.compare_realign) else None,
    }

    rollouts: Dict[str, List[Dict]] = {}
    if args.mode in {"latent", "both"}:
        assert raw_matrix is not None and target_norm is not None
        if args.compare_realign:
            rollouts["latent_realign_off"] = [asdict(s) for s in _rollout_latent(
                model,
                input_ids,
                attention_mask,
                latent_steps=args.latent_steps,
                apply_realign=False,
                matrix=raw_matrix,
                target_norm=target_norm,
                top_k=args.top_k,
            )]
            rollouts["latent_realign_on"] = [asdict(s) for s in _rollout_latent(
                model,
                input_ids,
                attention_mask,
                latent_steps=args.latent_steps,
                apply_realign=True,
                matrix=raw_matrix,
                target_norm=target_norm,
                top_k=args.top_k,
            )]
        else:
            rollouts["latent_realign_on"] = [asdict(s) for s in _rollout_latent(
                model,
                input_ids,
                attention_mask,
                latent_steps=args.latent_steps,
                apply_realign=True,
                matrix=raw_matrix,
                target_norm=target_norm,
                top_k=args.top_k,
            )]

    if args.mode in {"normal", "both"}:
        rollouts["normal_greedy"] = [asdict(s) for s in _rollout_normal(
            model,
            input_ids,
            attention_mask,
            steps=args.latent_steps,
            top_k=args.top_k,
        )]

    report["rollouts"] = rollouts

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Saved report to: {out_path.resolve()}")
    print(f"Target norm: {report['target_norm']:.4f}")
    if report["matrix_stats"] is not None:
        ms = report["matrix_stats"]
        print(
            "Matrix stats: "
            f"shape={ms['shape']}, "
            f"||R||_F={ms['fro_norm']:.4f}, "
            f"||R-I||_F={ms['fro_norm_to_identity']:.4f}, "
            f"sv_min={ms['min_singular']:.6f}, "
            f"sv_med={ms['median_singular']:.6f}, "
            f"sv_max={ms['max_singular']:.6f}, "
            f"cond={ms['condition_number']:.4f}"
        )

    for name, rows in report["rollouts"].items():
        print(f"\n== {name} ==")
        for row in rows[: min(10, len(rows))]:
            print(
                f"step {row['step']:>3}: "
                f"feed_norm={row['feed_norm']:.4f} "
                f"hidden_norm={row['hidden_norm']:.4f} "
                f"cos(prev)={row['cos_to_prev_hidden']:.4f} "
                f"cos(init)={row['cos_to_init_hidden']:.4f} "
                f"top1={row['top1_token']!r} "
                f"p={row['top_tokens'][0]['prob']:.4f}"
            )

    saved_plots = _plot_report(report, out_path)
    if saved_plots:
        print("\nSaved plots:")
        for path in saved_plots:
            print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()
