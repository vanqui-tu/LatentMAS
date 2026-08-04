"""Train the Section 3.2 MedLatentDx-H latent interface on CrossRare."""
import argparse
import random

import torch

from crossrare_data import CrossRareDataset
from methods.medlatentdx_h import MedLatentDxHMethod, SameBackboneDistiller, _hidden_size
from models import ModelWrapper
from utils import auto_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latent_steps", type=int, default=32)
    parser.add_argument("--num_hospitals", type=int, default=5)
    parser.add_argument("--hospital_agents", type=int, default=3)
    parser.add_argument("--agent_hospital_ids", type=int, nargs=3, default=[1, 2, 3],
                        help="One-based IDs of the three hospitals providing local retrieval.")
    parser.add_argument("--retrieval_top_k", type=int, default=1)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--partition_strategy", choices=["random", "round_robin", "skewed"], default="random")
    parser.add_argument("--skewed_dirichlet_alpha", type=float, default=0.3,
                        help="Per-disease Dirichlet alpha for the skewed partition (paper: 0.3).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr_schedule", choices=["constant", "linear"], default="constant",
                        help="Learning-rate schedule after warm-up; linear decays to zero by the final update.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Physical GPU batch size in episodes.")
    parser.add_argument("--grad_accumulation", type=int, default=1,
                        help="Number of physical batches accumulated per AdamW update.")
    parser.add_argument("--log_every", type=int, default=1,
                        help="Print training metrics every N optimizer updates.")
    parser.add_argument("--max_prompt_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.latent_steps, args.epochs, args.batch_size, args.grad_accumulation, args.log_every,
           args.max_prompt_length, args.max_target_length) <= 0:
        parser.error("all length, batch, epoch, and accumulation settings must be positive")
    if (args.num_hospitals, args.hospital_agents, args.retrieval_top_k) != (5, 3, 1):
        parser.error("MedLatentDx-H reproduction fixes five hospital partitions, three agents/query, and top-1 retrieval")

    set_seed(args.seed)
    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, args=args)
    # Frozen LLMs are part of the objective but never optimizer parameters.
    model.model.requires_grad_(False)
    model.model.eval()
    distiller = SameBackboneDistiller(_hidden_size(model.model))
    method = MedLatentDxHMethod(
        model, distiller=distiller, latent_steps=args.latent_steps,
        max_prompt_length=args.max_prompt_length, max_target_length=args.max_target_length, args=args,
    )
    optimizer = torch.optim.AdamW(method.distiller.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    dataset = CrossRareDataset(num_hospitals=args.num_hospitals, num_active_hospitals=args.hospital_agents,
                               agent_hospital_ids=args.agent_hospital_ids,
                               test_ratio=args.test_ratio, val_ratio=args.val_ratio,
                               top_k=args.retrieval_top_k, seed=args.seed,
                               partition_strategy=args.partition_strategy,
                               skewed_dirichlet_alpha=args.skewed_dirichlet_alpha)
    examples = dataset.train_items()
    total_query_examples = len(examples)
    if args.max_train_samples > 0:
        examples = examples[:args.max_train_samples]
    if not examples:
        raise RuntimeError("No CrossRare training examples available")
    validation_examples = dataset.val_items()
    if not validation_examples:
        raise RuntimeError("No CrossRare validation examples available")
    query_hospitals = [hospital_id for hospital_id in range(1, args.num_hospitals + 1)
                       if hospital_id not in args.agent_hospital_ids]
    print(
        f"[Train] Query/label episodes: {len(examples)}/{total_query_examples} from hospitals {query_hospitals}; "
        f"retrieval hospitals: {args.agent_hospital_ids}",
        flush=True,
    )

    # Each physical batch uses padded prompts but fixed-length compact KV blocks.
    # Accumulation preserves the effective batch size when a larger GPU batch does
    # not fit in memory.
    update_size = args.batch_size * args.grad_accumulation
    updates_per_epoch = (len(examples) + update_size - 1) // update_size
    total_updates = updates_per_epoch * args.epochs

    def lr_multiplier(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / max(1, args.warmup_steps)
        if args.lr_schedule == "constant":
            return 1.0
        decay_steps = max(1, total_updates - args.warmup_steps)
        progress = min(1.0, float(step - args.warmup_steps + 1) / decay_steps)
        return 1.0 - progress

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_multiplier,
    )

    method.distiller.train()
    optimizer.zero_grad(set_to_none=True)
    best_validation_loss = float("inf")
    global_update = 0
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(examples)
        mean_loss = 0.0
        update_loss = 0.0
        update_examples = 0
        for update_start in range(0, len(examples), update_size):
            update_items = examples[update_start:update_start + update_size]
            chunk_size = len(update_items)
            for batch_start in range(0, chunk_size, args.batch_size):
                batch = update_items[batch_start:batch_start + args.batch_size]
                loss = method.diagnosis_loss_batch(batch)
                # ``loss`` is a mean over this physical batch; weight it by the
                # number of episodes so the update remains a mean over its chunk.
                (loss * (len(batch) / chunk_size)).backward()
                loss_value = float(loss.detach())
                mean_loss += loss_value * len(batch)
                update_loss += loss_value * len(batch)
                update_examples += len(batch)

            torch.nn.utils.clip_grad_norm_(method.distiller.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            epoch_update = update_start // update_size + 1
            if global_update % args.log_every == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"update={epoch_update}/{updates_per_epoch} global_update={global_update} "
                    f"train_ce={update_loss / update_examples:.6f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
            update_loss = 0.0
            update_examples = 0
        method.distiller.eval()
        with torch.no_grad():
            validation_total = 0.0
            for batch_start in range(0, len(validation_examples), args.batch_size):
                batch = validation_examples[batch_start:batch_start + args.batch_size]
                validation_total += float(method.diagnosis_loss_batch(batch)) * len(batch)
            validation_loss = validation_total / len(validation_examples)
        print(f"epoch={epoch + 1} train_ce={mean_loss / len(examples):.6f} val_ce={validation_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            method.save_checkpoint(args.output_checkpoint)
            print(f"Saved best checkpoint (val_ce={validation_loss:.6f}): {args.output_checkpoint}")
        method.distiller.train()
    print(f"Best validation CE: {best_validation_loss:.6f}")


if __name__ == "__main__":
    main()
