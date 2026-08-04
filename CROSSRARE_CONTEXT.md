# CrossRare / MedLatentDx Context

Use this note as the working context for CrossRare experiments in this repo.

## Data And Episodes

- Source: `data_CROSSRARE/data_crossrare_v2.json` (case ID, HPO IDs/names,
  disease, and `source`). Sources are `rarebench`, `zenodo`, and
  `phenopackets26`.
- `CrossRareDataset` in `crossrare_data.py` makes a seeded `90/5/5`
  train/validation/test split. Only the train split is partitioned into five
  private hospital databases.
- Retrieval is HPO embedding nearest-neighbour search: IC-weighted mean HPO
  vectors followed by dot-product ranking. Retrieved records are formatted as
  disease and phenotype text; original case text is not sent to the host.
- Default reproduction topology is five partitions, three fixed retrieval
  hospitals (`--agent_hospital_ids 1 2 3`), and top-1 retrieval. Thus every
  diagnosis receives three retrieved cases.
- For MedLatentDx-H training, query/label cases come only from the two
  non-retrieval hospitals (4 and 5 by default). They are absent from all three
  local retrieval databases. Validation and test cases are held out before
  hospital partitioning.

## Item Contract

Each episode contains `test_phenotypes`, `hospital_cases`, `hospital_ids`,
`gold`, `gold_aliases`, `question`, and `solution`. `hospital_cases` has one
entry per active retrieval hospital, not one entry per all five partitions.

## Methods

| Method | File | Communication |
| --- | --- | --- |
| Raw-KV baseline | `methods/raredisease_mas.py` | Full local prompt KV cache, plus optional LatentMAS steps; `--latent_steps 0` is raw prompt KV only. |
| MedLatentDx-H | `methods/medlatentdx_h.py` | A shared minimal distiller generates `m` latent positions per hospital; only their KV suffix is sent, wrapped by learned BOP/EOP blocks. |

Both methods reuse the CrossRare hospital/host templates from `prompts.py`.
The host emits one disease inside `<answer>...</answer>`.

## MedLatentDx-H Training

- Entry point: `train_medlatentdx_h.py`.
- Frozen: all local/host LLM backbone parameters. Trainable: one shared
  `LayerNorm + bias-free Linear` distiller and shared BOP/EOP embeddings.
- Default paper-aligned settings: `m=32`, AdamW (`lr=1e-4`, `wd=0.01`),
  LambdaLR warm-up 100 optimizer updates, 5 epochs, physical batch size 8,
  prompt limit 320, target limit 64, seed 42.
- The trainer batches episodes physically: it encodes all three hospital prompts
  per episode together, stitches fixed-length KV suffixes per episode, and
  computes host CE in one padded batch. `--grad_accumulation` optionally grows
  the effective batch beyond the physical GPU batch. Progress prints after every
  optimizer update by default.
- Validation CE selects the checkpoint. Checkpoints contain only distiller
  weights (`norm`, `projection`, `bop`, `eop`) and small shape/config metadata,
  never LLM weights.
- Recent Qwen Transformers models require `DynamicCache`; `models.py`
  converts concatenated legacy KV tuples before host forward/generation.

## Evaluation

`run.py` reports alias-aware accuracy: `correct / total` and macro-F1. Prediction extraction
first uses `<answer>` tags, then falls back to the final non-empty output line.
Disease matching in `raredisease_mas.py` accepts, in order:

1. Normalized exact match against any gold alias.
2. Substring match when both strings have at least five characters.
3. At least 70% coverage of a multi-word gold alias by predicted words.

Results are saved as a per-case JSON and summary JSON under `--output_dir`.
For CrossRare, the summary reports accuracy and macro-F1 for `overall` and
each source cohort (`rarebench`, `zenodo`, `phenopackets26`).

## Reproduction Commands

Raw-KV test baseline:

```bash
python run.py --method raredisease_mas --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --task crossrare --num_hospitals 5 --hospital_agents 3 \
  --agent_hospital_ids 1 2 3 --retrieval_top_k 1 \
  --test_ratio 0.05 --val_ratio 0.05 --partition_strategy round_robin \
  --latent_steps 0 --temperature 0 --max_new_tokens 64 --generate_bs 1 --seed 42
```

Train and evaluate MedLatentDx-H using the matching commands in `README.md`.
Training and inference must use the same model, split seed, partition strategy,
retrieval hospital IDs, and latent-step count.
