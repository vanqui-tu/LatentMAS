"""MedLatentDx-H: same-backbone compact latent communication.

This implementation follows Section 3.2.  The local prompt cache is never
passed to the host: only the ``m`` KV positions created by the trainable
distiller, bracketed by learned BOP/EOP boundary embeddings, are shared.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models import ModelWrapper, _as_transformers_cache, _past_length, _positions_from_mask
from methods.raredisease_mas import (
    _concat_kv_list,
    _disease_match,
    _extract_answer_tag,
)
from prompts import build_crossrare_hospital_prompt, build_crossrare_host_prompt, build_crossrare_system_prompt


class SameBackboneDistiller(nn.Module):
    """The minimal trainable interface in Eq. 4 and Eq. 5 of the paper."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bop = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.eop = nn.Parameter(torch.empty(1, 1, hidden_size))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.normal_(self.bop, std=0.02)
        nn.init.normal_(self.eop, std=0.02)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(hidden))


def _last_kv_positions(past_kv: Tuple, positions: int) -> Tuple:
    """Return the compact suffix of a cache, preserving autograd links."""
    if positions <= 0:
        raise ValueError("latent_steps must be positive for MedLatentDx-H")
    return tuple(
        tuple(value[:, :, -positions:, :].contiguous() for value in layer)
        for layer in past_kv
    )


def _hidden_size(model: nn.Module) -> int:
    size = getattr(model.config, "hidden_size", None)
    if size is None:
        size = model.get_input_embeddings().weight.shape[1]
    return int(size)


def _model_dtype(model: nn.Module) -> torch.dtype:
    return model.get_input_embeddings().weight.dtype


class MedLatentDxHMethod:
    """Inference and teacher-forced diagnosis loss for MedLatentDx-H."""

    def __init__(
        self,
        model: ModelWrapper,
        *,
        distiller: SameBackboneDistiller,
        latent_steps: int,
        max_prompt_length: int = 320,
        max_target_length: int = 64,
        host_max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args=None,
    ) -> None:
        if model.use_vllm:
            raise ValueError("MedLatentDx-H requires the Transformers backend; vLLM cannot backpropagate through the latent interface.")
        if latent_steps <= 0:
            raise ValueError("MedLatentDx-H requires --latent_steps > 0")
        self.model = model
        self.distiller = distiller.to(device=model.device, dtype=_model_dtype(model.model))
        self.latent_steps = latent_steps
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        self.host_max_new_tokens = host_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.args = args

    @classmethod
    def from_checkpoint(cls, model: ModelWrapper, checkpoint: str, **kwargs):
        payload = torch.load(checkpoint, map_location="cpu")
        config = payload["config"]
        distiller = SameBackboneDistiller(config["hidden_size"])
        distiller.load_state_dict(payload["state_dict"])
        requested_steps = kwargs.get("latent_steps")
        if requested_steps not in (None, 0, config["latent_steps"]):
            raise ValueError("--latent_steps must match the distiller checkpoint")
        kwargs["latent_steps"] = config["latent_steps"]
        requested_prompt_length = kwargs.get("max_prompt_length", 320)
        checkpoint_prompt_length = config.get("max_prompt_length", 320)
        if requested_prompt_length not in (320, checkpoint_prompt_length):
            raise ValueError("--max_prompt_length must match the distiller checkpoint")
        kwargs["max_prompt_length"] = checkpoint_prompt_length
        return cls(model, distiller=distiller, **kwargs)

    def checkpoint_payload(self) -> Dict:
        return {
            "config": {
                "hidden_size": _hidden_size(self.model.model),
                "latent_steps": self.latent_steps,
                "max_prompt_length": self.max_prompt_length,
            },
            "state_dict": self.distiller.state_dict(),
        }

    def save_checkpoint(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)

    def _hospital_messages(self, item: Dict) -> List[List[Dict]]:
        query = ", ".join(item["test_phenotypes"])
        messages: List[List[Dict]] = []
        hospital_ids = item.get("hospital_ids", list(range(1, len(item["hospital_cases"]) + 1)))
        for hospital_id, cases in zip(hospital_ids, item["hospital_cases"]):
            if len(cases) <= 1:
                case = cases[0] if cases else {"case_disease": "(no local case available)", "case_phenotype": "(none)"}
                messages.append(build_crossrare_hospital_prompt(hospital_id, case["case_disease"], case["case_phenotype"], query))
                continue
            cases_text = "\n".join(
                f"  Case {j}: disease=[{case['case_disease']}], phenotype=[{case['case_phenotype']}]"
                for j, case in enumerate(cases, start=1)
            )
            messages.append([
                {"role": "system", "content": build_crossrare_system_prompt()},
                {"role": "user", "content": (
                    f"Similar cases from Hospital {hospital_id}:\n{cases_text}\n\n"
                    f"Now consider the following patient case:\nPatient's phenotype: {query}\n\n"
                    "Think about what diagnoses are most likely for this patient."
                )},
            ])
        return messages

    def _encode_hospital_messages(self, messages: List[List[Dict]]) -> Tuple:
        """Encode a batch of hospital prompts into compact latent KV suffixes."""
        prompts = [self.model.render_chat(message, add_generation_prompt=True) for message in messages]
        encoded = self.model.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.max_prompt_length, add_special_tokens=False,
        )
        ids = encoded["input_ids"].to(self.model.device)
        mask = encoded["attention_mask"].to(self.model.device)

        # The frozen local backbone provides the starting state only.
        with torch.no_grad():
            out = self.model.model(input_ids=ids, attention_mask=mask, use_cache=True, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
        hidden = out.hidden_states[-1][:, -1, :]
        full_mask = mask
        batch = ids.shape[0]

        # Gradients flow through phi and the frozen backbone's latent forwards.
        for _ in range(self.latent_steps):
            latent = self.distiller(hidden).to(dtype=_model_dtype(self.model.model)).unsqueeze(1)
            full_mask = torch.cat([full_mask, torch.ones(batch, 1, dtype=full_mask.dtype, device=full_mask.device)], dim=1)
            out = self.model.model(
                inputs_embeds=latent,
                attention_mask=full_mask,
                position_ids=_positions_from_mask(full_mask, 1),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = out.past_key_values
            hidden = out.hidden_states[-1][:, -1, :]

        # Every batch row's compact suffix has fixed m positions and no prompt padding.
        return _last_kv_positions(past, self.latent_steps)

    def _encode_hospitals(self, messages: List[List[Dict]]) -> List[Tuple]:
        """Create each hospital's compact suffix M-tilde_Hi, never its prompt KV."""
        compact_batch = self._encode_hospital_messages(messages)
        batch = compact_batch[0][0].shape[0]
        return [
            tuple(tuple(value[i : i + 1].contiguous() for value in layer) for layer in compact_batch)
            for i in range(batch)
        ]

    def _boundary_cache(self, embedding: torch.Tensor) -> Tuple:
        """Materialise a learned boundary embedding into an aligned one-position KV block."""
        out = self.model.model(
            inputs_embeds=embedding.to(device=self.model.device, dtype=_model_dtype(self.model.model)),
            attention_mask=torch.ones(1, 1, dtype=torch.long, device=self.model.device),
            use_cache=True,
            return_dict=True,
        )
        return out.past_key_values

    def _boundary_cache_batch(self, embedding: torch.Tensor, batch_size: int) -> Tuple:
        """Materialise one learned boundary KV block per episode in a batch."""
        embeddings = embedding.expand(batch_size, -1, -1).to(
            device=self.model.device, dtype=_model_dtype(self.model.model)
        )
        out = self.model.model(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(batch_size, 1, dtype=torch.long, device=self.model.device),
            use_cache=True,
            return_dict=True,
        )
        return out.past_key_values

    def _combined_compact_cache(self, item: Dict) -> Tuple:
        compact = self._encode_hospitals(self._hospital_messages(item))
        bop = self._boundary_cache(self.distiller.bop)
        eop = self._boundary_cache(self.distiller.eop)
        blocks: List[Tuple] = []
        for local_cache in compact:
            blocks.extend([bop, local_cache, eop])
        return _concat_kv_list(blocks)

    def _combined_compact_cache_batch(self, items: List[Dict]) -> Tuple:
        """Build one stitched [batch, KV] cache per episode without per-row slicing."""
        if not items:
            raise ValueError("items must not be empty")
        hospital_count = len(items[0]["hospital_cases"])
        if hospital_count == 0 or any(len(item["hospital_cases"]) != hospital_count for item in items):
            raise ValueError("all items must contain the same non-zero number of hospitals")

        messages = [
            message
            for item in items
            for message in self._hospital_messages(item)
        ]
        batch_size = len(items)
        compact = self._encode_hospital_messages(messages)
        bop = self._boundary_cache_batch(self.distiller.bop, batch_size)
        eop = self._boundary_cache_batch(self.distiller.eop, batch_size)

        stitched_layers = []
        for compact_layer, bop_layer, eop_layer in zip(compact, bop, eop):
            stitched_values = []
            for compact_value, bop_value, eop_value in zip(compact_layer, bop_layer, eop_layer):
                # [B * H, heads, m, dim] -> [B, heads, H * (m + 2), dim]
                grouped = compact_value.reshape(batch_size, hospital_count, *compact_value.shape[1:])
                bop_blocks = bop_value.unsqueeze(1).expand(-1, hospital_count, -1, -1, -1)
                eop_blocks = eop_value.unsqueeze(1).expand(-1, hospital_count, -1, -1, -1)
                blocks = torch.cat([bop_blocks, grouped, eop_blocks], dim=3)
                stitched_values.append(
                    blocks.permute(0, 2, 1, 3, 4).reshape(
                        batch_size, blocks.shape[2], -1, blocks.shape[-1]
                    )
                )
            stitched_layers.append(tuple(stitched_values))
        return tuple(stitched_layers)

    def diagnosis_loss(self, item: Dict) -> torch.Tensor:
        """Equation 6: CE only on the host answer, never on latent positions."""
        return self.diagnosis_loss_batch([item])

    def diagnosis_loss_batch(self, items: List[Dict]) -> torch.Tensor:
        """Mean teacher-forced CE for a physical batch of CrossRare episodes."""
        if not items:
            raise ValueError("items must not be empty")
        compact_cache = self._combined_compact_cache_batch(items)
        hosts = [
            self.model.render_chat(
                build_crossrare_host_prompt(", ".join(item["test_phenotypes"])),
                add_generation_prompt=True,
            )
            for item in items
        ]
        targets = [
            f"<answer>{item['gold']}</answer>{self.model.tokenizer.eos_token or ''}"
            for item in items
        ]
        host_ids = self.model.tokenizer(
            hosts, truncation=True, max_length=self.max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        target_ids = self.model.tokenizer(
            targets, truncation=True, max_length=self.max_target_length,
            add_special_tokens=False,
        )["input_ids"]
        sequence_lengths = [len(host) + len(target) for host, target in zip(host_ids, target_ids)]
        current_length = max(sequence_lengths)
        batch_size = len(items)
        current_ids = torch.full(
            (batch_size, current_length), self.model.tokenizer.pad_token_id,
            dtype=torch.long, device=self.model.device,
        )
        current_mask = torch.zeros_like(current_ids)
        labels = torch.full_like(current_ids, -100)
        for index, (host, target) in enumerate(zip(host_ids, target_ids)):
            sequence = host + target
            start = current_length - len(sequence)
            target_start = current_length - len(target)
            current_ids[index, start:] = torch.tensor(sequence, dtype=torch.long, device=self.model.device)
            current_mask[index, start:] = 1
            labels[index, target_start:] = torch.tensor(target, dtype=torch.long, device=self.model.device)

        inputs = self.model.model.get_input_embeddings()(current_ids)
        past_len = _past_length(compact_cache)
        past_mask = torch.ones(batch_size, past_len, dtype=current_mask.dtype, device=self.model.device)
        mask = torch.cat([past_mask, current_mask], dim=1)
        output = self.model.model(
            inputs_embeds=inputs,
            attention_mask=mask,
            position_ids=_positions_from_mask(mask, current_ids.shape[1]),
            past_key_values=_as_transformers_cache(compact_cache),
            use_cache=False,
            return_dict=True,
        )
        token_losses = F.cross_entropy(
            output.logits[:, :-1, :].float().transpose(1, 2), labels[:, 1:],
            reduction="none", ignore_index=-100,
        )
        target_counts = (labels[:, 1:] != -100).sum(dim=1)
        return (token_losses.sum(dim=1) / target_counts).mean()

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        results = []
        for item in items:
            cache = self._combined_compact_cache(item)
            query = ", ".join(item["test_phenotypes"])
            host_messages = build_crossrare_host_prompt(query)
            host_prompt = self.model.render_chat(host_messages, add_generation_prompt=True)
            encoded = self.model.tokenizer(
                host_prompt, return_tensors="pt", truncation=True,
                max_length=self.max_prompt_length, add_special_tokens=False,
            )
            generated, _ = self.model.generate_text_batch(
                encoded["input_ids"].to(self.model.device),
                encoded["attention_mask"].to(self.model.device),
                max_new_tokens=self.host_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                past_key_values=_as_transformers_cache(cache),
                past_attention_mask=torch.ones(1, _past_length(cache), dtype=torch.long, device=self.model.device),
            )
            raw = generated[0].strip()
            pred = _extract_answer_tag(raw)
            gold = item["gold"]
            results.append({
                "id": item.get("id", ""), "source": item.get("source", ""),
                "question": item["question"], "gold": gold,
                "gold_aliases": item.get("gold_aliases", [gold]), "solution": item.get("solution", gold),
                "prediction": pred, "raw_prediction": raw, "tag_found": "<answer>" in raw.lower(),
                "correct": _disease_match(pred, gold, item.get("gold_aliases", [gold])),
                "agents": [{"name": "Host", "role": "host", "input": host_prompt, "output": raw}],
            })
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
