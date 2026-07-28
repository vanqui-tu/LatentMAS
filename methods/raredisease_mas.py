"""
RarediseaseMASMethod — LatentMAS baseline for cross-hospital rare-disease diagnosis.

Architecture (matches MedLatentDx "Raw KV / LatentMAS" baseline, Appendix C.5):

  1. N hospital agents run in ONE batched forward pass (same backbone).
     Each agent encodes a different prompt:
       system + hospital_i_retrieved_case + test_phenotype
     and produces `latent_steps` autoregressive latent tokens via the
     distiller ϕ (identical to LatentMAS's generate_latent_batch).

  2. The resulting KV blocks are **concatenated along the sequence dimension**
     after stripping per-sample padding.

  3. A single host agent runs generate_text_batch conditioned on the
     combined KV, then decodes the final diagnosis.

Key design decisions:
  - Batch all hospital agents together (one model.forward per latent step).
  - Strip left-padding from each sample's KV before concat so the host
    doesn't attend to pad tokens.
  - When a hospital has multiple retrieved cases, each case becomes a
    separate hospital "agent" (effectively num_hospitals * top_k agents),
    OR (default) we format them all into one prompt per hospital.
    Default: one prompt per hospital (all top_k cases inlined).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch

from models import ModelWrapper, _past_length
from prompts import (
    build_crossrare_hospital_prompt,
    build_crossrare_host_prompt,
)
from crossrare_data import _parse_disease_aliases

# ────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ────────────────────────────────────────────────────────────────────────────

def _extract_answer_tag(text: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags.

    Falls back to the last non-empty line if no tags found.
    Also strips common model preambles like 'The diagnosis is:'.
    """
    # Primary: <answer> tags (as instructed in template C)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Secondary: last non-empty line (model may not follow template)
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    # Strip common preamble patterns
    last = re.sub(r"^(the (most likely |single )?diagnosis is:?\s*)", "", last, flags=re.IGNORECASE)
    last = re.sub(r"^(diagnosis:?\s*)", "", last, flags=re.IGNORECASE)
    return last.strip() or None


def _normalize_disease(name: str) -> str:
    """Lowercase, strip punctuation artifacts, collapse whitespace."""
    name = name.lower().strip()
    # Remove trailing punctuation
    name = re.sub(r"[.,;:!?]+$", "", name)
    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name)
    return name


def _disease_match(pred: Optional[str], gold: str, gold_aliases: Optional[List[str]] = None) -> bool:
    """Multi-level match between predicted disease name and any valid gold alias.

    gold_aliases should be the full list from _parse_disease_aliases(raw_disease).
    If not provided, falls back to matching against gold only.

    Levels per alias (in order):
      1. Exact match after normalization.
      2. Substring — one is contained in the other (min 5 chars each side).
      3. Word overlap ≥70% of alias words found in pred.
    """
    if not pred:
        return False

    candidates = gold_aliases if gold_aliases else [gold]

    p = _normalize_disease(pred)
    for alias in candidates:
        if not alias:
            continue
        g = _normalize_disease(alias)

        # Level 1: exact
        if p == g:
            return True

        # Level 2: substring
        if len(p) >= 5 and len(g) >= 5:
            if g in p or p in g:
                return True

        # Level 3: word overlap ≥70%
        g_words = set(g.split())
        p_words = set(p.split())
        if len(g_words) >= 2:
            overlap = len(g_words & p_words) / len(g_words)
            if overlap >= 0.7:
                return True

    return False


# ────────────────────────────────────────────────────────────────────────────
# KV stripping helper
# ────────────────────────────────────────────────────────────────────────────

def _strip_kv_padding(past_kv: Tuple, attention_mask: torch.Tensor) -> List[Tuple]:
    """For each sample in the batch, extract only the real (non-padded) KV
    positions and return as a list of per-sample KV tuples.

    Args:
        past_kv:        Standard transformers past_key_values tuple.
                        Shape per layer per head: [batch, heads, seq, dim].
        attention_mask: [batch, seq] — 1 for real tokens, 0 for left padding.

    Returns:
        List[batch] of per-sample KV tuples with only real positions kept.
        Each element has shape [1, heads, real_len, dim] per layer per tensor.
    """
    batch_size = attention_mask.shape[0]
    per_sample: List[Tuple] = []

    for i in range(batch_size):
        real_len = int(attention_mask[i].sum().item())
        # KV is left-padded → real tokens are at the end
        sample_kv = tuple(
            tuple(t[i : i + 1, :, -real_len:, :].contiguous() for t in layer)
            for layer in past_kv
        )
        per_sample.append(sample_kv)

    return per_sample


def _concat_kv_list(kv_list: List[Tuple]) -> Tuple:
    """Concatenate a list of per-sample KV tuples along the sequence (dim=2).

    All tuples must have the same number of layers and the same
    heads / hidden dimension.
    """
    num_layers = len(kv_list[0])
    return tuple(
        tuple(
            torch.cat([kv[layer][j] for kv in kv_list], dim=2)
            for j in range(len(kv_list[0][layer]))
        )
        for layer in range(num_layers)
    )


def _make_ones_mask(batch_size: int, seq_len: int, device: torch.device, dtype=torch.long) -> torch.Tensor:
    return torch.ones(batch_size, seq_len, dtype=dtype, device=device)


# ────────────────────────────────────────────────────────────────────────────
# Method class
# ────────────────────────────────────────────────────────────────────────────

class RarediseaseMASMethod:
    """LatentMAS applied to cross-hospital rare-disease diagnosis.

    Each item must contain:
        item["test_phenotypes"]   — List[str]  human-readable HPO terms
        item["hospital_cases"]    — List[List[{case_disease, case_phenotype}]]
                                    Outer list: one entry per hospital.
                                    Inner list: retrieved cases (top_k).
        item["gold"]              — str  ground-truth disease name
        item["question"]          — str  display text
    """

    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 5,
        host_max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args=None,
    ) -> None:
        self.model = model
        self.latent_steps = latent_steps
        self.host_max_new_tokens = host_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.args = args

    # ------------------------------------------------------------------ #
    # Internal: encode one hospital batch → strip-padded KV list          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _encode_hospitals(
        self,
        hospital_messages: List[List[Dict]],
    ) -> List[Tuple]:
        """Run all hospital prompts in one batched forward + latent steps.

        This implements the LatentMAS baseline (Appendix C.5 of MedLatentDx paper):
          - Prefill: forward hospital prompt → KV of length T_i (prompt length)
          - Latent steps: m autoregressive forward passes → add m more KV positions
          - Output: full KV cache containing BOTH prompt (T_i) and latent (m) positions

        Returns a list (one per hospital) of KV tuples with per-sample padding stripped.
        Shape per layer per tensor: [1, heads, T_i + latent_steps, dim].
        """
        num_hospitals = len(hospital_messages)
        device = self.model.device

        # ── tokenise all hospital prompts together ─────────────────────
        prompts: List[str] = [
            self.model.render_chat(msgs, add_generation_prompt=True)
            for msgs in hospital_messages
        ]
        encoded = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # ── prefill: one forward pass for all hospitals ────────────────
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_kv = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [H, dim]
        full_mask = attention_mask  # [H, seq]

        # ── latent steps (autoregressive, batched) ─────────────────────
        for _ in range(self.latent_steps):
            latent_vec = self.model._apply_latent_realignment(last_hidden, self.model.model)
            latent_embed = latent_vec.unsqueeze(1)  # [H, 1, dim]

            full_mask = torch.cat(
                [full_mask, torch.ones(num_hospitals, 1, dtype=full_mask.dtype, device=device)],
                dim=1,
            )
            from models import _positions_from_mask
            outputs = self.model.model(
                inputs_embeds=latent_embed,
                attention_mask=full_mask,
                position_ids=_positions_from_mask(full_mask, 1),
                past_key_values=past_kv,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_kv = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        # ── strip padding only ────────────────────────────────────────
        # full_mask has shape [H, prompt_len + latent_steps]
        # Strip left-padding but KEEP all real tokens (prompt + latent).
        kv_stripped = _strip_kv_padding(past_kv, full_mask)
        kv_latent = kv_stripped  # Do NOT truncate to latent_steps only

    # ------------------------------------------------------------------ #
    # run_batch — process a batch of items                                 #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        results: List[Dict] = []

        # Process each item independently (hospital fan-in is within one item).
        # generate_bs here controls how many test cases run in parallel at the
        # outer loop; since the hospital batch already saturates GPU memory we
        # default generate_bs=1.
        for item in items:
            result = self._run_single(item)
            results.append(result)

        return results

    # ------------------------------------------------------------------ #
    # _run_single — full pipeline for one test case                        #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _run_single(self, item: Dict) -> Dict:
        test_phenotypes: List[str] = item["test_phenotypes"]
        hospital_cases: List[List[Dict]] = item["hospital_cases"]
        gold: str = item["gold"]
        gold_aliases: List[str] = item.get("gold_aliases", [gold])
        num_hospitals = len(hospital_cases)
        device = self.model.device

        test_phenotype_str = ", ".join(test_phenotypes)

        # ── build hospital prompts ─────────────────────────────────────
        # For each hospital we inline all its retrieved cases into ONE prompt.
        hospital_messages: List[List[Dict]] = []
        for h_idx, cases in enumerate(hospital_cases):
            if not cases:
                # Fallback: empty hospital — just the query
                msgs = build_crossrare_hospital_prompt(
                    hospital_id=h_idx + 1,
                    case_disease="(no local case available)",
                    case_phenotype="(none)",
                    test_phenotype=test_phenotype_str,
                )
                hospital_messages.append(msgs)
                continue

            if len(cases) == 1:
                msgs = build_crossrare_hospital_prompt(
                    hospital_id=h_idx + 1,
                    case_disease=cases[0]["case_disease"],
                    case_phenotype=cases[0]["case_phenotype"],
                    test_phenotype=test_phenotype_str,
                )
            else:
                # Multiple retrieved cases: inline all of them in the user turn
                system = "You are a specialist in the field of rare diseases. You will be provided with similar cases from multiple hospitals to help diagnose a patient."
                cases_text = "\n".join(
                    f"  Case {j+1}: disease=[{c['case_disease']}], phenotype=[{c['case_phenotype']}]"
                    for j, c in enumerate(cases)
                )
                user = (
                    f"Similar cases from Hospital {h_idx + 1}:\n"
                    f"{cases_text}\n\n"
                    f"Now consider the following patient case:\n"
                    f"Patient's phenotype: {test_phenotype_str}\n\n"
                    f"Think about what diagnoses are most likely for this patient."
                )
                msgs = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            hospital_messages.append(msgs)

        # ── encode all hospitals → latent KV blocks ────────────────────
        kv_list = self._encode_hospitals(hospital_messages)  # List[H] of KV tuples

        # ── concatenate KV blocks from all hospitals ───────────────────
        combined_kv = _concat_kv_list(kv_list)
        # Mask: all real (no padding in combined KV since we stripped it above)
        latent_positions = self.latent_steps if self.latent_steps > 0 else sum(
            _past_length(kv) for kv in kv_list
        )
        combined_seq_len = _past_length(combined_kv)
        combined_mask = _make_ones_mask(1, combined_seq_len, device=device)

        # ── host agent: decode final diagnosis ─────────────────────────
        host_messages = build_crossrare_host_prompt(test_phenotype_str)
        host_prompt = self.model.render_chat(host_messages, add_generation_prompt=True)

        # Mirror LatentMAS judger: optionally force thinking via <think> token
        if getattr(self.args, "think", False):
            host_prompt = f"{host_prompt}<think>"

        host_encoded = self.model.tokenizer(
            host_prompt,
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        )
        host_ids = host_encoded["input_ids"].to(device)
        host_mask = host_encoded["attention_mask"].to(device)

        generated_texts, _ = self.model.generate_text_batch(
            host_ids,
            host_mask,
            max_new_tokens=self.host_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            past_key_values=combined_kv,
            past_attention_mask=combined_mask,
        )

        raw_output = generated_texts[0].strip()
        pred = _extract_answer_tag(raw_output)
        tag_found = bool(re.search(r"<answer>", raw_output, re.IGNORECASE))
        ok = _disease_match(pred, gold, gold_aliases)

        # ── clean up GPU memory ────────────────────────────────────────
        del combined_kv, kv_list
        torch.cuda.empty_cache()

        # ── build per-hospital trace for logging ───────────────────────
        hospital_trace = []
        for h_idx, msgs in enumerate(hospital_messages):
            hospital_trace.append({
                "name": f"Hospital_{h_idx + 1}",
                "role": "hospital",
                "input": self.model.render_chat(msgs, add_generation_prompt=True),
                "latent_steps": self.latent_steps,
                "output": "",  # latent — no text output
            })
        hospital_trace.append({
            "name": "Host",
            "role": "host",
            "input": host_prompt,
            "output": raw_output,
        })

        return {
            "id": item.get("id", ""),
            "question": item["question"],
            "gold": gold,
            "gold_aliases": gold_aliases,
            "solution": item.get("solution", gold),
            "prediction": pred,
            "raw_prediction": raw_output,
            "tag_found": tag_found,
            "agents": hospital_trace,
            "correct": ok,
        }

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
