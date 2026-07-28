"""
CrossRare data loading, hospital partitioning, and HPO-based retrieval.

Data format (data_crossrare.json):
  List of dicts with keys:
    - id: str
    - phenotype_ids: List[str]   (HP:XXXXXXX)
    - phenotypes: List[str]      (human-readable names)
    - disease_ids: List[str]
    - disease: str               (primary disease name — used as gold label)

Split: 9/10 train (distributed across hospitals), 1/10 test.
Hospitals: 5 hospitals, each receives an equal partition of the train split.

Retrieval: For a query case, compute a weighted dot-product similarity between
the query's HPO embedding and each database case's HPO embedding.
  - Each case is represented as the IC-weighted mean of its HPO term embeddings.
  - Similarity = dot(query_vec, case_vec)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Paths (relative to project root; override via env or pass explicitly)
# ────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data_CROSSRARE"
_DATA_FILE = _DATA_DIR / "data_crossrare.json"
_PHE2EMB_FILE = _DATA_DIR / "phe2embedding.json"
_IC_FILE = _DATA_DIR / "ic_dict.json"
_DISEASE_MAP_FILE = _DATA_DIR / "disease_mapping.json"
_PHENOTYPE_MAP_FILE = _DATA_DIR / "phenotype_mapping.json"


# ────────────────────────────────────────────────────────────────────────────
# Loading helpers
# ────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _load_crossrare_raw() -> List[Dict]:
    return _load_json(_DATA_FILE)


def _load_phe2embedding() -> Dict[str, List[float]]:
    return _load_json(_PHE2EMB_FILE)


def _load_ic_dict() -> Dict[str, float]:
    return _load_json(_IC_FILE)


# ────────────────────────────────────────────────────────────────────────────
# Case embedding
# ────────────────────────────────────────────────────────────────────────────

def case_embedding(
    phenotype_ids: List[str],
    phe2emb: Dict[str, List[float]],
    ic_dict: Dict[str, float],
) -> np.ndarray:
    """IC-weighted mean of HPO term embeddings for a single case.

    Terms missing from either lookup are silently skipped.
    Returns a zero vector if no terms match (edge case).
    """
    vecs: List[np.ndarray] = []
    weights: List[float] = []

    for hpo_id in phenotype_ids:
        emb = phe2emb.get(hpo_id)
        ic = ic_dict.get(hpo_id)
        if emb is not None and ic is not None:
            vecs.append(np.array(emb, dtype=np.float32))
            weights.append(float(ic))

    if not vecs:
        dim = len(next(iter(phe2emb.values())))
        return np.zeros(dim, dtype=np.float32)

    weight_arr = np.array(weights, dtype=np.float32)
    weight_arr = weight_arr / (weight_arr.sum() + 1e-9)
    return sum(w * v for w, v in zip(weight_arr, vecs))


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / (norm + 1e-9)


# ────────────────────────────────────────────────────────────────────────────
# Split and partition
# ────────────────────────────────────────────────────────────────────────────

def _parse_disease_aliases(raw: str) -> List[str]:
    """Parse all valid alias names from a raw disease string.

    The field can look like:
      "Disease A"
      "Disease A/Disease A"                    (exact dup)
      "Disease A/Disease A'"                   (two different aliases)
      "Chinese/English name; abbrev./Alt name/Alt name 2"
      "Name; abbrev;/Alt name/Alt name 2"

    Strategy:
      1. Split on '/' to get segments.
      2. Each segment may contain ';'-separated sub-names.
      3. Clean each sub-name (strip whitespace, trailing semicolons/dots).
      4. Deduplicate (case-insensitive) while preserving order.
    """
    segments = raw.split("/")
    aliases: List[str] = []
    seen: set = set()
    for seg in segments:
        parts = seg.split(";")
        for part in parts:
            name = part.strip().rstrip(";.,: ")
            if not name:
                continue
            key = name.lower()
            if key not in seen:
                seen.add(key)
                aliases.append(name)
    return aliases


def _clean_disease_name(raw: str) -> str:
    """Return the primary (first) name from the raw disease field."""
    aliases = _parse_disease_aliases(raw)
    return aliases[0] if aliases else raw.strip()


def split_and_partition(
    data: List[Dict],
    num_hospitals: int = 5,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[List[Dict]], List[Dict]]:
    """Shuffle, split 9:1, distribute train across hospitals.

    Returns:
        hospital_dbs: List[num_hospitals] of case lists (train partition).
        test_cases:   List of test cases.
    """
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_cases = shuffled[:n_test]
    train_cases = shuffled[n_test:]

    # Round-robin distribute train cases to hospitals
    hospital_dbs: List[List[Dict]] = [[] for _ in range(num_hospitals)]
    for i, case in enumerate(train_cases):
        hospital_dbs[i % num_hospitals].append(case)

    return hospital_dbs, test_cases


# ────────────────────────────────────────────────────────────────────────────
# Retrieval
# ────────────────────────────────────────────────────────────────────────────

class HospitalRetriever:
    """Pre-computes embeddings for a hospital's local database and supports
    fast dot-product nearest-neighbour retrieval."""

    def __init__(
        self,
        cases: List[Dict],
        phe2emb: Dict[str, List[float]],
        ic_dict: Dict[str, float],
        hospital_id: int,
    ) -> None:
        self.cases = cases
        self.hospital_id = hospital_id

        # Pre-compute and normalise embeddings: shape [N, D]
        embs = [
            _normalize(case_embedding(c["phenotype_ids"], phe2emb, ic_dict))
            for c in cases
        ]
        self.embeddings = np.stack(embs, axis=0)  # [N, D]

    def retrieve(self, query_ids: List[str], phe2emb: Dict, ic_dict: Dict, top_k: int = 3) -> List[Dict]:
        """Return top_k most similar cases from local database."""
        if not self.cases:
            return []

        q_emb = _normalize(case_embedding(query_ids, phe2emb, ic_dict))
        scores = self.embeddings @ q_emb  # [N]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.cases[i] for i in top_indices]


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

class CrossRareDataset:
    """One-stop object for loading, splitting, and retrieving CrossRare data.

    Usage:
        ds = CrossRareDataset(num_hospitals=5, seed=42)
        for item in ds.test_items(top_k=3):
            # item["test_phenotypes"]         — List[str] human-readable
            # item["test_phenotype_ids"]      — List[str] HP codes
            # item["gold"]                    — str disease name
            # item["hospital_cases"]          — List[List[Dict]]  (one list per hospital)
            #   each inner dict: {case_disease, case_phenotype}
            # item["question"]                — formatted text for the model
    """

    def __init__(
        self,
        num_hospitals: int = 5,
        test_ratio: float = 0.1,
        top_k: int = 3,
        seed: int = 42,
    ) -> None:
        self.num_hospitals = num_hospitals
        self.top_k = top_k

        print("[CrossRare] Loading data...")
        raw = _load_crossrare_raw()
        self.phe2emb = _load_phe2embedding()
        self.ic_dict = _load_ic_dict()

        hospital_dbs, test_cases = split_and_partition(
            raw, num_hospitals=num_hospitals, test_ratio=test_ratio, seed=seed
        )

        print(f"[CrossRare] Train: {sum(len(h) for h in hospital_dbs)} cases across {num_hospitals} hospitals | Test: {len(test_cases)}")

        self.retrievers = [
            HospitalRetriever(db, self.phe2emb, self.ic_dict, hospital_id=i)
            for i, db in enumerate(hospital_dbs)
        ]
        self.test_cases = test_cases

    def test_items(self) -> Iterable[Dict]:
        """Yield one dict per test case, with retrieval results pre-filled."""
        for case in self.test_cases:
            query_ids = case["phenotype_ids"]
            query_phenotypes = case["phenotypes"]
            gold = _clean_disease_name(case["disease"])
            gold_aliases = _parse_disease_aliases(case["disease"])

            hospital_cases: List[List[Dict]] = []
            for retriever in self.retrievers:
                retrieved = retriever.retrieve(query_ids, self.phe2emb, self.ic_dict, top_k=self.top_k)
                # Normalise to {case_disease, case_phenotype} format expected by prompts
                formatted = [
                    {
                        "case_disease": _clean_disease_name(r["disease"]),
                        "case_phenotype": ", ".join(r["phenotypes"]),
                    }
                    for r in retrieved
                ]
                hospital_cases.append(formatted)

            # Build a simple text question for display / logging
            question = "Patient's phenotype: " + ", ".join(query_phenotypes)

            yield {
                "id": case.get("id", ""),
                "test_phenotypes": query_phenotypes,
                "test_phenotype_ids": query_ids,
                "gold": gold,
                "gold_aliases": gold_aliases,   # all valid names for this disease
                "hospital_cases": hospital_cases,
                "question": question,
                "solution": gold,
            }


def load_crossrare(num_hospitals: int = 5, test_ratio: float = 0.1, top_k: int = 3, seed: int = 42) -> Iterable[Dict]:
    """Thin wrapper for use in run.py — returns an iterable of test items."""
    ds = CrossRareDataset(num_hospitals=num_hospitals, test_ratio=test_ratio, top_k=top_k, seed=seed)
    return list(ds.test_items())
