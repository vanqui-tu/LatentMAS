"""
CrossRare data loading, hospital partitioning, and HPO-based retrieval.

Data format (data_crossrare_v2.json):
  List of dicts with keys:
    - id: str
    - phenotype_ids: List[str]   (HP:XXXXXXX)
    - phenotypes: List[str]      (human-readable names)
    - disease_ids: List[str]
    - disease: str               (primary disease name — used as gold label)

Split: 9/10 train (distributed across hospitals), 1/10 test.
Hospitals: 5 hospitals receive partitions of the train split.

Retrieval: For a query case, compute a weighted dot-product similarity between
the query's HPO embedding and each database case's HPO embedding.
  - Each case is represented as the IC-weighted mean of its HPO term embeddings.
  - Similarity = dot(query_vec, case_vec)
"""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────────────────────
# Paths (relative to project root; override via env or pass explicitly)
# ────────────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data_CROSSRARE"
_DATA_FILE = _DATA_DIR / "data_crossrare_v2.json"
_PHE2EMB_FILE = _DATA_DIR / "phe2embedding.json"
_IC_FILE = _DATA_DIR / "ic_dict.json"
_DISEASE_MAP_FILE = _DATA_DIR / "disease_mapping.json"
_PHENOTYPE_MAP_FILE = _DATA_DIR / "phenotype_mapping.json"

_HF_MAPPING_BASE = (
    "https://huggingface.co/datasets/chenxz/RareBench/resolve/main/mapping/"
)

# Files that may be absent locally and need to be downloaded from HuggingFace.
_REMOTE_FILES: Dict[Path, str] = {
    _PHE2EMB_FILE: _HF_MAPPING_BASE + "phe2embedding.json",
}


# ────────────────────────────────────────────────────────────────────────────
# Download helper
# ────────────────────────────────────────────────────────────────────────────

def _ensure_file(path: Path, url: str) -> None:
    """Download *url* to *path* if the file does not already exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[CrossRare] '{path.name}' not found locally — downloading from {url} ...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"[CrossRare] Saved to {path}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{path.name}' from {url}.\n"
            f"Please download it manually and place it at:\n  {path}\n"
            f"Original error: {exc}"
        ) from exc


def _ensure_all_remote_files() -> None:
    for path, url in _REMOTE_FILES.items():
        _ensure_file(path, url)


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


def _disease_partition_key(case: Dict) -> str:
    """Return the OMIM disease label used for disease-level partitioning."""
    disease_ids = case.get("disease_ids") or []
    if disease_ids:
        return str(disease_ids[0])
    # This fallback keeps the partitioner usable with minimally formatted data.
    return str(case.get("disease", ""))


def _skewed_dirichlet_partition(
    train_cases: List[Dict],
    num_hospitals: int,
    alpha: float,
    seed: int,
) -> List[List[Dict]]:
    """Allocate each OMIM disease across hospitals with a Dirichlet draw.

    Integer counts use largest-fractional-remainder rounding, so every case is
    assigned exactly once while retaining the sampled disease-level proportions.
    """
    if alpha <= 0:
        raise ValueError("skewed_dirichlet_alpha must be positive")

    cases_by_disease: Dict[str, List[Dict]] = {}
    for case in train_cases:
        cases_by_disease.setdefault(_disease_partition_key(case), []).append(case)

    allocation_rng = np.random.default_rng(seed)
    hospital_dbs: List[List[Dict]] = [[] for _ in range(num_hospitals)]
    for disease in sorted(cases_by_disease):
        cases = list(cases_by_disease[disease])
        # Paper protocol: shuffle each disease's cases with the experiment seed.
        random.Random(seed).shuffle(cases)
        proportions = allocation_rng.dirichlet(np.full(num_hospitals, alpha))
        expected_counts = proportions * len(cases)
        counts = np.floor(expected_counts).astype(int)
        remainder = len(cases) - int(counts.sum())
        for hospital_index in sorted(
            range(num_hospitals), key=lambda i: (-float(expected_counts[i] - counts[i]), i)
        )[:remainder]:
            counts[hospital_index] += 1

        offset = 0
        for hospital_index, count in enumerate(counts):
            hospital_dbs[hospital_index].extend(cases[offset:offset + int(count)])
            offset += int(count)
    return hospital_dbs


def split_and_partition(
    data: List[Dict],
    num_hospitals: int = 5,
    test_ratio: float = 0.1,
    val_ratio: float = 0.0,
    seed: int = 42,
    partition_strategy: str = "random",
    skewed_dirichlet_alpha: float = 0.3,
) -> Tuple:
    """Shuffle, split train/validation/test, then distribute train hospitals.

    partition_strategy:
      - "random"      : each train case is randomly assigned to one hospital
                        (sizes vary slightly around N/H).
      - "round_robin" : cases assigned in interleaved order 0,1,...,H-1,0,1,...
                        (most uniform disease distribution).
      - "skewed"      : sample a per-OMIM-disease Dirichlet allocation, using
                        ``skewed_dirichlet_alpha`` (paper setting: 0.3).

    Returns:
        hospital_dbs: List[num_hospitals] of case lists (train partition).
        If ``val_ratio`` is zero, returns ``(hospital_dbs, test_cases)`` for
        backwards compatibility. Otherwise returns
        ``(hospital_dbs, val_cases, test_cases)``.
    """
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    if test_ratio <= 0 or val_ratio < 0 or test_ratio + val_ratio >= 1:
        raise ValueError("test_ratio must be positive and test_ratio + val_ratio must be below 1")
    n_test = max(1, int(len(shuffled) * test_ratio))
    n_val = max(1, int(len(shuffled) * val_ratio)) if val_ratio else 0
    test_cases = shuffled[:n_test]
    val_cases = shuffled[n_test:n_test + n_val]
    train_cases = shuffled[n_test + n_val:]

    hospital_dbs: List[List[Dict]] = [[] for _ in range(num_hospitals)]

    if partition_strategy == "round_robin":
        for i, case in enumerate(train_cases):
            hospital_dbs[i % num_hospitals].append(case)

    elif partition_strategy == "random":
        for case in train_cases:
            hospital_dbs[rng.randrange(num_hospitals)].append(case)

    elif partition_strategy == "skewed":
        hospital_dbs = _skewed_dirichlet_partition(
            train_cases, num_hospitals, skewed_dirichlet_alpha, seed
        )

    else:
        raise ValueError(
            f"Unknown partition_strategy '{partition_strategy}'. "
            "Choose from: 'random', 'round_robin', 'skewed'."
        )

    if val_ratio:
        return hospital_dbs, val_cases, test_cases
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

    def retrieve(self, query_ids: List[str], phe2emb: Dict, ic_dict: Dict, top_k: int = 3, exclude_id: Optional[str] = None) -> List[Dict]:
        """Return top_k most similar cases from local database."""
        if not self.cases:
            return []

        q_emb = _normalize(case_embedding(query_ids, phe2emb, ic_dict))
        scores = self.embeddings @ q_emb  # [N]
        top_indices = np.argsort(scores)[::-1]
        if exclude_id is not None:
            top_indices = [i for i in top_indices if self.cases[i].get("id") != exclude_id]
        top_indices = top_indices[:top_k]
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
            # item["gold_aliases"]            — List[str] all valid aliases
            # item["hospital_cases"]          — List[List[Dict]]  (one list per hospital)
            #   each inner dict: {case_disease, case_phenotype}
            # item["question"]                — formatted text for the model
    """

    def __init__(
        self,
        num_hospitals: int = 5,
        num_active_hospitals: int = 3,
        agent_hospital_ids: Optional[List[int]] = None,
        test_ratio: float = 0.05,
        val_ratio: float = 0.05,
        top_k: int = 3,
        seed: int = 42,
        partition_strategy: str = "random",
        skewed_dirichlet_alpha: float = 0.3,
    ) -> None:
        self.num_hospitals = num_hospitals
        self.num_active_hospitals = num_active_hospitals
        self.top_k = top_k
        self.seed = seed
        if not 1 <= num_active_hospitals <= num_hospitals:
            raise ValueError("num_active_hospitals must be between 1 and num_hospitals")
        if agent_hospital_ids is None:
            agent_hospital_ids = list(range(1, num_active_hospitals + 1))
        if len(agent_hospital_ids) != num_active_hospitals:
            raise ValueError("agent_hospital_ids must contain num_active_hospitals unique one-based IDs")
        if len(set(agent_hospital_ids)) != len(agent_hospital_ids) or any(
            hospital_id < 1 or hospital_id > num_hospitals for hospital_id in agent_hospital_ids
        ):
            raise ValueError("agent_hospital_ids must be unique IDs from 1 through num_hospitals")
        self.agent_hospital_ids = tuple(sorted(agent_hospital_ids))

        print("[CrossRare] Loading data...")
        _ensure_all_remote_files()
        raw = _load_crossrare_raw()
        self.phe2emb = _load_phe2embedding()
        self.ic_dict = _load_ic_dict()

        hospital_dbs, val_cases, test_cases = split_and_partition(
            raw,
            num_hospitals=num_hospitals,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
            partition_strategy=partition_strategy,
            skewed_dirichlet_alpha=skewed_dirichlet_alpha,
        )

        sizes = [len(h) for h in hospital_dbs]
        print(
            f"[CrossRare] Strategy='{partition_strategy}' | "
            f"Skewed alpha: {skewed_dirichlet_alpha if partition_strategy == 'skewed' else 'n/a'} | "
            f"Train: {sum(sizes)} cases across {num_hospitals} hospitals "
            f"(sizes: {sizes}) | Val: {len(val_cases)} | Test: {len(test_cases)} | "
            f"Retrieval hospitals: {self.agent_hospital_ids} | Agents/query: {num_active_hospitals}"
        )

        self.retrievers = [
            HospitalRetriever(db, self.phe2emb, self.ic_dict, hospital_id=i)
            for i, db in enumerate(hospital_dbs)
        ]
        self.hospital_dbs = hospital_dbs
        self.val_cases = val_cases
        self.test_cases = test_cases

    def _active_retrievers(self) -> List[Tuple[int, HospitalRetriever]]:
        """The fixed N local hospitals that serve retrieval for this experiment."""
        return [(hospital_id - 1, self.retrievers[hospital_id - 1]) for hospital_id in self.agent_hospital_ids]

    def _make_item(self, case: Dict, exclude_self: bool) -> Dict:
        query_ids = case["phenotype_ids"]
        query_phenotypes = case["phenotypes"]
        gold = _clean_disease_name(case["disease"])
        hospital_cases = []
        hospital_ids = []
        for hospital_index, retriever in self._active_retrievers():
            retrieved = retriever.retrieve(query_ids, self.phe2emb, self.ic_dict, top_k=self.top_k,
                                           exclude_id=case.get("id") if exclude_self else None)
            hospital_cases.append([{"case_disease": _clean_disease_name(r["disease"]),
                                    "case_phenotype": ", ".join(r["phenotypes"])} for r in retrieved])
            hospital_ids.append(hospital_index + 1)
        return {"id": case.get("id", ""), "source": case.get("source", ""),
                "test_phenotypes": query_phenotypes,
                "test_phenotype_ids": query_ids, "gold": gold,
                "gold_aliases": _parse_disease_aliases(case["disease"]),
                "hospital_cases": hospital_cases, "hospital_ids": hospital_ids,
                "question": "Patient's phenotype: " + ", ".join(query_phenotypes), "solution": gold}

    def train_items(self) -> List[Dict]:
        """Query/labels from the two hospitals outside the retrieval coalition.

        With five partitions and N=3 local agents, these records are not present
        in any hospital database used by retrieval for the episode.
        """
        return [
            self._make_item(case, exclude_self=True)
            for hospital_index, database in enumerate(self.hospital_dbs, start=1)
            if hospital_index not in self.agent_hospital_ids
            for case in database
        ]

    def val_items(self) -> List[Dict]:
        """Validation episodes; no validation record belongs to a hospital DB."""
        return [self._make_item(case, exclude_self=False) for case in self.val_cases]

    def test_items(self) -> Iterable[Dict]:
        """Yield one dict per test case, with retrieval results pre-filled."""
        for case in self.test_cases:
            yield self._make_item(case, exclude_self=False)


def load_crossrare(
    num_hospitals: int = 5,
    num_active_hospitals: int = 3,
    agent_hospital_ids: Optional[List[int]] = None,
    test_ratio: float = 0.05,
    val_ratio: float = 0.05,
    top_k: int = 3,
    seed: int = 42,
    partition_strategy: str = "random",
    skewed_dirichlet_alpha: float = 0.3,
) -> Iterable[Dict]:
    """Thin wrapper for use in run.py — returns a list of test items."""
    ds = CrossRareDataset(
        num_hospitals=num_hospitals,
        num_active_hospitals=num_active_hospitals,
        agent_hospital_ids=agent_hospital_ids,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        top_k=top_k,
        seed=seed,
        partition_strategy=partition_strategy,
        skewed_dirichlet_alpha=skewed_dirichlet_alpha,
    )
    return list(ds.test_items())
