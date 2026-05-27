"""Utility functions and embedding backends for the MAPFM repair package."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from logging_config import logger

DEFAULT_MEDICAL_AREAS: List[str] = [
    # Common diseases (high prevalence, primary care)
    "Hypertension",
    "Type 2 Diabetes",
    "Influenza",
    "Common Cold",
    "Asthma",
    "Pneumonia",
    "Urinary Tract Infection",
    "Anemia",
    "Migraine",
    "Gastroenteritis",
    "Allergic Rhinitis",
    "Hypothyroidism",
    "Chronic Kidney Disease",
    "Osteoarthritis",
    "Depression",
    "Anxiety Disorder",
    # Moderate prevalence
    "Breast Cancer",
    "Prostate Cancer",
    "Stroke",
    "Skin Cancer",
    "Alzheimer's Disease",
    "Lung Cancer",
    "Colorectal Cancer",
    "High Blood Cholesterol",
    "Heart Attack",
    "Heart Failure",
    "High Blood Pressure",
    "Parkinson's Disease",
    "Leukemia",
    "Shingles",
    "Osteoporosis",
    "Diabetes",
    "Age-related Macular Degeneration",
    "Autoimmune Disorders",
    # Rare diseases (low prevalence)
    "Hemochromatosis",
    "Wilson Disease",
    "Polycythemia Vera",
    "Causes of Diabetes",
]

AUTHORITATIVE_SOURCES: tuple[str, ...] = ("UpToDate", "IEEE Xplore", "arXiv", "PubMed")


def stable_seed(text: str) -> int:
    """Generate a deterministic integer seed from text."""
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def tokenize(text: str) -> List[str]:
    """Tokenize biomedical text into a lightweight, case-normalized token list."""
    return re.findall(r"[A-Za-z][A-Za-z\-']+|\d+(?:\.\d+)?", str(text).lower())


def canonical_area(area: str) -> str:
    """Normalize area labels and collapse case-only duplicates."""
    raw = re.sub(r"\s+", " ", str(area).strip())
    if not raw:
        return "Unknown"
    mapping = {
        # Case normalization
        "breast cancer": "Breast Cancer",
        "prostate cancer": "Prostate Cancer",
        "lung cancer": "Lung Cancer",
        "age-related macular degeneration": "Age-related Macular Degeneration",
        "polycythemia vera": "Polycythemia Vera",
        "wilson disease": "Wilson Disease",
        "wilson disease ": "Wilson Disease",
        "alzheimer's disease": "Alzheimer's Disease",
        "parkinson's disease": "Parkinson's Disease",
        # Medical synonym normalization — collapse confusable aliases
        "high blood pressure": "Hypertension",
        "hypertension": "Hypertension",
        "elevated blood pressure": "Hypertension",
        "hbp": "Hypertension",
        "colon cancer": "Colorectal Cancer",
        "colorectal cancer": "Colorectal Cancer",
        "bowel cancer": "Colorectal Cancer",
        "heart attack": "Myocardial Infarction",
        "myocardial infarction": "Myocardial Infarction",
        "mi": "Myocardial Infarction",
        "cerebrovascular accident": "Stroke",
        "cva": "Stroke",
        "skin cancer": "Skin Cancer",
        "melanoma": "Melanoma",
        "basal cell carcinoma": "Skin Cancer",
    }
    key = raw.lower()
    if key in mapping:
        return mapping[key]
    return " ".join(part.capitalize() if part.lower() not in {"of", "and"} else part.lower() for part in raw.split())


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized float32 vector with zero-safe handling."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-12:
        return array.astype(np.float32)
    return (array / norm).astype(np.float32)


def legacy_hash_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Legacy deterministic sparse-hash embedding kept as an explicit fallback."""
    vector = np.zeros(dim, dtype=np.float32)
    words = tokenize(text)
    if not words:
        return vector
    for token in words:
        digest = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        tf = 1.0 + min(len(token), 12) / 12.0
        vector[index] += sign * tf
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    for token in bigrams:
        digest = hashlib.sha1(token.encode("utf-8", errors="ignore")).hexdigest()
        index = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[index] += 0.5 * sign
    return l2_normalize(vector)


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str) -> Any:
    """Lazy-load the sentence-transformers encoder."""
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - environment/model availability
        logger.warning("SentenceTransformer unavailable; using legacy fallback: {}", exc)
        return None


def _resize_embedding(vector: np.ndarray, dim: int) -> np.ndarray:
    """Resize semantic embeddings to the configured project dimension."""
    arr = np.asarray(vector, dtype=np.float32).flatten()
    if arr.size == dim:
        return l2_normalize(arr)
    if arr.size > dim:
        chunks = np.array_split(arr, dim)
        reduced = np.asarray([float(np.mean(chunk)) for chunk in chunks], dtype=np.float32)
        return l2_normalize(reduced)
    padded = np.pad(arr, (0, dim - arr.size))
    return l2_normalize(padded.astype(np.float32))


def semantic_embedding(text: str, dim: int = 128, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Encode text using MiniLM when available, otherwise use a documented fallback.

    The dimensionality is normalized back to the legacy project dimension to keep all
    original agent interfaces compatible.
    """
    model = _load_sentence_transformer(model_name)
    if model is None:
        return legacy_hash_embedding(text, dim=dim)
    encoded = model.encode([str(text)], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
    return _resize_embedding(np.asarray(encoded, dtype=np.float32), dim=dim)


def batch_semantic_embeddings(texts: Sequence[str], dim: int = 128, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Batch-encode texts and resize each vector to the configured dimension."""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    model = _load_sentence_transformer(model_name)
    if model is None:
        return np.vstack([legacy_hash_embedding(text, dim=dim) for text in texts]).astype(np.float32)
    encoded = model.encode(list(map(str, texts)), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.vstack([_resize_embedding(vector, dim=dim) for vector in encoded]).astype(np.float32)


# === MODIFIED: hash_embedding retained as compatibility alias while MiniLM semantic_embedding is used by repaired paths ===
def hash_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Compatibility alias for legacy callers; returns the semantic embedding backend."""
    return semantic_embedding(text, dim=dim)


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """Compute a temperature-scaled softmax."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    temp = max(float(temperature), 1e-6)
    arr = arr / temp
    arr = arr - np.max(arr)
    exp = np.exp(np.clip(arr, -60, 60))
    denom = float(np.sum(exp))
    if denom <= 1e-12:
        return np.ones_like(exp) / max(len(exp), 1)
    return exp / denom


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic function."""
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -60, 60))))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Compute entropy normalized to the 0-1 range."""
    p = np.asarray(list(probabilities), dtype=np.float64)
    if p.size <= 1:
        return 0.0
    p = p[p > 1e-12]
    if p.size <= 1:
        return 0.0
    entropy = -float(np.sum(p * np.log(p)))
    return float(entropy / np.log(len(p)))


def mean_or_zero(values: Sequence[float]) -> float:
    """Return the mean value or 0 for an empty sequence."""
    return float(np.mean(values)) if values else 0.0


def clip01(value: float) -> float:
    """Clip a scalar to the inclusive 0-1 interval."""
    return float(np.clip(value, 0.0, 1.0))


def hamming_distance_int(a: int, b: int) -> int:
    """Return bitwise Hamming distance between two integers."""
    return int((a ^ b).bit_count())


def simhash(text: str, bits: int = 64) -> int:
    """Compute SimHash over word tokens for conflict detection and deduplication."""
    tokens = tokenize(text)
    if not tokens:
        return 0
    accumulator = np.zeros(bits, dtype=np.int32)
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
        token_hash = int(digest, 16)
        for bit in range(bits):
            accumulator[bit] += 1 if ((token_hash >> bit) & 1) else -1
    value = 0
    for bit, weight in enumerate(accumulator):
        if weight >= 0:
            value |= (1 << bit)
    return value


def mask_query_keywords(query: str, ratio: float, rng: np.random.Generator) -> str:
    """Mask a configurable ratio of query keywords for robustness experiments."""
    words = tokenize(query)
    if not words:
        return query
    unique_keywords = [word for word in words if len(word) >= 4] or words
    n_mask = max(1, int(round(len(unique_keywords) * clip01(ratio))))
    chosen = set(rng.choice(unique_keywords, size=min(n_mask, len(unique_keywords)), replace=False).tolist())
    masked_tokens = ["[MASK]" if token.lower() in chosen else token for token in re.findall(r"\w+|[^\w\s]", query)]
    return " ".join(masked_tokens).replace(" [", "[").replace("] ", "]")


def safe_datetime(value: Any, default: Optional[datetime] = None) -> datetime:
    """Safely parse datetime-like values."""
    if isinstance(value, datetime):
        return value
    if value is None or str(value).strip() == "":
        return default or datetime.now()
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return default or datetime.now()
        return parsed.to_pydatetime()
    except Exception:
        return default or datetime.now()


def create_synthetic_medical_rows(area: str, n: int, start_id: int, source: str = "SyntheticBenchmark") -> List[Dict[str, Any]]:
    """Create deterministic synthetic rows when the medical CSV is absent."""
    templates = [
        ("What is {area}?", "{area} is a medical topic requiring evidence-based evaluation and domain-specific guidance."),
        ("What are common symptoms of {area}?", "Symptoms associated with {area} vary across patients and should be clinically assessed."),
        ("How is {area} diagnosed?", "Diagnosis of {area} combines history, examination, and appropriate tests."),
        ("What are treatments for {area}?", "Treatment for {area} depends on severity, comorbidities, and current medical guidelines."),
        ("Who is at risk for {area}?", "Risk factors for {area} may include age, genetics, lifestyle, and coexisting conditions."),
        ("How can {area} be prevented?", "Prevention of {area} relies on modifiable risk reduction and timely screening when applicable."),
    ]
    rows: List[Dict[str, Any]] = []
    for index in range(n):
        question, answer = templates[index % len(templates)]
        rows.append({
            "id": start_id + index,
            "question": question.format(area=area),
            "answer": answer.format(area=area),
            "source": source,
            "area_id": -1,
            "area": area,
            "area_output": "",
        })
    return rows


def load_medical_dataset(csv_path: Optional[Any], target_classes: int = 40, seed: int = 42) -> pd.DataFrame:
    """Load medNo.22.csv when present; otherwise build a 22-class benchmark dataset."""
    rng = np.random.default_rng(seed)
    dataframe: Optional[pd.DataFrame] = None
    path = None if csv_path is None else pd.io.common.stringify_path(csv_path)
    if path is not None:
        from pathlib import Path
        path_obj = Path(path)
        if path_obj.exists():
            for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1", "gbk"):
                try:
                    dataframe = pd.read_csv(path_obj, encoding=encoding)
                    break
                except Exception:
                    continue
    if dataframe is None or dataframe.empty:
        rows: List[Dict[str, Any]] = []
        next_id = 1
        for area in DEFAULT_MEDICAL_AREAS[:target_classes]:
            count = int(rng.integers(18, 32))
            area_rows = create_synthetic_medical_rows(area, count, next_id)
            rows.extend(area_rows)
            next_id += count
        dataframe = pd.DataFrame(rows)

    required_defaults: Dict[str, Any] = {
        "id": np.arange(1, len(dataframe) + 1),
        "question": "",
        "answer": "",
        "source": "Dataset",
        "area_id": -1,
        "area": "Unknown",
        "area_output": "",
    }
    for column, default in required_defaults.items():
        if column not in dataframe.columns:
            dataframe[column] = default

    dataframe = dataframe.copy()
    dataframe["question"] = dataframe["question"].fillna("").astype(str)
    dataframe["answer"] = dataframe["answer"].fillna("").astype(str)
    dataframe["source"] = dataframe["source"].fillna("Dataset").astype(str)
    dataframe["area"] = dataframe["area"].fillna("Unknown").astype(str).map(canonical_area)
    dataframe = dataframe.dropna(subset=["question", "answer", "area"])
    dataframe = dataframe.drop_duplicates(subset=["question", "answer", "area"], keep="first").reset_index(drop=True)

    class_counts = dataframe["area"].value_counts()
    if class_counts.size > target_classes:
        keep_areas = class_counts.head(target_classes).index.tolist()
        dataframe = dataframe[dataframe["area"].isin(keep_areas)].reset_index(drop=True)
    elif class_counts.size < target_classes:
        existing = set(class_counts.index.tolist())
        missing = [area for area in DEFAULT_MEDICAL_AREAS if area not in existing]
        next_id = int(pd.to_numeric(dataframe["id"], errors="coerce").fillna(0).max()) + 1
        supplement: List[Dict[str, Any]] = []
        for area in missing[: max(0, target_classes - class_counts.size)]:
            supplement.extend(create_synthetic_medical_rows(area, 8, next_id))
            next_id += 8
        if supplement:
            dataframe = pd.concat([dataframe, pd.DataFrame(supplement)], ignore_index=True)

    dataframe["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    dataframe["area"] = dataframe["area"].astype(str).map(canonical_area)
    dataframe = dataframe.reset_index(drop=True)
    return dataframe
