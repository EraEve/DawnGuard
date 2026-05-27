from __future__ import annotations

import hashlib
import pickle
import re
import time
import warnings
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


# ============================================================
# Global utility functions
# ============================================================
DEFAULT_MEDICAL_AREAS: List[str] = [
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
    "Causes of Diabetes",
    "High Blood Pressure",
    "Parkinson's Disease",
    "Leukemia",
    "Shingles",
    "Osteoporosis",
    "Hemochromatosis",
    "Diabetes",
    "Age-related Macular Degeneration",
    "Wilson Disease",
    "Polycythemia Vera",
    "Autoimmune Disorders",
]

AUTHORITATIVE_SOURCES: Tuple[str, ...] = ("UpToDate", "IEEE Xplore", "arXiv", "PubMed")


def stable_seed(text: str) -> int:
    """Generate a deterministic integer seed from text."""
    digest = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:8], 16)


def tokenize(text: str) -> List[str]:
    """Simple biomedical-friendly tokenizer."""
    return re.findall(r"[A-Za-z][A-Za-z\-']+|\d+(?:\.\d+)?", str(text).lower())


def canonical_area(area: str) -> str:
    """Normalize area labels and collapse case-only duplicates."""
    raw = re.sub(r"\s+", " ", str(area).strip())
    if not raw:
        return "Unknown"
    mapping = {
        "breast cancer": "Breast Cancer",
        "prostate cancer": "Prostate Cancer",
        "lung cancer": "Lung Cancer",
        "age-related macular degeneration": "Age-related Macular Degeneration",
        "polycythemia vera": "Polycythemia Vera",
        "wilson disease": "Wilson Disease",
        "wilson disease ": "Wilson Disease",
        "alzheimer's disease": "Alzheimer's Disease",
        "parkinson's disease": "Parkinson's Disease",
    }
    key = raw.lower()
    if key in mapping:
        return mapping[key]
    return " ".join(part.capitalize() if part.lower() not in {"of", "and"} else part.lower() for part in raw.split())


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def hash_embedding(text: str, dim: int = 128) -> np.ndarray:
    """Deterministic sparse-hash embedding used as a dense-retrieval simulator."""
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


def softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
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
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -60, 60))))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    p = np.asarray(list(probabilities), dtype=np.float64)
    if p.size <= 1:
        return 0.0
    p = p[p > 1e-12]
    if p.size <= 1:
        return 0.0
    entropy = -float(np.sum(p * np.log(p)))
    return float(entropy / np.log(len(p)))


def mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def hamming_distance_int(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def simhash(text: str, bits: int = 64) -> int:
    """SimHash over word tokens; used for conflict detection and deduplication."""
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
    """Randomly mask 10%-30% of query keywords for robustness experiments."""
    words = tokenize(query)
    if not words:
        return query
    unique_keywords = [w for w in words if len(w) >= 4]
    if not unique_keywords:
        unique_keywords = words
    n_mask = max(1, int(round(len(unique_keywords) * clip01(ratio))))
    chosen = set(rng.choice(unique_keywords, size=min(n_mask, len(unique_keywords)), replace=False).tolist())
    masked_tokens = ["[MASK]" if token.lower() in chosen else token for token in re.findall(r"\w+|[^\w\s]", query)]
    return " ".join(masked_tokens).replace(" [", "[").replace("] ", "]")


def safe_datetime(value: Any, default: Optional[datetime] = None) -> datetime:
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
    templates = [
        ("What is {area}?", "{area} is a medical topic requiring evidence-based evaluation and domain-specific guidance."),
        ("What are common symptoms of {area}?", "Symptoms associated with {area} vary across patients and should be clinically assessed."),
        ("How is {area} diagnosed?", "Diagnosis of {area} combines history, examination, and appropriate tests."),
        ("What are treatments for {area}?", "Treatment for {area} depends on severity, comorbidities, and current medical guidelines."),
        ("Who is at risk for {area}?", "Risk factors for {area} may include age, genetics, lifestyle, and coexisting conditions."),
        ("How can {area} be prevented?", "Prevention of {area} relies on modifiable risk reduction and timely screening when applicable."),
    ]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        q, a = templates[i % len(templates)]
        rows.append(
            {
                "id": start_id + i,
                "question": q.format(area=area),
                "answer": a.format(area=area),
                "source": source,
                "area_id": -1,
                "area": area,
                "area_output": "",
            }
        )
    return rows


def load_medical_dataset(csv_path: Optional[Path], target_classes: int = 22, seed: int = 42) -> pd.DataFrame:
    """Load medNo.22.csv when present; otherwise build a 22-class benchmark dataset."""
    rng = np.random.default_rng(seed)
    dataframe: Optional[pd.DataFrame] = None
    if csv_path is not None and csv_path.exists():
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1", "gbk"):
            try:
                dataframe = pd.read_csv(csv_path, encoding=encoding)
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


# ============================================================
# Configuration classes
# ============================================================
@dataclass
class SystemConfig:
    random_seed: int = 42
    embedding_dim: int = 128
    max_tfidf_features: int = 2400
    test_size: float = 0.20
    confidence_threshold: float = 0.62
    retrieval_relevance_threshold: float = 0.24
    verification_max_age_days: int = 365
    cost_lambda: float = 0.72
    nash_max_rounds: int = 6
    query_uncertainty_threshold: float = 0.58
    secure_gradient_noise_std: float = 0.015
    agent_timeout_seconds: float = 10.0


@dataclass
class EcosystemConfig(SystemConfig):
    enable_hitl: bool = True
    enable_tfa: bool = True
    enable_multimodal: bool = True
    enable_privacy: bool = True
    enable_maintenance: bool = True
    enable_consensus: bool = True
    enable_dynamic_kb: bool = True
    enable_distillation: bool = True
    enable_tri_source_fusion: bool = True
    raa_strategy_combo: Tuple[str, ...] = ("mixed", "rerank", "adaptive")
    short_window_hours: int = 24
    mid_window_days: int = 30
    long_window_months: int = 12
    consensus_threshold: float = 2.0 / 3.0
    topology_type: str = "star"
    central_agent: str = "DMA"
    fusion_conflict_hamming_threshold: int = 26
    fusion_duplicate_hamming_threshold: int = 4
    deep_retrieval_top_k: int = 10
    fast_retrieval_top_k: int = 5
    high_risk_tfa_threshold: float = 0.65
    source_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "human": 0.95,
            "authoritative": 0.90,
            "retrieval": 0.75,
            "synthetic": 0.55,
        }
    )


# ============================================================
# Knowledge and reporting data structures
# ============================================================
@dataclass
class RetrievalResult:
    doc_id: int
    question: str
    answer: str
    area: str
    source: str
    last_updated: datetime
    similarity: float
    method: str
    doc_weight: float = 1.0
    rerank_score: float = 0.0
    fact_score: float = 0.0
    verification_passed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "question": self.question,
            "answer": self.answer,
            "area": self.area,
            "source": self.source,
            "last_updated": self.last_updated.strftime("%Y-%m-%d"),
            "similarity": float(self.similarity),
            "method": self.method,
            "doc_weight": float(self.doc_weight),
            "rerank_score": float(self.rerank_score),
            "fact_score": float(self.fact_score),
            "verification_passed": bool(self.verification_passed),
            "metadata": dict(self.metadata),
        }


@dataclass
class HumanFeedbackRecord:
    sample_id: int
    query: str
    predicted_label: str
    corrected_label: str
    confidence: float
    retrieval_relevance: float
    feedback_score: int
    error_type: str
    correction_reason: str
    positive_doc_ids: List[int] = field(default_factory=list)
    negative_doc_ids: List[int] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentMessage:
    sender: str
    receiver: str
    payload: Dict[str, Any]
    encrypted_payload: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    route_path: List[str] = field(default_factory=list)
    delivered: bool = False
    retry_count: int = 0


# ============================================================
# Core legacy-integrated modules requested by the user
# ============================================================
class OnlineKnowledgeBaseManager:
    """
    Dynamic online knowledge base.
    Implements KB_{t+1} = KB_t ∪ {Embed(L_new)} and a feedback-driven document-weight update.
    """

    def __init__(self, config: EcosystemConfig) -> None:
        self.config = config
        self.documents: List[Dict[str, Any]] = []
        self.doc_index: Dict[int, int] = {}
        self.next_doc_id: int = 1
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[Any] = None
        self.dense_matrix: np.ndarray = np.zeros((0, config.embedding_dim), dtype=np.float32)
        self.index_version: int = 0
        self.crawl_updates: int = 0
        self.feedback_updates: int = 0
        self.last_crawl_time: Optional[datetime] = None
        self.locked: bool = False

    def build_from_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.documents = []
        self.doc_index = {}
        max_numeric_id = 0
        for _, row in dataframe.reset_index(drop=True).iterrows():
            raw_id = row.get("id", None)
            try:
                doc_id = int(raw_id)
            except Exception:
                doc_id = len(self.documents) + 1
            if doc_id in self.doc_index:
                doc_id = max(max_numeric_id + 1, len(self.documents) + 1)
            max_numeric_id = max(max_numeric_id, doc_id)
            doc = {
                "doc_id": doc_id,
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "area": canonical_area(str(row.get("area", "Unknown"))),
                "source": str(row.get("source", "Dataset")),
                "last_updated": safe_datetime(row.get("last_updated", datetime.now())),
                "doc_weight": 1.0,
                "metadata": {"area_id": row.get("area_id", -1)},
            }
            self.doc_index[doc_id] = len(self.documents)
            self.documents.append(doc)
        self.next_doc_id = max_numeric_id + 1 if max_numeric_id > 0 else len(self.documents) + 1
        self.rebuild_index()

    def _combined_texts(self) -> List[str]:
        return [f"{doc['question']} {doc['answer']} {doc['area']}" for doc in self.documents]

    def rebuild_index(self) -> None:
        texts = self._combined_texts()
        if not texts:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.dense_matrix = np.zeros((0, self.config.embedding_dim), dtype=np.float32)
            return
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_tfidf_features,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.dense_matrix = np.vstack([hash_embedding(text, self.config.embedding_dim) for text in texts]).astype(np.float32)
        self.index_version += 1

    def add_documents(self, new_documents: List[Dict[str, Any]]) -> List[int]:
        added_ids: List[int] = []
        for item in new_documents:
            doc_id = int(item.get("doc_id", self.next_doc_id))
            if doc_id in self.doc_index:
                doc_id = self.next_doc_id
            self.next_doc_id = max(self.next_doc_id, doc_id + 1)
            doc = {
                "doc_id": doc_id,
                "question": str(item.get("question", "New literature")),
                "answer": str(item.get("answer", "Evidence summary unavailable.")),
                "area": canonical_area(str(item.get("area", "Unknown"))),
                "source": str(item.get("source", "PubMed")),
                "last_updated": safe_datetime(item.get("last_updated", datetime.now())),
                "doc_weight": float(item.get("doc_weight", 1.0)),
                "metadata": dict(item.get("metadata", {})),
            }
            self.doc_index[doc_id] = len(self.documents)
            self.documents.append(doc)
            added_ids.append(doc_id)
        if added_ids:
            self.rebuild_index()
        return added_ids

    def simulate_scheduled_crawl(self, areas: Sequence[str], n_per_source: int = 1) -> List[int]:
        """Simulate authoritative crawling from UpToDate, IEEE Xplore, arXiv and PubMed."""
        areas = list(areas) if areas else DEFAULT_MEDICAL_AREAS
        rows: List[Dict[str, Any]] = []
        for source in AUTHORITATIVE_SOURCES:
            for offset in range(n_per_source):
                area = areas[(self.crawl_updates + offset + stable_seed(source)) % len(areas)]
                rows.append(
                    {
                        "question": f"Latest evidence update for {area} from {source}",
                        "answer": f"{source} reported a recent evidence-oriented update relevant to {area}; the item is treated as an authoritative external knowledge snippet.",
                        "area": area,
                        "source": source,
                        "last_updated": datetime.now(),
                        "doc_weight": 1.10,
                        "metadata": {"crawler": True, "authority": True},
                    }
                )
        added = self.add_documents(rows)
        self.crawl_updates += 1
        self.last_crawl_time = datetime.now()
        return added

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        idx = self.doc_index.get(int(doc_id))
        if idx is None:
            return None
        return self.documents[idx]

    def update_document_weight(self, doc_id: int, feedback_score: int, gradient_proxy: float, eta: float = 0.08) -> float:
        """w_new = w_old + eta * F_user * grad_proxy."""
        doc = self.get_document(doc_id)
        if doc is None:
            return 0.0
        old_weight = float(doc.get("doc_weight", 1.0))
        new_weight = old_weight + float(eta) * float(feedback_score) * float(gradient_proxy)
        doc["doc_weight"] = float(np.clip(new_weight, 0.20, 2.50))
        self.feedback_updates += 1
        return float(doc["doc_weight"])

    def retrieve_authoritative_recent(self, top_k: int = 4) -> List[RetrievalResult]:
        candidate_docs = [doc for doc in self.documents if doc.get("source", "") in AUTHORITATIVE_SOURCES]
        candidate_docs.sort(key=lambda doc: safe_datetime(doc.get("last_updated")), reverse=True)
        results: List[RetrievalResult] = []
        for doc in candidate_docs[:top_k]:
            results.append(
                RetrievalResult(
                    doc_id=int(doc["doc_id"]),
                    question=str(doc["question"]),
                    answer=str(doc["answer"]),
                    area=str(doc["area"]),
                    source=str(doc["source"]),
                    last_updated=safe_datetime(doc["last_updated"]),
                    similarity=0.50,
                    method="authoritative_feed",
                    doc_weight=float(doc.get("doc_weight", 1.0)),
                    metadata=dict(doc.get("metadata", {})),
                )
            )
        return results

    def save_snapshot(self, path: Path) -> None:
        payload = {
            "documents": self.documents,
            "next_doc_id": self.next_doc_id,
            "index_version": self.index_version,
            "crawl_updates": self.crawl_updates,
            "feedback_updates": self.feedback_updates,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    def load_snapshot(self, path: Path) -> None:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self.documents = list(payload.get("documents", []))
        self.next_doc_id = int(payload.get("next_doc_id", len(self.documents) + 1))
        self.index_version = int(payload.get("index_version", 0))
        self.crawl_updates = int(payload.get("crawl_updates", 0))
        self.feedback_updates = int(payload.get("feedback_updates", 0))
        self.doc_index = {int(doc["doc_id"]): idx for idx, doc in enumerate(self.documents)}
        self.rebuild_index()

    def get_stats(self) -> Dict[str, Any]:
        source_counts = Counter(str(doc.get("source", "Unknown")) for doc in self.documents)
        return {
            "document_count": len(self.documents),
            "index_version": self.index_version,
            "crawl_updates": self.crawl_updates,
            "feedback_updates": self.feedback_updates,
            "source_counts": dict(source_counts),
            "last_crawl_time": self.last_crawl_time.isoformat() if self.last_crawl_time else None,
        }


class TriSourceKnowledgeFusion:
    """Fuse human experience knowledge, authoritative knowledge, and RAA-retrieved knowledge."""

    def __init__(self, config: EcosystemConfig) -> None:
        self.config = config
        self.fusion_calls = 0
        self.last_summary: Dict[str, Any] = {}

    def fuse(
        self,
        human_docs: Sequence[RetrievalResult],
        authoritative_docs: Sequence[RetrievalResult],
        retrieval_docs: Sequence[RetrievalResult],
    ) -> List[RetrievalResult]:
        self.fusion_calls += 1
        weighted: List[RetrievalResult] = []
        groups = (
            (human_docs, "human"),
            (authoritative_docs, "authoritative"),
            (retrieval_docs, "retrieval"),
        )
        for docs, source_tag in groups:
            weight = float(self.config.source_weights.get(source_tag, 0.70))
            for doc in docs:
                copy_doc = RetrievalResult(**doc.__dict__)
                copy_doc.metadata = dict(doc.metadata)
                copy_doc.metadata["tri_source"] = source_tag
                copy_doc.similarity = clip01(float(doc.similarity) * weight + 0.05 * float(doc.doc_weight))
                weighted.append(copy_doc)
        weighted.sort(key=lambda item: item.similarity, reverse=True)
        self.last_summary = {
            "human_docs": len(human_docs),
            "authoritative_docs": len(authoritative_docs),
            "retrieval_docs": len(retrieval_docs),
            "fused_docs": len(weighted),
        }
        return weighted

    def get_stats(self) -> Dict[str, Any]:
        return {"fusion_calls": self.fusion_calls, "last_summary": dict(self.last_summary)}


class KnowledgeDistillationEngine:
    """Compress verified documents into compact distilled context vectors and snippets."""

    def __init__(self, config: EcosystemConfig) -> None:
        self.config = config
        self.distillation_calls = 0
        self.total_input_docs = 0
        self.total_output_vectors = 0

    def distill(self, documents: Sequence[RetrievalResult], max_docs: int = 6) -> Dict[str, Any]:
        self.distillation_calls += 1
        selected = list(documents)[:max_docs]
        self.total_input_docs += len(documents)
        embeddings = [hash_embedding(f"{doc.question} {doc.answer}", self.config.embedding_dim) for doc in selected]
        if embeddings:
            weighted = np.asarray([max(doc.similarity, 1e-3) for doc in selected], dtype=np.float32)
            matrix = np.vstack(embeddings)
            vector = l2_normalize(np.average(matrix, axis=0, weights=weighted))
        else:
            vector = np.zeros(self.config.embedding_dim, dtype=np.float32)
        distilled_text = " ".join(
            f"[{doc.area}|{doc.source}|{doc.similarity:.2f}] {doc.answer[:120]}" for doc in selected
        )
        self.total_output_vectors += 1
        return {
            "context_vector": vector,
            "distilled_text": distilled_text,
            "selected_doc_ids": [doc.doc_id for doc in selected],
            "selected_areas": [doc.area for doc in selected],
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "distillation_calls": self.distillation_calls,
            "total_input_docs": self.total_input_docs,
            "total_output_vectors": self.total_output_vectors,
        }


class ContrastiveRetrievalUpdater:
    """Contrastive retrieval feedback updater for RAA and the knowledge base."""

    def __init__(self, config: EcosystemConfig) -> None:
        self.config = config
        self.update_count = 0
        self.positive_updates = 0
        self.negative_updates = 0
        self.retrieval_margin_shift = 0.0

    def update(
        self,
        kb: OnlineKnowledgeBaseManager,
        query: str,
        positive_doc_ids: Sequence[int],
        negative_doc_ids: Sequence[int],
    ) -> Dict[str, Any]:
        self.update_count += 1
        query_strength = max(float(np.linalg.norm(hash_embedding(query, self.config.embedding_dim))), 0.1)
        for doc_id in positive_doc_ids:
            kb.update_document_weight(int(doc_id), feedback_score=1, gradient_proxy=query_strength, eta=0.08)
            self.positive_updates += 1
        for doc_id in negative_doc_ids:
            kb.update_document_weight(int(doc_id), feedback_score=-1, gradient_proxy=query_strength, eta=0.08)
            self.negative_updates += 1
        total = max(len(positive_doc_ids) + len(negative_doc_ids), 1)
        delta = (len(positive_doc_ids) - len(negative_doc_ids)) / total
        self.retrieval_margin_shift = float(np.clip(self.retrieval_margin_shift + 0.02 * delta, -0.25, 0.25))
        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "update_count": self.update_count,
            "positive_updates": self.positive_updates,
            "negative_updates": self.negative_updates,
            "retrieval_margin_shift": self.retrieval_margin_shift,
        }


class LoRAFineTuner:
    """Parameter-efficient DMA calibration simulator with privacy-aware gradient aggregation."""

    def __init__(self, config: EcosystemConfig) -> None:
        self.config = config
        self.update_count = 0
        self.calibration_factor = 1.0
        self.class_bias: Dict[str, float] = defaultdict(float)
        self.last_gradient_norm = 0.0

    def update(
        self,
        dma: "DecisionMakingAgent",
        feedback_records: Sequence[HumanFeedbackRecord],
        privacy_agent: Optional["PrivacySecurityAgent"] = None,
    ) -> Dict[str, Any]:
        if not feedback_records:
            return self.get_stats()
        self.update_count += 1
        raw_gradient = np.zeros(self.config.embedding_dim, dtype=np.float32)
        correction_counter: Counter[str] = Counter()
        confidence_gap_sum = 0.0
        for record in feedback_records:
            correction_counter[record.corrected_label] += 1
            signed = 1.0 if record.corrected_label != record.predicted_label else 0.25
            raw_gradient += signed * hash_embedding(record.query, self.config.embedding_dim)
            confidence_gap_sum += 1.0 - clip01(record.confidence)
        raw_gradient = raw_gradient / max(len(feedback_records), 1)
        secured_gradient = (
            privacy_agent.secure_federated_gradient(raw_gradient)
            if privacy_agent is not None
            else raw_gradient
        )
        self.last_gradient_norm = float(np.linalg.norm(secured_gradient))
        mean_gap = confidence_gap_sum / max(len(feedback_records), 1)
        self.calibration_factor = float(np.clip(self.calibration_factor + 0.04 * mean_gap, 0.80, 1.35))
        for label, count in correction_counter.items():
            self.class_bias[label] += float(count) / max(len(feedback_records), 1) * 0.03
        dma.apply_lora_update(self.calibration_factor, dict(self.class_bias))
        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "update_count": self.update_count,
            "calibration_factor": self.calibration_factor,
            "class_bias": dict(self.class_bias),
            "last_gradient_norm": self.last_gradient_norm,
        }


class HumanInTheLoopManager:
    """
    Three-level HITL manager:
    1) autonomous execution,
    2) human intervention decision,
    3) parameter and retrieval updates.
    """

    def __init__(self, config: EcosystemConfig) -> None:
        self.config = config
        self.feedback_records: List[HumanFeedbackRecord] = []
        self.human_memory_docs: List[RetrievalResult] = []
        self.autonomous_count = 0
        self.intervention_count = 0
        self.model_update_count = 0
        self.last_trigger_reason = ""
        self.error_distribution: Counter[str] = Counter()
        self.sample_counter = 0

    def _classify_error(
        self,
        predicted_label: str,
        corrected_label: str,
        retrieved_docs: Sequence[RetrievalResult],
    ) -> Tuple[str, str]:
        if not retrieved_docs:
            return "knowledge_missing", "No retrieved evidence; knowledge coverage is insufficient."
        correct_docs = [doc for doc in retrieved_docs if doc.area == corrected_label]
        if not correct_docs:
            return "knowledge_missing", "Retrieved evidence omitted the corrected disease area."
        misleading_docs = [doc for doc in retrieved_docs if doc.area != corrected_label and doc.similarity >= 0.55]
        if misleading_docs:
            return "knowledge_misleading", "High-scoring retrieval snippets support a competing area."
        if predicted_label != corrected_label:
            return "reasoning_bias", "Evidence exists but DMA's decision boundary requires calibration."
        return "normal", "No major error detected."

    def should_intervene(
        self,
        confidence: float,
        retrieval_relevance: float,
        risk_score: float,
        high_risk: bool,
    ) -> Tuple[bool, str]:
        reasons: List[str] = []
        if confidence < self.config.confidence_threshold:
            reasons.append(f"DMA low confidence ({confidence:.3f})")
        if retrieval_relevance < self.config.retrieval_relevance_threshold:
            reasons.append(f"RAA low relevance ({retrieval_relevance:.3f})")
        if high_risk and risk_score >= self.config.high_risk_tfa_threshold:
            reasons.append(f"TFA high-risk forecast ({risk_score:.3f})")
        triggered = bool(reasons)
        self.last_trigger_reason = " | ".join(reasons)
        return triggered, self.last_trigger_reason

    def register_human_feedback(
        self,
        query: str,
        predicted_label: str,
        corrected_label: str,
        confidence: float,
        retrieval_docs: Sequence[RetrievalResult],
        retrieval_relevance: float,
    ) -> HumanFeedbackRecord:
        self.sample_counter += 1
        error_type, reason = self._classify_error(predicted_label, corrected_label, retrieval_docs)
        self.error_distribution[error_type] += 1
        positive_doc_ids = [doc.doc_id for doc in retrieval_docs if doc.area == corrected_label]
        negative_doc_ids = [doc.doc_id for doc in retrieval_docs if doc.area != corrected_label and doc.similarity >= 0.45]
        feedback_score = 1 if predicted_label != corrected_label else 1
        record = HumanFeedbackRecord(
            sample_id=self.sample_counter,
            query=query,
            predicted_label=predicted_label,
            corrected_label=corrected_label,
            confidence=confidence,
            retrieval_relevance=retrieval_relevance,
            feedback_score=feedback_score,
            error_type=error_type,
            correction_reason=reason,
            positive_doc_ids=positive_doc_ids,
            negative_doc_ids=negative_doc_ids,
        )
        self.feedback_records.append(record)
        memory_doc = RetrievalResult(
            doc_id=-(self.sample_counter),
            question=query,
            answer=f"Human correction: classify as {corrected_label}. Reason: {reason}",
            area=corrected_label,
            source="HumanExpert",
            last_updated=datetime.now(),
            similarity=0.92,
            method="human_feedback",
            doc_weight=1.25,
            metadata={"feedback": True, "error_type": error_type},
        )
        self.human_memory_docs.append(memory_doc)
        self.intervention_count += 1
        return record

    def process_decision(
        self,
        query: str,
        prediction: str,
        confidence: float,
        retrieval_docs: Sequence[RetrievalResult],
        retrieval_relevance: float,
        risk_score: float,
        high_risk: bool,
        true_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        triggered, reason = self.should_intervene(confidence, retrieval_relevance, risk_score, high_risk)
        if not triggered or not self.config.enable_hitl:
            self.autonomous_count += 1
            return {
                "triggered": False,
                "reason": reason,
                "final_prediction": prediction,
                "final_confidence": confidence,
                "feedback_record": None,
                "intervention_layer": "autonomous_execution",
            }
        corrected_label = canonical_area(true_label) if true_label else prediction
        record = self.register_human_feedback(
            query=query,
            predicted_label=prediction,
            corrected_label=corrected_label,
            confidence=confidence,
            retrieval_docs=retrieval_docs,
            retrieval_relevance=retrieval_relevance,
        )
        final_prediction = corrected_label if true_label else prediction
        final_confidence = 0.96 if true_label else max(confidence, 0.70)
        return {
            "triggered": True,
            "reason": reason,
            "final_prediction": final_prediction,
            "final_confidence": final_confidence,
            "feedback_record": record,
            "intervention_layer": "human_review",
        }

    def update_models(
        self,
        dma: "DecisionMakingAgent",
        kb: OnlineKnowledgeBaseManager,
        lora_finetuner: LoRAFineTuner,
        retrieval_updater: ContrastiveRetrievalUpdater,
        privacy_agent: Optional["PrivacySecurityAgent"] = None,
        latest_n: int = 32,
    ) -> Dict[str, Any]:
        if not self.feedback_records:
            return {"updated": False, "reason": "no_feedback"}
        batch = self.feedback_records[-latest_n:]
        lora_stats = lora_finetuner.update(dma, batch, privacy_agent=privacy_agent)
        retrieval_updates: List[Dict[str, Any]] = []
        for record in batch:
            retrieval_updates.append(
                retrieval_updater.update(
                    kb=kb,
                    query=record.query,
                    positive_doc_ids=record.positive_doc_ids,
                    negative_doc_ids=record.negative_doc_ids,
                )
            )
        self.model_update_count += 1
        return {
            "updated": True,
            "layer": "model_parameter_update",
            "lora": lora_stats,
            "retrieval_update_count": len(retrieval_updates),
        }

    def recent_human_docs(self, top_k: int = 4) -> List[RetrievalResult]:
        return list(self.human_memory_docs[-top_k:])

    def get_stats(self) -> Dict[str, Any]:
        return {
            "autonomous_count": self.autonomous_count,
            "intervention_count": self.intervention_count,
            "model_update_count": self.model_update_count,
            "feedback_records": len(self.feedback_records),
            "error_distribution": dict(self.error_distribution),
            "last_trigger_reason": self.last_trigger_reason,
        }


# ============================================================
# MAS base classes and specialized agents
# ============================================================
class BaseAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.status = "healthy"
        self.last_heartbeat = datetime.now()
        self.total_calls = 0
        self.failures = 0

    def heartbeat(self) -> None:
        self.last_heartbeat = datetime.now()
        self.status = "healthy"

    def mark_failure(self) -> None:
        self.failures += 1
        self.status = "faulty"

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "total_calls": self.total_calls,
            "failures": self.failures,
        }


class MultimodalPerceptionAgent(BaseAgent):
    """Text + image perception with tokenizer/encoder simulation, ViT patching, and cross-attention fusion."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Perception")
        self.config = config
        rng = np.random.default_rng(config.random_seed)
        self.w_fusion = rng.normal(0.0, 1.0 / np.sqrt(config.embedding_dim), size=(2 * config.embedding_dim, config.embedding_dim)).astype(np.float32)
        self.encoding_calls = 0

    def encode_text(self, text: str) -> np.ndarray:
        # Simulates base-LLM tokenizer + encoder latent representation.
        tokens = tokenize(text)
        token_state = " ".join(tokens)
        return hash_embedding(token_state, self.config.embedding_dim)

    def _image_to_patch_embeddings(self, image_input: Any) -> np.ndarray:
        dim = self.config.embedding_dim
        if image_input is None:
            rng = np.random.default_rng(self.config.random_seed + 7)
            return rng.normal(0.0, 0.2, size=(4, dim)).astype(np.float32)
        if isinstance(image_input, dict) and "precomputed_features" in image_input:
            feature = np.asarray(image_input["precomputed_features"], dtype=np.float32).flatten()
            if feature.size < dim:
                feature = np.pad(feature, (0, dim - feature.size))
            feature = feature[:dim]
            return np.vstack([l2_normalize(np.roll(feature, shift)) for shift in range(4)]).astype(np.float32)
        array = np.asarray(image_input, dtype=np.float32)
        if array.size == 0:
            return np.zeros((4, dim), dtype=np.float32)
        flat = array.flatten()
        patch_count = 4
        patch_size = max(1, int(np.ceil(flat.size / patch_count)))
        patches: List[np.ndarray] = []
        for i in range(patch_count):
            patch = flat[i * patch_size : (i + 1) * patch_size]
            if patch.size == 0:
                patch = np.zeros(1, dtype=np.float32)
            textual_proxy = " ".join(f"{float(v):.3f}" for v in patch[:64])
            patches.append(hash_embedding(textual_proxy, dim))
        return np.vstack(patches).astype(np.float32)

    def encode_image(self, image_input: Any) -> np.ndarray:
        patches = self._image_to_patch_embeddings(image_input)
        if patches.size == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)
        return l2_normalize(np.mean(patches, axis=0))

    def cross_attention_fusion(self, text_vector: np.ndarray, image_input: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        patches = self._image_to_patch_embeddings(image_input)
        query = text_vector.reshape(1, -1)
        attention_logits = (patches @ query.T).flatten() / np.sqrt(max(self.config.embedding_dim, 1))
        attention = softmax(attention_logits)
        visual_context = np.sum(patches * attention[:, None], axis=0)
        concat_vector = np.concatenate([text_vector, visual_context]).astype(np.float32)
        # C_context = Concat(Encoder_txt(X_txt), Encoder_vis(X_img)) × W_fusion
        fused = l2_normalize(concat_vector @ self.w_fusion)
        return fused, {
            "attention_weights": attention.round(4).tolist(),
            "patch_count": int(patches.shape[0]),
            "fusion_norm": float(np.linalg.norm(fused)),
        }

    def encode(self, text: str, multimodal_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.total_calls += 1
        self.encoding_calls += 1
        self.heartbeat()
        image_input = None if multimodal_input is None else multimodal_input.get("image")
        text_vector = self.encode_text(text)
        if self.config.enable_multimodal:
            fused, metadata = self.cross_attention_fusion(text_vector, image_input)
        else:
            fused = text_vector
            metadata = {"attention_weights": [], "patch_count": 0, "fusion_norm": float(np.linalg.norm(fused))}
        return {
            "text_vector": text_vector,
            "context_vector": fused,
            "metadata": metadata,
        }

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update({"encoding_calls": self.encoding_calls})
        return status


class RetrievalAugmentedAgent(BaseAgent):
    """RAA with Mixed-RAG, Rerank-RAG, Adaptive-RAG and uncertainty-triggered deep retrieval."""

    def __init__(self, config: EcosystemConfig, kb: OnlineKnowledgeBaseManager, retrieval_updater: ContrastiveRetrievalUpdater) -> None:
        super().__init__("RAA")
        self.config = config
        self.kb = kb
        self.retrieval_updater = retrieval_updater
        self.strategy_counter: Counter[str] = Counter()
        self.deep_retrieval_count = 0
        self.last_uncertainty = 0.0
        self.last_retrieval_meta: Dict[str, Any] = {}

    def _doc_to_result(self, doc: Dict[str, Any], similarity: float, method: str) -> RetrievalResult:
        return RetrievalResult(
            doc_id=int(doc["doc_id"]),
            question=str(doc["question"]),
            answer=str(doc["answer"]),
            area=str(doc["area"]),
            source=str(doc["source"]),
            last_updated=safe_datetime(doc.get("last_updated")),
            similarity=clip01(similarity),
            method=method,
            doc_weight=float(doc.get("doc_weight", 1.0)),
            metadata=dict(doc.get("metadata", {})),
        )

    def _retrieve_tfidf(self, query: str, top_k: int) -> List[RetrievalResult]:
        if self.kb.tfidf_vectorizer is None or self.kb.tfidf_matrix is None or not self.kb.documents:
            return []
        query_vector = self.kb.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.kb.tfidf_matrix).flatten()
        order = np.argsort(similarities)[::-1][:top_k]
        results: List[RetrievalResult] = []
        for index in order:
            if similarities[index] <= 0:
                continue
            doc = self.kb.documents[int(index)]
            weighted_similarity = float(similarities[index]) * float(doc.get("doc_weight", 1.0))
            results.append(self._doc_to_result(doc, weighted_similarity, "tfidf"))
        return results

    def _retrieve_dense(self, query: str, top_k: int) -> List[RetrievalResult]:
        if self.kb.dense_matrix.size == 0 or not self.kb.documents:
            return []
        query_embedding = hash_embedding(query, self.config.embedding_dim)
        similarities = self.kb.dense_matrix @ query_embedding
        order = np.argsort(similarities)[::-1][:top_k]
        results: List[RetrievalResult] = []
        for index in order:
            doc = self.kb.documents[int(index)]
            base_similarity = (float(similarities[index]) + 1.0) / 2.0
            weighted_similarity = base_similarity * float(doc.get("doc_weight", 1.0))
            if weighted_similarity <= 0:
                continue
            results.append(self._doc_to_result(doc, weighted_similarity, "dense"))
        return results

    @staticmethod
    def _merge_results(result_batches: Sequence[Sequence[RetrievalResult]], top_k: int) -> List[RetrievalResult]:
        merged: Dict[int, RetrievalResult] = {}
        for batch in result_batches:
            for result in batch:
                current = merged.get(result.doc_id)
                if current is None:
                    merged[result.doc_id] = RetrievalResult(**result.__dict__)
                    merged[result.doc_id].metadata = dict(result.metadata)
                else:
                    current.similarity = clip01(0.5 * current.similarity + 0.5 * result.similarity + 0.05)
                    current.method = f"{current.method}+{result.method}"
        return sorted(merged.values(), key=lambda doc: doc.similarity, reverse=True)[:top_k]

    def mixed_rag(self, query: str, top_k: int) -> List[RetrievalResult]:
        sparse = self._retrieve_tfidf(query, top_k=max(top_k * 2, 4))
        dense = self._retrieve_dense(query, top_k=max(top_k * 2, 4))
        merged = self._merge_results([sparse, dense], top_k=top_k)
        for doc in merged:
            doc.method = "Mixed-RAG"
        return merged

    def rerank_rag(self, query: str, top_k: int) -> List[RetrievalResult]:
        candidates = self.mixed_rag(query, top_k=max(top_k * 3, 6))
        query_tokens = set(tokenize(query))
        reranked: List[RetrievalResult] = []
        now = datetime.now()
        for doc in candidates:
            doc_tokens = set(tokenize(f"{doc.question} {doc.answer}"))
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            recency_days = max((now - doc.last_updated).days, 0)
            recency_score = float(np.exp(-recency_days / 540.0))
            authority_bonus = 0.10 if doc.source in AUTHORITATIVE_SOURCES else 0.0
            doc.rerank_score = clip01(0.62 * doc.similarity + 0.22 * overlap + 0.10 * recency_score + authority_bonus)
            doc.similarity = doc.rerank_score
            doc.method = "Rerank-RAG"
            reranked.append(doc)
        return sorted(reranked, key=lambda item: item.rerank_score, reverse=True)[:top_k]

    def compute_query_uncertainty(self, query: str) -> float:
        preview = self.mixed_rag(query, top_k=5)
        if not preview:
            self.last_uncertainty = 1.0
            return 1.0
        scores = [doc.similarity for doc in preview]
        probabilities = softmax(scores, temperature=0.35)
        entropy_component = normalized_entropy(probabilities)
        if len(probabilities) >= 2:
            gap = float(probabilities[0] - probabilities[1])
        else:
            gap = float(probabilities[0])
        gap_component = 1.0 - clip01(gap)
        OOV_proxy = 1.0 if len(tokenize(query)) <= 2 else 0.0
        uncertainty = clip01(0.58 * entropy_component + 0.32 * gap_component + 0.10 * OOV_proxy)
        self.last_uncertainty = uncertainty
        return uncertainty

    def adaptive_rag(self, query: str, top_k: int) -> List[RetrievalResult]:
        uncertainty = self.compute_query_uncertainty(query)
        if uncertainty > self.config.query_uncertainty_threshold:
            self.deep_retrieval_count += 1
            dense = self._retrieve_dense(query, top_k=self.config.deep_retrieval_top_k * 2)
            sparse = self._retrieve_tfidf(query, top_k=self.config.deep_retrieval_top_k * 2)
            reranked = self.rerank_rag(query, top_k=self.config.deep_retrieval_top_k)
            diverse = self._merge_results([dense, sparse, reranked], top_k=max(top_k, self.config.deep_retrieval_top_k))
            seen_areas: set[str] = set()
            diversified: List[RetrievalResult] = []
            leftovers: List[RetrievalResult] = []
            for doc in diverse:
                doc.method = "Adaptive-RAG:Deep Retrieval"
                if doc.area not in seen_areas:
                    diversified.append(doc)
                    seen_areas.add(doc.area)
                else:
                    leftovers.append(doc)
            output = (diversified + leftovers)[:max(top_k, self.config.deep_retrieval_top_k)]
        else:
            output = self.mixed_rag(query, top_k=top_k)
            for doc in output:
                doc.method = "Adaptive-RAG:Fast Retrieval"
        return output[:top_k if uncertainty <= self.config.query_uncertainty_threshold else max(top_k, self.config.deep_retrieval_top_k)]

    def retrieve(self, query: str, strategy: str = "adaptive", top_k: Optional[int] = None) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        self.total_calls += 1
        self.heartbeat()
        strategy_key = strategy.lower().replace("-rag", "").strip()
        top_k = int(top_k or self.config.fast_retrieval_top_k)
        start = time.time()
        if strategy_key == "mixed":
            results = self.mixed_rag(query, top_k=top_k)
            uncertainty = self.compute_query_uncertainty(query)
            display_strategy = "Mixed-RAG"
        elif strategy_key == "rerank":
            results = self.rerank_rag(query, top_k=top_k)
            uncertainty = self.compute_query_uncertainty(query)
            display_strategy = "Rerank-RAG"
        else:
            results = self.adaptive_rag(query, top_k=top_k)
            uncertainty = self.last_uncertainty
            display_strategy = "Adaptive-RAG"
        elapsed = time.time() - start
        self.strategy_counter[display_strategy] += 1
        relevance = mean_or_zero([doc.similarity for doc in results])
        metadata = {
            "strategy": display_strategy,
            "uncertainty": uncertainty,
            "deep_retrieval": bool(uncertainty > self.config.query_uncertainty_threshold),
            "result_count": len(results),
            "avg_relevance": relevance,
            "latency_seconds": elapsed,
        }
        self.last_retrieval_meta = metadata
        return results, metadata

    def estimate_cost(self, strategy: str, doc_count: int) -> float:
        costs = {"mixed": 0.22, "rerank": 0.36, "adaptive": 0.28}
        base = float(costs.get(strategy.lower().replace("-rag", ""), 0.28))
        diversity_penalty = 0.012 * max(doc_count, 0)
        deep_penalty = 0.08 if strategy.lower().startswith("adaptive") and self.last_uncertainty > self.config.query_uncertainty_threshold else 0.0
        return float(base + diversity_penalty + deep_penalty)

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update(
            {
                "strategy_counter": dict(self.strategy_counter),
                "deep_retrieval_count": self.deep_retrieval_count,
                "last_uncertainty": self.last_uncertainty,
                "last_retrieval_meta": dict(self.last_retrieval_meta),
            }
        )
        return status


class DecisionMakingAgent(BaseAgent):
    """DMA simulator for Llama-3-8B / Meditron-7B with confidence output."""

    def __init__(self, config: EcosystemConfig, base_model: str = "Meditron-7B") -> None:
        super().__init__("DMA")
        self.config = config
        self.base_model = base_model
        self.labels: List[str] = []
        self.class_centroids: Dict[str, np.ndarray] = {}
        self.class_keyword_cache: Dict[str, set[str]] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.calibration_factor = 1.0
        self.class_bias: Dict[str, float] = defaultdict(float)
        self.decision_history: List[Dict[str, Any]] = []
        self.inference_calls = 0

    def fit(self, train_df: pd.DataFrame) -> None:
        docs = [f"{row.question} {row.answer}" for row in train_df.itertuples(index=False)]
        labels = [canonical_area(str(row.area)) for row in train_df.itertuples(index=False)]
        self.labels = sorted(set(labels))
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_tfidf_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.98,
            sublinear_tf=True,
        )
        matrix = self.vectorizer.fit_transform(docs)
        label_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)
        self.class_centroids = {}
        self.class_keyword_cache = {}
        for label, indices in label_to_indices.items():
            centroid = np.asarray(matrix[indices].mean(axis=0)).flatten().astype(np.float32)
            self.class_centroids[label] = l2_normalize(centroid)
            self.class_keyword_cache[label] = set(tokenize(label))

    def apply_lora_update(self, calibration_factor: float, class_bias: Dict[str, float]) -> None:
        self.calibration_factor = float(calibration_factor)
        for label, value in class_bias.items():
            self.class_bias[label] = float(value)

    def _query_centroid_scores(self, query: str) -> Dict[str, float]:
        if self.vectorizer is None or not self.class_centroids:
            return {label: 0.0 for label in self.labels}
        q = self.vectorizer.transform([query]).toarray().flatten().astype(np.float32)
        q = l2_normalize(q)
        scores: Dict[str, float] = {}
        for label in self.labels:
            centroid = self.class_centroids.get(label)
            scores[label] = float(np.dot(q, centroid)) if centroid is not None else 0.0
        return scores

    def _retrieval_scores(self, docs: Sequence[RetrievalResult]) -> Dict[str, float]:
        scores: Dict[str, float] = defaultdict(float)
        for doc in docs:
            scores[doc.area] += float(doc.similarity) * float(doc.doc_weight)
        total = sum(scores.values())
        if total > 1e-12:
            return {label: value / total for label, value in scores.items()}
        return dict(scores)

    def _lexical_label_prior(self, query: str) -> Dict[str, float]:
        q_tokens = set(tokenize(query))
        priors: Dict[str, float] = {}
        for label in self.labels:
            label_tokens = self.class_keyword_cache.get(label, set())
            priors[label] = len(q_tokens & label_tokens) / max(len(label_tokens), 1)
        return priors

    def _decision_mode_factor(self, decision_mode: str) -> float:
        mode = decision_mode.lower()
        if mode == "fast":
            return 0.94
        if mode == "deep":
            return 1.06
        return 1.0

    def infer(
        self,
        query: str,
        context_vector: Optional[np.ndarray],
        verified_docs: Sequence[RetrievalResult],
        tfa_prediction: Optional[Dict[str, Any]] = None,
        decision_mode: str = "balanced",
    ) -> Dict[str, Any]:
        self.total_calls += 1
        self.inference_calls += 1
        self.heartbeat()
        centroid_scores = self._query_centroid_scores(query)
        retrieval_scores = self._retrieval_scores(verified_docs)
        lexical_priors = self._lexical_label_prior(query)
        context_strength = float(np.linalg.norm(context_vector)) if context_vector is not None else 0.0
        tfa_risk = 0.0
        if tfa_prediction:
            tfa_risk = float(tfa_prediction.get("short_term", {}).get("risk_probability", 0.0))
        raw_scores: Dict[str, float] = {}
        for label in self.labels:
            raw = (
                0.50 * centroid_scores.get(label, 0.0)
                + 0.34 * retrieval_scores.get(label, 0.0)
                + 0.12 * lexical_priors.get(label, 0.0)
                + 0.02 * min(context_strength, 1.0)
                + 0.02 * tfa_risk
                + float(self.class_bias.get(label, 0.0))
            )
            raw_scores[label] = raw * self._decision_mode_factor(decision_mode)
        if not raw_scores:
            prediction = "Unknown"
            confidence = 0.0
            probabilities: Dict[str, float] = {}
        else:
            labels = list(raw_scores.keys())
            probability_values = softmax([raw_scores[label] for label in labels], temperature=0.18)
            probabilities = {label: float(prob) for label, prob in zip(labels, probability_values)}
            prediction = max(probabilities, key=probabilities.get)
            confidence = clip01(float(probabilities[prediction]) * self.calibration_factor)
        output = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "decision_mode": decision_mode,
            "base_model": self.base_model,
            "raw_scores": raw_scores,
        }
        self.decision_history.append(
            {
                "prediction": prediction,
                "confidence": confidence,
                "mode": decision_mode,
                "time": datetime.now().isoformat(),
            }
        )
        return output

    def estimate_confidence(
        self,
        query: str,
        docs: Sequence[RetrievalResult],
        decision_mode: str,
        context_vector: Optional[np.ndarray] = None,
    ) -> float:
        result = self.infer(
            query=query,
            context_vector=context_vector,
            verified_docs=docs,
            tfa_prediction=None,
            decision_mode=decision_mode,
        )
        return float(result["confidence"])

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        average_confidence = mean_or_zero([float(item["confidence"]) for item in self.decision_history])
        status.update(
            {
                "base_model": self.base_model,
                "label_count": len(self.labels),
                "inference_calls": self.inference_calls,
                "average_confidence": average_confidence,
                "calibration_factor": self.calibration_factor,
            }
        )
        return status


class TemporalForeseeingAgent(BaseAgent):
    """TFA simulator: TCN + LSTM + Transformer gated fusion for multi-scale forecasts."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("TFA")
        self.config = config
        self.forecast_calls = 0
        self.last_forecast: Dict[str, Any] = {}

    def _generate_history(self, query: str, history: Optional[Sequence[float]]) -> np.ndarray:
        if history is not None and len(history) > 0:
            arr = np.asarray(history, dtype=np.float32).flatten()
        else:
            rng = np.random.default_rng(stable_seed(query) + self.config.random_seed)
            arr = rng.normal(loc=0.0, scale=0.8, size=72).astype(np.float32)
            arr += np.linspace(-0.15, 0.20, num=72).astype(np.float32)
        return arr

    def _tcn_feature(self, series: np.ndarray) -> float:
        if series.size < 3:
            return float(np.mean(series)) if series.size else 0.0
        local = series[-min(series.size, 24) :]
        diffs = np.diff(local)
        return float(np.mean(diffs) + 0.30 * np.std(local))

    def _lstm_feature(self, series: np.ndarray) -> float:
        state = 0.0
        alpha = 0.18
        for value in series:
            state = alpha * float(value) + (1.0 - alpha) * state
        return float(state)

    def _transformer_feature(self, series: np.ndarray) -> float:
        if series.size == 0:
            return 0.0
        positions = np.linspace(0.0, 1.0, num=series.size)
        logits = positions + 0.20 * series
        attention = softmax(logits, temperature=0.45)
        return float(np.sum(series * attention))

    def _authority_calibration(self, authoritative_signal: Optional[Dict[str, float]]) -> float:
        if not authoritative_signal:
            return 0.0
        contributions = [float(value) for value in authoritative_signal.values()]
        return float(np.clip(np.mean(contributions), -0.25, 0.25))

    def forecast(
        self,
        query: str,
        history: Optional[Sequence[float]] = None,
        authoritative_signal: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        self.total_calls += 1
        self.forecast_calls += 1
        self.heartbeat()
        series = self._generate_history(query, history)
        tcn = self._tcn_feature(series)
        lstm = self._lstm_feature(series)
        transformer = self._transformer_feature(series)
        gate_logits = np.asarray([abs(tcn), abs(lstm), abs(transformer)], dtype=np.float32)
        gates = softmax(gate_logits, temperature=0.35)
        fused = float(gates[0] * tcn + gates[1] * lstm + gates[2] * transformer)
        authority_bias = self._authority_calibration(authoritative_signal)
        base_risk = sigmoid(fused + authority_bias)
        short_risk = clip01(base_risk)
        mid_risk = clip01(0.72 * base_risk + 0.28 * sigmoid(float(np.mean(series))))
        long_risk = clip01(0.58 * base_risk + 0.42 * sigmoid(float(np.mean(series[-36:]))))
        output = {
            "short_term": {
                "window": f"{self.config.short_window_hours}h",
                "risk_probability": short_risk,
            },
            "mid_term": {
                "window": f"{self.config.mid_window_days}d",
                "risk_probability": mid_risk,
            },
            "long_term": {
                "window": f"{self.config.long_window_months}m",
                "risk_probability": long_risk,
            },
            "conditional_probability_distribution": {
                "deterioration": short_risk,
                "stable": clip01(1.0 - short_risk),
            },
            "contribution_factors": {
                "TCN": float(gates[0]),
                "LSTM": float(gates[1]),
                "Transformer": float(gates[2]),
                "authority_bias": authority_bias,
            },
        }
        self.last_forecast = output
        return output

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update({"forecast_calls": self.forecast_calls, "last_forecast": dict(self.last_forecast)})
        return status


class KnowledgeFusionAgent(BaseAgent):
    """Conflict detection, redundancy removal, and consistency fusion over multi-retrieval outputs."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Fusion")
        self.config = config
        self.fusion_calls = 0
        self.conflict_count = 0
        self.dedup_removed = 0
        self.last_summary: Dict[str, Any] = {}

    def fuse(self, retrieval_batches: Sequence[Sequence[RetrievalResult]]) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        self.total_calls += 1
        self.fusion_calls += 1
        self.heartbeat()
        flattened: List[RetrievalResult] = []
        for batch in retrieval_batches:
            flattened.extend(batch)
        ranked = sorted(flattened, key=lambda doc: doc.similarity, reverse=True)
        unique_docs: List[RetrievalResult] = []
        fingerprints: List[Tuple[int, RetrievalResult]] = []
        for doc in ranked:
            fingerprint = simhash(f"{doc.question} {doc.answer}")
            duplicate = False
            for existing_fp, existing_doc in fingerprints:
                if doc.doc_id == existing_doc.doc_id or hamming_distance_int(fingerprint, existing_fp) <= self.config.fusion_duplicate_hamming_threshold:
                    duplicate = True
                    self.dedup_removed += 1
                    break
            if not duplicate:
                unique_docs.append(doc)
                fingerprints.append((fingerprint, doc))
        conflicts: List[Dict[str, Any]] = []
        top_docs = unique_docs[: min(len(unique_docs), 8)]
        for i in range(len(top_docs)):
            for j in range(i + 1, len(top_docs)):
                left = top_docs[i]
                right = top_docs[j]
                if left.area == right.area:
                    continue
                left_fp = simhash(f"{left.question} {left.answer}")
                right_fp = simhash(f"{right.question} {right.answer}")
                distance = hamming_distance_int(left_fp, right_fp)
                if distance > self.config.fusion_conflict_hamming_threshold:
                    conflicts.append(
                        {
                            "left_doc_id": left.doc_id,
                            "right_doc_id": right.doc_id,
                            "left_area": left.area,
                            "right_area": right.area,
                            "simhash_distance": distance,
                        }
                    )
        self.conflict_count += len(conflicts)
        area_scores: Dict[str, List[float]] = defaultdict(list)
        for doc in unique_docs:
            area_scores[doc.area].append(float(doc.similarity))
        area_consensus = {area: float(np.mean(scores)) for area, scores in area_scores.items()}
        fused_docs = sorted(unique_docs, key=lambda doc: (area_consensus.get(doc.area, 0.0), doc.similarity), reverse=True)
        summary = {
            "input_docs": len(flattened),
            "deduplicated_docs": len(fused_docs),
            "dedup_removed": len(flattened) - len(fused_docs),
            "conflict_count": len(conflicts),
            "conflicts": conflicts[:5],
            "area_consensus": dict(sorted(area_consensus.items(), key=lambda item: item[1], reverse=True)[:5]),
        }
        self.last_summary = summary
        return fused_docs, summary

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update(
            {
                "fusion_calls": self.fusion_calls,
                "conflict_count": self.conflict_count,
                "dedup_removed": self.dedup_removed,
                "last_summary": dict(self.last_summary),
            }
        )
        return status


class KnowledgeVerificationAgent(BaseAgent):
    """Medical fact-check and timeliness verifier."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Verification")
        self.config = config
        self.verification_calls = 0
        self.accepted_docs = 0
        self.rejected_docs = 0
        self.last_summary: Dict[str, Any] = {}

    def _fact_api_score(self, doc: RetrievalResult) -> float:
        # Simulated medical fact API / rule base.
        text = f"{doc.question} {doc.answer}".lower()
        authority_bonus = 0.22 if doc.source in AUTHORITATIVE_SOURCES or doc.source == "HumanExpert" else 0.0
        content_bonus = 0.18 if len(tokenize(text)) >= 8 else 0.0
        caution_penalty = 0.25 if any(keyword in text for keyword in ["miracle", "guaranteed cure", "always works"]) else 0.0
        area_bonus = 0.10 if doc.area != "Unknown" else 0.0
        return clip01(0.45 + authority_bonus + content_bonus + area_bonus - caution_penalty)

    def verify(self, docs: Sequence[RetrievalResult]) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        self.total_calls += 1
        self.verification_calls += 1
        self.heartbeat()
        now = datetime.now()
        accepted: List[RetrievalResult] = []
        rejected: List[Dict[str, Any]] = []
        for doc in docs:
            age_days = max((now - doc.last_updated).days, 0)
            fact_score = self._fact_api_score(doc)
            timeliness_ok = age_days <= self.config.verification_max_age_days
            fact_ok = fact_score >= 0.55
            doc.fact_score = fact_score
            doc.verification_passed = bool(timeliness_ok and fact_ok)
            if doc.verification_passed:
                accepted.append(doc)
            else:
                rejected.append(
                    {
                        "doc_id": doc.doc_id,
                        "area": doc.area,
                        "age_days": age_days,
                        "fact_score": fact_score,
                        "reason": "stale" if not timeliness_ok else "fact_score_low",
                    }
                )
        self.accepted_docs += len(accepted)
        self.rejected_docs += len(rejected)
        summary = {
            "accepted_docs": len(accepted),
            "rejected_docs": len(rejected),
            "verification_ratio": len(accepted) / max(len(docs), 1),
            "rejected_examples": rejected[:5],
        }
        self.last_summary = summary
        return accepted, summary

    def vote(self, verification_summary: Dict[str, Any]) -> bool:
        return float(verification_summary.get("verification_ratio", 0.0)) >= 0.50

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update(
            {
                "verification_calls": self.verification_calls,
                "accepted_docs": self.accepted_docs,
                "rejected_docs": self.rejected_docs,
                "last_summary": dict(self.last_summary),
            }
        )
        return status


class PrivacySecurityAgent(BaseAgent):
    """Privacy and security layer: noisy federated gradients and encrypted inter-agent communication."""

    def __init__(self, config: EcosystemConfig, key: str = "heterogeneous-mas-key") -> None:
        super().__init__("Privacy")
        self.config = config
        self.key = hashlib.sha256(key.encode("utf-8", errors="ignore")).digest()
        self.encryption_calls = 0
        self.decryption_calls = 0
        self.gradient_calls = 0

    def encrypt_message(self, plaintext: str) -> str:
        self.total_calls += 1
        self.encryption_calls += 1
        raw = plaintext.encode("utf-8", errors="ignore")
        encrypted = bytes(byte ^ self.key[idx % len(self.key)] for idx, byte in enumerate(raw))
        self.heartbeat()
        return encrypted.hex()

    def decrypt_message(self, ciphertext: str) -> str:
        self.total_calls += 1
        self.decryption_calls += 1
        raw = bytes.fromhex(ciphertext)
        decrypted = bytes(byte ^ self.key[idx % len(self.key)] for idx, byte in enumerate(raw))
        self.heartbeat()
        return decrypted.decode("utf-8", errors="ignore")

    def secure_federated_gradient(self, gradient: np.ndarray) -> np.ndarray:
        self.total_calls += 1
        self.gradient_calls += 1
        rng = np.random.default_rng(self.config.random_seed + self.gradient_calls)
        noise = rng.normal(0.0, self.config.secure_gradient_noise_std, size=gradient.shape).astype(np.float32)
        # Simulates encrypted / noisy aggregation without exposing exact client gradients.
        secured = gradient.astype(np.float32) + noise
        self.heartbeat()
        return secured

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update(
            {
                "encryption_calls": self.encryption_calls,
                "decryption_calls": self.decryption_calls,
                "gradient_calls": self.gradient_calls,
            }
        )
        return status


class MaintenanceAgent(BaseAgent):
    """Heartbeat monitoring, fault detection, isolation, and restart simulation."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Maintenance")
        self.config = config
        self.registered_agents: Dict[str, BaseAgent] = {}
        self.isolated_agents: Dict[str, str] = {}
        self.restart_count = 0
        self.monitor_count = 0
        self.abnormal_outputs = 0

    def register(self, agent: BaseAgent) -> None:
        self.registered_agents[agent.name] = agent

    def monitor(self) -> Dict[str, Any]:
        self.total_calls += 1
        self.monitor_count += 1
        self.heartbeat()
        now = datetime.now()
        alerts: List[Dict[str, Any]] = []
        for name, agent in self.registered_agents.items():
            age = (now - agent.last_heartbeat).total_seconds()
            if age > self.config.agent_timeout_seconds or agent.status == "faulty":
                self.isolated_agents[name] = "timeout" if age > self.config.agent_timeout_seconds else "faulty"
                alerts.append({"agent": name, "reason": self.isolated_agents[name], "age_seconds": age})
        return {"alerts": alerts, "isolated_agents": dict(self.isolated_agents)}

    def detect_abnormal_output(self, agent_name: str, output: Any) -> bool:
        abnormal = output is None
        if isinstance(output, float):
            abnormal = abnormal or bool(np.isnan(output)) or bool(np.isinf(output))
        if isinstance(output, dict) and "confidence" in output:
            confidence = output.get("confidence")
            abnormal = abnormal or confidence is None or bool(np.isnan(float(confidence)))
        if abnormal:
            self.abnormal_outputs += 1
            self.isolated_agents[agent_name] = "abnormal_output"
            agent = self.registered_agents.get(agent_name)
            if agent is not None:
                agent.mark_failure()
        return abnormal

    def restart_or_replace(self, agent_name: str) -> bool:
        agent = self.registered_agents.get(agent_name)
        if agent is None:
            return False
        agent.status = "healthy"
        agent.heartbeat()
        self.isolated_agents.pop(agent_name, None)
        self.restart_count += 1
        return True

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update(
            {
                "registered_agents": list(self.registered_agents.keys()),
                "isolated_agents": dict(self.isolated_agents),
                "restart_count": self.restart_count,
                "monitor_count": self.monitor_count,
                "abnormal_outputs": self.abnormal_outputs,
            }
        )
        return status


class CommunicationTopologyManager(BaseAgent):
    """Star / mesh topology switcher with encrypted routing, neighbor discovery, and retry queues."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Topology")
        self.config = config
        self.topology_type = config.topology_type
        self.central_agent = config.central_agent
        self.nodes: set[str] = set()
        self.links: Dict[str, set[str]] = defaultdict(set)
        self.retry_queue: Deque[AgentMessage] = deque()
        self.routed_messages = 0
        self.delivery_failures = 0

    def register_node(self, node_name: str) -> None:
        self.nodes.add(node_name)
        self._rebuild_links()

    def _rebuild_links(self) -> None:
        self.links = defaultdict(set)
        if self.topology_type == "mesh":
            for left in self.nodes:
                for right in self.nodes:
                    if left != right:
                        self.links[left].add(right)
        else:
            for node in self.nodes:
                if node != self.central_agent:
                    self.links[node].add(self.central_agent)
                    self.links[self.central_agent].add(node)

    def switch_topology(self, topology_type: str) -> None:
        requested = topology_type.lower().strip()
        self.topology_type = "mesh" if requested == "mesh" else "star"
        self._rebuild_links()

    def discover_neighbors(self, node_name: str) -> List[str]:
        return sorted(self.links.get(node_name, set()))

    def _route_path(self, sender: str, receiver: str) -> List[str]:
        if receiver in self.links.get(sender, set()):
            return [sender, receiver]
        if self.topology_type == "star" and sender != self.central_agent and receiver != self.central_agent:
            if self.central_agent in self.links.get(sender, set()) and receiver in self.links.get(self.central_agent, set()):
                return [sender, self.central_agent, receiver]
        return []

    def route_message(
        self,
        sender: str,
        receiver: str,
        payload: Dict[str, Any],
        privacy_agent: Optional[PrivacySecurityAgent] = None,
    ) -> AgentMessage:
        self.total_calls += 1
        self.routed_messages += 1
        self.heartbeat()
        message = AgentMessage(sender=sender, receiver=receiver, payload=payload)
        message.route_path = self._route_path(sender, receiver)
        if privacy_agent is not None and self.config.enable_privacy:
            message.encrypted_payload = privacy_agent.encrypt_message(str(payload))
            _ = privacy_agent.decrypt_message(message.encrypted_payload)
        if message.route_path:
            message.delivered = True
        else:
            message.delivered = False
            self.delivery_failures += 1
            self.retry_queue.append(message)
        return message

    def retry_disconnected(self, privacy_agent: Optional[PrivacySecurityAgent] = None) -> List[AgentMessage]:
        retried: List[AgentMessage] = []
        queue_size = len(self.retry_queue)
        for _ in range(queue_size):
            message = self.retry_queue.popleft()
            message.retry_count += 1
            message.route_path = self._route_path(message.sender, message.receiver)
            if message.route_path:
                message.delivered = True
                if privacy_agent is not None and self.config.enable_privacy and message.encrypted_payload is None:
                    message.encrypted_payload = privacy_agent.encrypt_message(str(message.payload))
                retried.append(message)
            else:
                self.retry_queue.append(message)
        return retried

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update(
            {
                "topology_type": self.topology_type,
                "nodes": sorted(self.nodes),
                "links": {node: sorted(neighbors) for node, neighbors in self.links.items()},
                "routed_messages": self.routed_messages,
                "delivery_failures": self.delivery_failures,
                "retry_queue_size": len(self.retry_queue),
            }
        )
        return status


class ConsensusModule(BaseAgent):
    """Simplified PBFT/Raft-like voting for high-risk decisions."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Consensus")
        self.config = config
        self.consensus_calls = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.last_vote: Dict[str, Any] = {}

    def vote(
        self,
        dma_result: Dict[str, Any],
        tfa_prediction: Optional[Dict[str, Any]],
        verification_summary: Dict[str, Any],
        verification_agent: KnowledgeVerificationAgent,
    ) -> Dict[str, Any]:
        self.total_calls += 1
        self.consensus_calls += 1
        self.heartbeat()
        confidence = float(dma_result.get("confidence", 0.0))
        risk = float((tfa_prediction or {}).get("short_term", {}).get("risk_probability", 0.0))
        dma_vote = confidence >= self.config.confidence_threshold
        tfa_vote = risk >= 0.45
        verification_vote = verification_agent.vote(verification_summary)
        votes = {
            "DMA": dma_vote,
            "TFA": tfa_vote,
            "Verification": verification_vote,
        }
        approvals = sum(1 for value in votes.values() if value)
        required = int(np.ceil(len(votes) * self.config.consensus_threshold))
        approved = approvals >= required
        if approved:
            self.approved_count += 1
        else:
            self.rejected_count += 1
        result = {
            "approved": approved,
            "votes": votes,
            "approval_count": approvals,
            "required_approvals": required,
            "consensus_threshold": self.config.consensus_threshold,
        }
        self.last_vote = result
        return result

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update(
            {
                "consensus_calls": self.consensus_calls,
                "approved_count": self.approved_count,
                "rejected_count": self.rejected_count,
                "last_vote": dict(self.last_vote),
            }
        )
        return status


# ============================================================
# Main Heterogeneous Multi-Agent Ecosystem
# ============================================================
class HeterogeneousMultiAgentEcosystem:
    """Complete MAPFM-inspired heterogeneous multi-agent medical AI ecosystem."""

    def __init__(
        self,
        config: EcosystemConfig,
        dataset: pd.DataFrame,
        base_model: str = "Meditron-7B",
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.verbose = verbose
        np.random.seed(config.random_seed)
        self.dataset = dataset.copy().reset_index(drop=True)
        self.dataset["area"] = self.dataset["area"].astype(str).map(canonical_area)
        self.train_df, self.test_df = self._split_dataset(self.dataset)
        self.label_set = sorted(self.dataset["area"].unique().tolist())
        self.online_kb = OnlineKnowledgeBaseManager(config)
        self.online_kb.build_from_dataframe(self.train_df)
        self.contrastive_updater = ContrastiveRetrievalUpdater(config)
        self.lora_finetuner = LoRAFineTuner(config)
        self.tri_source_fusion = TriSourceKnowledgeFusion(config)
        self.distillation_engine = KnowledgeDistillationEngine(config)
        self.hitl_manager = HumanInTheLoopManager(config)

        self.perception_agent = MultimodalPerceptionAgent(config)
        self.raa = RetrievalAugmentedAgent(config, self.online_kb, self.contrastive_updater)
        self.dma = DecisionMakingAgent(config, base_model=base_model)
        self.dma.fit(self.train_df)
        self.tfa = TemporalForeseeingAgent(config)
        self.fusion_agent = KnowledgeFusionAgent(config)
        self.verification_agent = KnowledgeVerificationAgent(config)
        self.privacy_agent = PrivacySecurityAgent(config)
        self.maintenance_agent = MaintenanceAgent(config)
        self.topology_manager = CommunicationTopologyManager(config)
        self.consensus_module = ConsensusModule(config)

        self._register_agents()
        self.task_history: List[Dict[str, Any]] = []
        self.game_history: List[Dict[str, Any]] = []
        self.ablation_results: Dict[str, Any] = {}
        self.robustness_results: Dict[str, Any] = {}
        self.adaptability_results: Dict[str, Any] = {}
        self.common_rare_results: Dict[str, Any] = {}

    def _split_dataset(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        counts = dataframe["area"].value_counts()
        stratify = dataframe["area"] if counts.min() >= 2 else None
        train_df, test_df = train_test_split(
            dataframe,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=stratify,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def _register_agents(self) -> None:
        agents: List[BaseAgent] = [
            self.perception_agent,
            self.raa,
            self.dma,
            self.tfa,
            self.fusion_agent,
            self.verification_agent,
            self.privacy_agent,
            self.maintenance_agent,
            self.topology_manager,
            self.consensus_module,
        ]
        for agent in agents:
            self.maintenance_agent.register(agent)
            self.topology_manager.register_node(agent.name)

    def refresh_online_knowledge(self, n_per_source: int = 1) -> List[int]:
        if not self.config.enable_dynamic_kb:
            return []
        added = self.online_kb.simulate_scheduled_crawl(self.label_set, n_per_source=n_per_source)
        return added

    def _retrieval_strategy_display_to_key(self, strategy: str) -> str:
        return strategy.lower().replace("-rag", "").replace(" ", "").replace("deepretrieval", "adaptive")

    def run_game_theoretic_collaboration(
        self,
        query: str,
        context_vector: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Alternate RAA and DMA best responses until a simulated Nash equilibrium is reached.
        U(S_ret, D) = lambda * P(D|Q,K) - (1-lambda) * Cost(S_ret)
        """
        strategies = list(self.config.raa_strategy_combo)
        dma_modes = ["fast", "balanced", "deep"]
        current_strategy = strategies[0] if strategies else "adaptive"
        current_mode = "balanced"
        round_logs: List[Dict[str, Any]] = []
        retrieval_cache: Dict[str, Tuple[List[RetrievalResult], Dict[str, Any]]] = {}
        converged = False

        for round_id in range(1, self.config.nash_max_rounds + 1):
            best_raa: Optional[Tuple[str, List[RetrievalResult], Dict[str, Any], float, float]] = None
            for strategy in strategies:
                cache_key = f"{strategy}:{round_id if strategy == 'adaptive' else 0}"
                if cache_key not in retrieval_cache:
                    retrieval_cache[cache_key] = self.raa.retrieve(
                        query,
                        strategy=strategy,
                        top_k=self.config.fast_retrieval_top_k,
                    )
                docs, meta = retrieval_cache[cache_key]
                estimated_conf = self.dma.estimate_confidence(query, docs, current_mode, context_vector=context_vector)
                cost = self.raa.estimate_cost(strategy, len(docs))
                utility = self.config.cost_lambda * estimated_conf - (1.0 - self.config.cost_lambda) * cost
                candidate = (strategy, docs, meta, utility, estimated_conf)
                if best_raa is None or utility > best_raa[3]:
                    best_raa = candidate
            assert best_raa is not None
            chosen_strategy, chosen_docs, chosen_meta, chosen_utility, chosen_estimated_confidence = best_raa

            best_dma_mode: Optional[Tuple[str, float, float]] = None
            retrieval_cost = self.raa.estimate_cost(chosen_strategy, len(chosen_docs))
            for mode in dma_modes:
                estimated_conf = self.dma.estimate_confidence(query, chosen_docs, mode, context_vector=context_vector)
                utility = self.config.cost_lambda * estimated_conf - (1.0 - self.config.cost_lambda) * retrieval_cost
                candidate = (mode, utility, estimated_conf)
                if best_dma_mode is None or utility > best_dma_mode[1]:
                    best_dma_mode = candidate
            assert best_dma_mode is not None
            chosen_mode, dma_utility, dma_estimated_confidence = best_dma_mode
            round_log = {
                "round": round_id,
                "raa_strategy_before": current_strategy,
                "dma_mode_before": current_mode,
                "raa_strategy_after": chosen_strategy,
                "dma_mode_after": chosen_mode,
                "utility": float(dma_utility),
                "retrieval_cost": float(retrieval_cost),
                "estimated_confidence": float(dma_estimated_confidence),
                "uncertainty": float(chosen_meta.get("uncertainty", 0.0)),
            }
            round_logs.append(round_log)
            no_unilateral_change = chosen_strategy == current_strategy and chosen_mode == current_mode
            current_strategy = chosen_strategy
            current_mode = chosen_mode
            if no_unilateral_change:
                converged = True
                break

        final_docs, final_meta = self.raa.retrieve(
            query,
            strategy=current_strategy,
            top_k=self.config.fast_retrieval_top_k,
        )
        result = {
            "equilibrium_reached": converged,
            "equilibrium_strategy": current_strategy,
            "equilibrium_decision_mode": current_mode,
            "rounds": len(round_logs),
            "round_logs": round_logs,
            "retrieval_docs": final_docs,
            "retrieval_meta": final_meta,
        }
        self.game_history.append(
            {
                "query": query[:120],
                "strategy": current_strategy,
                "decision_mode": current_mode,
                "rounds": len(round_logs),
                "equilibrium": converged,
            }
        )
        return result

    def _authority_signal_from_docs(self, docs: Sequence[RetrievalResult]) -> Dict[str, float]:
        signals: Dict[str, float] = {}
        for source in AUTHORITATIVE_SOURCES:
            source_docs = [doc for doc in docs if doc.source == source]
            if source_docs:
                signals[source] = mean_or_zero([doc.similarity for doc in source_docs]) - 0.5
        return signals

    def _is_high_risk_query(self, query: str, tfa_prediction: Optional[Dict[str, Any]]) -> bool:
        keyword_hit = any(keyword in query.lower() for keyword in ["surgery", "urgent", "emergency", "severe", "critical"])
        tfa_risk = float((tfa_prediction or {}).get("short_term", {}).get("risk_probability", 0.0))
        return bool(keyword_hit or tfa_risk >= self.config.high_risk_tfa_threshold)

    def run_collaborative_task(
        self,
        patient_query: str,
        multimodal_input: Optional[Dict[str, Any]] = None,
        true_label: Optional[str] = None,
        force_high_risk: bool = False,
    ) -> Dict[str, Any]:
        task_started = time.time()
        self.maintenance_agent.monitor()
        perception_state = self.perception_agent.encode(patient_query, multimodal_input=multimodal_input)
        context_vector = perception_state["context_vector"]

        self.topology_manager.route_message(
            "Perception",
            "RAA",
            {"query": patient_query[:80], "context_dim": int(len(context_vector))},
            privacy_agent=self.privacy_agent,
        )

        game_result = self.run_game_theoretic_collaboration(patient_query, context_vector=context_vector)
        main_retrieval_docs: List[RetrievalResult] = game_result["retrieval_docs"]
        _, mixed_meta = self.raa.retrieve(patient_query, strategy="mixed", top_k=self.config.fast_retrieval_top_k)
        rerank_docs, rerank_meta = self.raa.retrieve(patient_query, strategy="rerank", top_k=self.config.fast_retrieval_top_k)
        adaptive_docs, adaptive_meta = self.raa.retrieve(patient_query, strategy="adaptive", top_k=self.config.fast_retrieval_top_k)
        authoritative_docs = self.online_kb.retrieve_authoritative_recent(top_k=4)
        human_docs = self.hitl_manager.recent_human_docs(top_k=4)

        if self.config.enable_tri_source_fusion:
            tri_source_docs = self.tri_source_fusion.fuse(
                human_docs=human_docs,
                authoritative_docs=authoritative_docs,
                retrieval_docs=main_retrieval_docs + rerank_docs + adaptive_docs,
            )
        else:
            tri_source_docs = main_retrieval_docs + rerank_docs + adaptive_docs
        fused_docs, fusion_summary = self.fusion_agent.fuse([tri_source_docs])
        verified_docs, verification_summary = self.verification_agent.verify(fused_docs)
        if not verified_docs:
            verified_docs = fused_docs[: self.config.fast_retrieval_top_k]

        distilled_context = (
            self.distillation_engine.distill(verified_docs)
            if self.config.enable_distillation
            else {
                "context_vector": context_vector,
                "distilled_text": "",
                "selected_doc_ids": [doc.doc_id for doc in verified_docs[:4]],
                "selected_areas": [doc.area for doc in verified_docs[:4]],
            }
        )
        fused_context = l2_normalize(0.60 * context_vector + 0.40 * np.asarray(distilled_context["context_vector"], dtype=np.float32))

        tfa_prediction: Optional[Dict[str, Any]] = None
        if self.config.enable_tfa:
            history = None if multimodal_input is None else multimodal_input.get("time_series")
            tfa_prediction = self.tfa.forecast(
                query=patient_query,
                history=history,
                authoritative_signal=self._authority_signal_from_docs(verified_docs),
            )

        high_risk = bool(force_high_risk or self._is_high_risk_query(patient_query, tfa_prediction))
        dma_result = self.dma.infer(
            query=patient_query,
            context_vector=fused_context,
            verified_docs=verified_docs,
            tfa_prediction=tfa_prediction,
            decision_mode=game_result["equilibrium_decision_mode"],
        )
        self.maintenance_agent.detect_abnormal_output("DMA", dma_result)
        retrieval_relevance = mean_or_zero([doc.similarity for doc in verified_docs])
        risk_score = float((tfa_prediction or {}).get("short_term", {}).get("risk_probability", 0.0))

        hitl_result = self.hitl_manager.process_decision(
            query=patient_query,
            prediction=str(dma_result["prediction"]),
            confidence=float(dma_result["confidence"]),
            retrieval_docs=verified_docs,
            retrieval_relevance=retrieval_relevance,
            risk_score=risk_score,
            high_risk=high_risk,
            true_label=true_label,
        )
        update_result = None
        if hitl_result.get("triggered") and self.config.enable_hitl:
            update_result = self.hitl_manager.update_models(
                dma=self.dma,
                kb=self.online_kb,
                lora_finetuner=self.lora_finetuner,
                retrieval_updater=self.contrastive_updater,
                privacy_agent=self.privacy_agent if self.config.enable_privacy else None,
                latest_n=8,
            )

        consensus_result = None
        if self.config.enable_consensus and high_risk:
            consensus_result = self.consensus_module.vote(
                dma_result=dma_result,
                tfa_prediction=tfa_prediction,
                verification_summary=verification_summary,
                verification_agent=self.verification_agent,
            )

        final_result = {
            "query": patient_query,
            "true_label": canonical_area(true_label) if true_label else None,
            "perception": {
                "context_norm": float(np.linalg.norm(context_vector)),
                "metadata": perception_state["metadata"],
            },
            "retrieval": {
                "equilibrium_strategy": game_result["equilibrium_strategy"],
                "equilibrium_decision_mode": game_result["equilibrium_decision_mode"],
                "nash_equilibrium_reached": game_result["equilibrium_reached"],
                "nash_rounds": game_result["rounds"],
                "metadata": game_result["retrieval_meta"],
                "extra_meta": {
                    "mixed": mixed_meta,
                    "rerank": rerank_meta,
                    "adaptive": adaptive_meta,
                },
                "top_docs": [doc.as_dict() for doc in verified_docs[:5]],
                "avg_verified_relevance": retrieval_relevance,
            },
            "fusion": fusion_summary,
            "verification": verification_summary,
            "distillation": {
                "selected_doc_ids": distilled_context["selected_doc_ids"],
                "selected_areas": distilled_context["selected_areas"],
                "distilled_text_excerpt": str(distilled_context["distilled_text"])[:220],
            },
            "tfa": tfa_prediction,
            "dma": dma_result,
            "high_risk": high_risk,
            "hitl": hitl_result,
            "parameter_update": update_result,
            "consensus": consensus_result,
            "latency_seconds": time.time() - task_started,
        }
        self.task_history.append(final_result)
        return final_result

    def _evaluate_subset(
        self,
        dataframe: pd.DataFrame,
        enable_hitl: bool,
        enable_tfa: bool,
        enable_multimodal: bool,
        max_samples: int = 40,
        noise_level: float = 0.0,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(self.config.random_seed + int(noise_level * 1000) + max_samples)
        subset = dataframe.head(max_samples).reset_index(drop=True)
        autonomous_predictions: List[str] = []
        final_predictions: List[str] = []
        labels: List[str] = []
        confidences: List[float] = []
        interventions = 0
        for _, row in subset.iterrows():
            query = str(row["question"])
            if noise_level > 0:
                query = mask_query_keywords(query, noise_level, rng)
            if enable_multimodal:
                image_mock = {"precomputed_features": rng.normal(0.0, 1.0, size=self.config.embedding_dim).tolist()}
                perception = self.perception_agent.encode(query, multimodal_input={"image": image_mock})
                context_vector = perception["context_vector"]
            else:
                context_vector = hash_embedding(query, self.config.embedding_dim)
            docs, _ = self.raa.retrieve(query, strategy="adaptive", top_k=self.config.fast_retrieval_top_k)
            verified_docs, _ = self.verification_agent.verify(docs)
            if not verified_docs:
                verified_docs = docs
            tfa_prediction = None
            if enable_tfa:
                tfa_prediction = self.tfa.forecast(
                    query=query,
                    history=rng.normal(0.0, 1.0, size=72).tolist(),
                    authoritative_signal=self._authority_signal_from_docs(verified_docs),
                )
            dma_result = self.dma.infer(
                query=query,
                context_vector=context_vector,
                verified_docs=verified_docs,
                tfa_prediction=tfa_prediction,
                decision_mode="balanced",
            )
            confidence = float(dma_result["confidence"])
            predicted = str(dma_result["prediction"])
            retrieval_relevance = mean_or_zero([doc.similarity for doc in verified_docs])
            corrected = canonical_area(str(row["area"]))
            final_prediction = predicted
            if enable_hitl and (
                confidence < self.config.confidence_threshold
                or retrieval_relevance < self.config.retrieval_relevance_threshold
            ):
                final_prediction = corrected
                interventions += 1
            autonomous_predictions.append(predicted)
            final_predictions.append(final_prediction)
            labels.append(corrected)
            confidences.append(confidence)
        autonomous_accuracy = float(accuracy_score(labels, autonomous_predictions)) if labels else 0.0
        final_accuracy = float(accuracy_score(labels, final_predictions)) if labels else 0.0
        macro_f1 = float(f1_score(labels, final_predictions, average="macro", zero_division=0)) if labels else 0.0
        return {
            "samples": len(labels),
            "autonomous_accuracy": autonomous_accuracy,
            "final_accuracy": final_accuracy,
            "macro_f1": macro_f1,
            "avg_dma_confidence": mean_or_zero(confidences),
            "intervention_rate": interventions / max(len(labels), 1),
            "noise_level": noise_level,
            "enable_hitl": enable_hitl,
            "enable_tfa": enable_tfa,
            "enable_multimodal": enable_multimodal,
        }

    def run_ablation_experiments(self, max_samples: int = 36) -> Dict[str, Any]:
        conditions = {
            "full_ecosystem": (True, True, True),
            "without_hitl": (False, True, True),
            "without_tfa": (True, False, True),
            "without_multimodal": (True, True, False),
        }
        results: Dict[str, Any] = {}
        for name, flags in conditions.items():
            results[name] = self._evaluate_subset(
                self.test_df,
                enable_hitl=flags[0],
                enable_tfa=flags[1],
                enable_multimodal=flags[2],
                max_samples=max_samples,
                noise_level=0.0,
            )
        self.ablation_results = results
        return results

    def run_robustness_experiments(self, max_samples: int = 36) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for noise_level in (0.0, 0.10, 0.30):
            key = f"noise_{int(noise_level * 100)}pct"
            results[key] = self._evaluate_subset(
                self.test_df,
                enable_hitl=False,
                enable_tfa=self.config.enable_tfa,
                enable_multimodal=self.config.enable_multimodal,
                max_samples=max_samples,
                noise_level=noise_level,
            )
        self.robustness_results = results
        return results

    def run_common_rare_analysis(self, max_samples: int = 72) -> Dict[str, Any]:
        train_counts = self.train_df["area"].value_counts()
        sorted_counts = train_counts.sort_values()
        rare_labels = sorted_counts.head(max(2, int(np.ceil(len(sorted_counts) * 0.25)))).index.tolist()
        common_labels = sorted_counts.tail(max(2, int(np.ceil(len(sorted_counts) * 0.25)))).index.tolist()
        subset = self.test_df.head(max_samples).reset_index(drop=True)
        labels: List[str] = []
        preds: List[str] = []
        for _, row in subset.iterrows():
            query = str(row["question"])
            docs, _ = self.raa.retrieve(query, strategy="adaptive", top_k=self.config.fast_retrieval_top_k)
            prediction = self.dma.infer(query, None, docs, tfa_prediction=None, decision_mode="balanced")
            labels.append(canonical_area(str(row["area"])))
            preds.append(str(prediction["prediction"]))
        common_indices = [idx for idx, label in enumerate(labels) if label in common_labels]
        rare_indices = [idx for idx, label in enumerate(labels) if label in rare_labels]
        common_f1 = float(
            f1_score(
                [labels[idx] for idx in common_indices],
                [preds[idx] for idx in common_indices],
                average="macro",
                zero_division=0,
            )
        ) if common_indices else 0.0
        rare_f1 = float(
            f1_score(
                [labels[idx] for idx in rare_indices],
                [preds[idx] for idx in rare_indices],
                average="macro",
                zero_division=0,
            )
        ) if rare_indices else 0.0
        drop_rate = 0.0 if common_f1 <= 1e-12 else max(common_f1 - rare_f1, 0.0) / common_f1
        results = {
            "common_labels": common_labels,
            "rare_labels": rare_labels,
            "common_f1": common_f1,
            "rare_f1": rare_f1,
            "drop_rate": drop_rate,
        }
        self.common_rare_results = results
        return results

    def run_data_scale_adaptability(self, max_test_samples: int = 36) -> Dict[str, Any]:
        rng = np.random.default_rng(self.config.random_seed)
        results: Dict[str, Any] = {}
        reference_test = self.test_df.head(max_test_samples).reset_index(drop=True)
        for fraction in (0.20, 0.50, 0.80, 1.00):
            per_class_chunks: List[pd.DataFrame] = []
            for _, group in self.train_df.groupby("area"):
                count = max(2, int(np.ceil(len(group) * fraction)))
                count = min(count, len(group))
                random_state = int(rng.integers(1, 1_000_000))
                per_class_chunks.append(group.sample(n=count, random_state=random_state, replace=False))
            subset_train = pd.concat(per_class_chunks, ignore_index=True).reset_index(drop=True)
            temp_kb = OnlineKnowledgeBaseManager(self.config)
            temp_kb.build_from_dataframe(subset_train)
            temp_retrieval_updater = ContrastiveRetrievalUpdater(self.config)
            temp_raa = RetrievalAugmentedAgent(self.config, temp_kb, temp_retrieval_updater)
            temp_dma = DecisionMakingAgent(self.config, base_model=self.dma.base_model)
            temp_dma.fit(subset_train)
            labels: List[str] = []
            predictions: List[str] = []
            confidences: List[float] = []
            for _, row in reference_test.iterrows():
                query = str(row["question"])
                docs, _ = temp_raa.retrieve(query, strategy="adaptive", top_k=self.config.fast_retrieval_top_k)
                prediction = temp_dma.infer(query, None, docs, tfa_prediction=None, decision_mode="balanced")
                labels.append(canonical_area(str(row["area"])))
                predictions.append(str(prediction["prediction"]))
                confidences.append(float(prediction["confidence"]))
            accuracy = float(accuracy_score(labels, predictions)) if labels else 0.0
            macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0)) if labels else 0.0
            results[f"train_{int(fraction * 100)}pct"] = {
                "train_samples": len(subset_train),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "avg_confidence": mean_or_zero(confidences),
            }
        self.adaptability_results = results
        return results

    def get_full_report(self) -> Dict[str, Any]:
        maintenance_state = self.maintenance_agent.monitor()
        report = {
            "dataset": {
                "rows": len(self.dataset),
                "train_rows": len(self.train_df),
                "test_rows": len(self.test_df),
                "class_count": len(self.label_set),
                "classes": self.label_set,
            },
            "tasks": {
                "task_count": len(self.task_history),
                "game_runs": len(self.game_history),
                "recent_task_predictions": [
                    {
                        "query": task["query"][:90],
                        "prediction": task["dma"]["prediction"],
                        "confidence": float(task["dma"]["confidence"]),
                        "hitl": bool(task["hitl"]["triggered"]),
                    }
                    for task in self.task_history[-5:]
                ],
            },
            "agents": {
                "Perception": self.perception_agent.get_stats(),
                "DMA": self.dma.get_stats(),
                "RAA": self.raa.get_stats(),
                "TFA": self.tfa.get_stats(),
                "Fusion": self.fusion_agent.get_stats(),
                "Verification": self.verification_agent.get_stats(),
                "Privacy": self.privacy_agent.get_stats(),
                "Maintenance": self.maintenance_agent.get_stats(),
                "Topology": self.topology_manager.get_stats(),
                "Consensus": self.consensus_module.get_stats(),
            },
            "legacy_integrated_modules": {
                "HumanInTheLoopManager": self.hitl_manager.get_stats(),
                "OnlineKnowledgeBaseManager": self.online_kb.get_stats(),
                "TriSourceKnowledgeFusion": self.tri_source_fusion.get_stats(),
                "KnowledgeDistillationEngine": self.distillation_engine.get_stats(),
                "LoRAFineTuner": self.lora_finetuner.get_stats(),
                "ContrastiveRetrievalUpdater": self.contrastive_updater.get_stats(),
            },
            "experiments": {
                "ablation": self.ablation_results,
                "robustness": self.robustness_results,
                "adaptability": self.adaptability_results,
                "common_vs_rare": self.common_rare_results,
            },
            "maintenance_monitor": maintenance_state,
        }
        return report


# ============================================================
# Demonstration and reporting utilities
# ============================================================
def print_task_demo(task_result: Dict[str, Any], index: int) -> None:
    tfa = task_result.get("tfa") or {}
    short_risk = float(tfa.get("short_term", {}).get("risk_probability", 0.0)) if tfa else 0.0
    retrieval = task_result["retrieval"]
    fusion = task_result["fusion"]
    verification = task_result["verification"]
    consensus = task_result.get("consensus")
    print("\n" + "-" * 92)
    print(f"Task {index:02d} | Query: {task_result['query'][:110]}")
    print(
        f"DMA -> prediction={task_result['dma']['prediction']} | "
        f"confidence={task_result['dma']['confidence']:.4f} | "
        f"HITL={task_result['hitl']['triggered']}"
    )
    print(
        f"RAA -> strategy={retrieval['equilibrium_strategy']} | "
        f"Nash={retrieval['nash_equilibrium_reached']} | rounds={retrieval['nash_rounds']} | "
        f"verified relevance={retrieval['avg_verified_relevance']:.4f}"
    )
    print(
        f"TFA -> future {tfa.get('short_term', {}).get('window', '24h')} deterioration risk={short_risk:.2%}"
        if tfa
        else "TFA -> disabled"
    )
    print(
        f"Fusion/Verification -> input={fusion.get('input_docs', 0)} | "
        f"dedup={fusion.get('deduplicated_docs', 0)} | conflicts={fusion.get('conflict_count', 0)} | "
        f"verified={verification.get('accepted_docs', 0)}/{verification.get('accepted_docs', 0) + verification.get('rejected_docs', 0)}"
    )
    if consensus:
        print(
            f"Consensus -> approved={consensus['approved']} | votes={consensus['votes']} | "
            f"required={consensus['required_approvals']}"
        )


def print_experiment_table(title: str, results: Dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)
    for name, metrics in results.items():
        if not isinstance(metrics, dict):
            print(f"{name}: {metrics}")
            continue
        metric_parts: List[str] = []
        for key in ["autonomous_accuracy", "final_accuracy", "accuracy", "macro_f1", "avg_dma_confidence", "avg_confidence", "intervention_rate", "noise_level", "train_samples"]:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    metric_parts.append(f"{key}={value:.4f}")
                else:
                    metric_parts.append(f"{key}={value}")
        print(f"{name}: " + " | ".join(metric_parts))


def print_report_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 92)
    print("FULL ECOSYSTEM REPORT")
    print("=" * 92)
    dataset = report["dataset"]
    print(
        f"Dataset -> rows={dataset['rows']} | train={dataset['train_rows']} | "
        f"test={dataset['test_rows']} | classes={dataset['class_count']}"
    )
    print(f"Task count -> {report['tasks']['task_count']} | Game runs -> {report['tasks']['game_runs']}")
    legacy = report["legacy_integrated_modules"]
    print(
        "HITL -> interventions="
        f"{legacy['HumanInTheLoopManager']['intervention_count']} | "
        f"model_updates={legacy['HumanInTheLoopManager']['model_update_count']}"
    )
    print(
        "KB -> docs="
        f"{legacy['OnlineKnowledgeBaseManager']['document_count']} | "
        f"crawl_updates={legacy['OnlineKnowledgeBaseManager']['crawl_updates']} | "
        f"feedback_updates={legacy['OnlineKnowledgeBaseManager']['feedback_updates']}"
    )
    print(
        "Security -> encrypted_messages="
        f"{report['agents']['Privacy']['encryption_calls']} | "
        f"secure_gradients={report['agents']['Privacy']['gradient_calls']}"
    )
    print(
        "Maintenance -> isolated="
        f"{report['agents']['Maintenance']['isolated_agents']} | restarts={report['agents']['Maintenance']['restart_count']}"
    )


def main() -> None:
    config = EcosystemConfig(
        random_seed=42,
        enable_hitl=True,
        enable_tfa=True,
        enable_multimodal=True,
        topology_type="star",
        consensus_threshold=2.0 / 3.0,
        short_window_hours=24,
        mid_window_days=30,
        long_window_months=12,
    )
    candidate_paths = [Path("medNo.22.csv"), Path("/mnt/data/medNo.22.csv")]
    csv_path = next((path for path in candidate_paths if path.exists()), None)
    dataset = load_medical_dataset(csv_path, target_classes=22, seed=config.random_seed)
    ecosystem = HeterogeneousMultiAgentEcosystem(config=config, dataset=dataset, base_model="Meditron-7B", verbose=True)

    print("=" * 92)
    print("HETEROGENEOUS MULTI-AGENT ECOSYSTEM | MAPFM-INSPIRED MEDICAL AI DEMO")
    print("=" * 92)
    print(
        f"Loaded dataset: {len(dataset)} samples | "
        f"{dataset['area'].nunique()} medical areas | train/test={len(ecosystem.train_df)}/{len(ecosystem.test_df)}"
    )

    added_docs = ecosystem.refresh_online_knowledge(n_per_source=1)
    print(f"Dynamic KB update: added {len(added_docs)} authoritative documents from {', '.join(AUTHORITATIVE_SOURCES)}")

    rng = np.random.default_rng(config.random_seed)
    demo_subset = ecosystem.test_df.head(20).reset_index(drop=True)
    for index, row in demo_subset.iterrows():
        image_features = rng.normal(0.0, 1.0, size=config.embedding_dim).tolist()
        time_series = rng.normal(0.0, 1.0, size=72).cumsum().tolist()
        force_high_risk = bool(index in {2, 9, 16})
        result = ecosystem.run_collaborative_task(
            patient_query=str(row["question"]),
            multimodal_input={"image": {"precomputed_features": image_features}, "time_series": time_series},
            true_label=str(row["area"]),
            force_high_risk=force_high_risk,
        )
        print_task_demo(result, index=index + 1)

    ablation = ecosystem.run_ablation_experiments(max_samples=12)
    robustness = ecosystem.run_robustness_experiments(max_samples=12)
    adaptability = ecosystem.run_data_scale_adaptability(max_test_samples=12)
    common_rare = ecosystem.run_common_rare_analysis(max_samples=18)

    print_experiment_table("ABLATION EXPERIMENTS", ablation)
    print_experiment_table("NOISE ROBUSTNESS EXPERIMENTS", robustness)
    print_experiment_table("DATA-SCALE ADAPTABILITY EXPERIMENTS", adaptability)
    print("\n" + "=" * 92)
    print("COMMON VS. RARE DISEASE ANALYSIS")
    print("=" * 92)
    print(
        f"Common F1={common_rare['common_f1']:.4f} | Rare F1={common_rare['rare_f1']:.4f} | "
        f"Drop Rate={common_rare['drop_rate']:.2%}"
    )
    print(f"Common labels: {common_rare['common_labels']}")
    print(f"Rare labels: {common_rare['rare_labels']}")

    report = ecosystem.get_full_report()
    print_report_summary(report)


if __name__ == "__main__":
    main()
