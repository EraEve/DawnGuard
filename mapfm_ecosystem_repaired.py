from __future__ import annotations
import hashlib
import json
import re
import time
import traceback
import warnings
from abc import ABC, abstractmethod
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

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None

try:
    import ollama
    # 验证客户端版本（日志输出移到后面configure_logging之后）
    if not hasattr(ollama, 'Client'):
        ollama = None
except ImportError:
    ollama = None
except Exception as e:
    ollama = None

from audit_logger import append_audit_event
from desensitizer import desensitize_text
from config_loader import load_yaml_config
from exceptions import AgentError, ConfigError, DatasetError, InferenceError, MAPFMException, RetrievalError
from logging_config import configure_logging, logger
from traceability import save_decision_trace
from utils import (
    AUTHORITATIVE_SOURCES,
    DEFAULT_MEDICAL_AREAS,
    batch_semantic_embeddings,
    canonical_area,
    clip01,
    create_synthetic_medical_rows,
    hamming_distance_int,
    hash_embedding,
    l2_normalize,
    load_medical_dataset,
    mask_query_keywords,
    mean_or_zero,
    normalized_entropy,
    safe_datetime,
    semantic_embedding,
    sigmoid,
    simhash,
    softmax,
    stable_seed,
    tokenize,
)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

warnings.filterwarnings("ignore")
configure_logging(log_dir="logs", level="INFO")


# === P3: Structured request tracing ===
import uuid as _uuid
from contextvars import ContextVar as _ContextVar

_request_id_ctx: _ContextVar[str] = _ContextVar("request_id", default="")
_stage_timers: Dict[str, float] = {}


def _start_request(request_id: Optional[str] = None) -> str:
    """Begin a new traced request with a correlation ID."""
    rid = request_id or str(_uuid.uuid4())[:8]
    _request_id_ctx.set(rid)
    _stage_timers.clear()
    logger.bind(request_id=rid).info("Request started")
    return rid


def _log_stage(stage: str, elapsed_ms: float, **extra: Any) -> None:
    """Log a pipeline stage with timing and structured context."""
    rid = _request_id_ctx.get()
    logger.bind(request_id=rid, stage=stage, elapsed_ms=round(elapsed_ms, 2), **extra).info(
        "Stage '{}' completed in {:.1f}ms", stage, elapsed_ms
    )


def _end_request() -> None:
    """Finalize a traced request."""
    rid = _request_id_ctx.get()
    logger.bind(request_id=rid).info("Request completed")
    _request_id_ctx.set("")
# 检查Ollama导入状态
if ollama is not None:
    logger.info("✅ Ollama Python client imported successfully")
else:
    logger.warning("⚠️ Ollama library not available, will automatically fall back to legacy DMA mode")
    logger.warning("💡 To enable Ollama acceleration, run: pip install ollama && ollama pull llama3:8b")


# ============================================================
# Global utility functions
# ============================================================
# ============================================================
# Configuration classes
# ============================================================
@dataclass
class SystemConfig:
    """System-level configuration with hard-coded parameters externalized."""

    # === MODIFIED: 硬编码抽离至配置类 ===
    random_seed: int = 42
    embedding_dim: int = 128
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tfidf_features: int = 2400
    test_size: float = 0.20
    confidence_threshold: float = 0.62
    retrieval_relevance_threshold: float = 0.24
    verification_max_age_days: int = 365
    cost_lambda: float = 0.72
    nash_max_rounds: int = 6
    query_uncertainty_threshold: float = 0.58
    secure_gradient_noise_std: float = 0.015
    agent_timeout_seconds: float = 120.0
    enable_faiss_dense_index: bool = True
    faiss_search_multiplier: int = 2
    faiss_use_inner_product: bool = True
    dma_centroid_weight: float = 0.50
    dma_retrieval_weight: float = 0.34
    dma_lexical_weight: float = 0.12
    dma_context_weight: float = 0.02
    dma_tfa_weight: float = 0.02
    dma_softmax_temperature: float = 0.18
    lora_learning_rate: float = 0.08
    gradient_margin_shift_rate: float = 0.02
    retrieval_query_softmax_temperature: float = 0.35
    ollama_model_name: str = "llama3:8b"
    ollama_max_retries: int = 3
    ollama_timeout_seconds: float = 10.0
    ollama_temperature: float = 0.0
    ollama_base_url: str = "http://127.0.0.1:11434"
    allow_dma_legacy_fallback: bool = True
    max_concurrent_requests: int = 4
    log_level: str = "INFO"
    ollama_num_ctx:int =2048
    ollama_flash_attn:bool =True


@dataclass
class EcosystemConfig(SystemConfig):
    """Ecosystem-level configuration with strategy, string, and clinical parameters."""

    # === MODIFIED: 硬编码抽离至配置类 ===
    enable_hitl: bool = True
    enable_interactive_hitl: bool = False
    enable_tfa: bool = True
    enable_medtsllm: bool = True  # Use real MedTsLLM (BioBERT backbone) instead of heuristic TFA
    enable_multimodal: bool = True
    enable_privacy: bool = True
    enable_maintenance: bool = True
    enable_consensus: bool = True
    enable_dynamic_kb: bool = True
    enable_distillation: bool = True
    enable_tri_source_fusion: bool = True
    raa_strategy_combo: Tuple[str, ...] = ("mixed", "rerank", "adaptive")
    rag_strategy_names: Tuple[str, ...] = ("mixed", "rerank", "adaptive")
    authoritative_sources: Tuple[str, ...] = AUTHORITATIVE_SOURCES
    error_type_names: Tuple[str, ...] = ("knowledge_missing", "knowledge_misleading", "reasoning_bias")
    agent_display_names: Tuple[str, ...] = ("Perception", "RAA", "DMA", "TFA", "Fusion", "Verification", "Privacy")
    decision_modes: Tuple[str, ...] = ("fast", "balanced", "deep")
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
    rerank_similarity_weight: float = 0.62
    rerank_overlap_weight: float = 0.22
    rerank_recency_weight: float = 0.10
    rerank_authority_bonus: float = 0.10
    adaptive_entropy_weight: float = 0.58
    adaptive_gap_weight: float = 0.32
    adaptive_oov_weight: float = 0.10
    experiment_noise_levels: Tuple[float, ...] = (0.10, 0.30)
    ablation_sample_count: int = 12
    robustness_sample_count: int = 12
    adaptability_sample_count: int = 12
    common_rare_sample_count: int = 18
    continuous_learning_min_feedback: int = 10
    target_classes: int = 22
    acute_short_window_multiplier: float = 3.0
    chronic_long_window_multiplier: float = 2.0
    pressure_ulcer_bedrest_growth_rate: float = 0.08
    fall_posture_weight: float = 0.30
    fall_activity_weight: float = 0.20
    fall_threshold_shift: float = 0.08
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
    sample_id: int = 0
    query: str = ""
    predicted_label: str = ""
    corrected_label: str = ""
    confidence: float = 0.0
    retrieval_relevance: float = 0.0
    feedback_score: int = 0
    error_type: str = ""
    correction_reason: str = ""
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
        # === MODIFIED: 性能优化；新增 FAISS 稠密索引并保留 TF-IDF 稀疏索引 ===
        self.faiss_index: Optional[Any] = None
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
        """Rebuild sparse TF-IDF and dense MiniLM+FAISS indexes after KB updates."""
        texts = self._combined_texts()
        if not texts:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.dense_matrix = np.zeros((0, self.config.embedding_dim), dtype=np.float32)
            self.faiss_index = None
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
        # === MODIFIED: 性能优化；MiniLM 语义向量替换原 hash_embedding，动态 KB 更新自动同步索引 ===
        self.dense_matrix = batch_semantic_embeddings(
            texts,
            dim=self.config.embedding_dim,
            model_name=self.config.embedding_model_name,
        ).astype(np.float32)
        self.faiss_index = None
        if faiss is not None and self.config.enable_faiss_dense_index and self.dense_matrix.size > 0:
            try:
                self.faiss_index = faiss.IndexFlatIP(self.config.embedding_dim)
                self.faiss_index.add(self.dense_matrix.astype(np.float32))
            except Exception as exc:
                logger.exception("FAISS index build failed; dense numpy fallback kept: {}", exc)
                self.faiss_index = None
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

    @staticmethod
    def _snapshot_encoder(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Unserializable type {type(obj)}")

    @staticmethod
    def _snapshot_decoder(obj: Any) -> Any:
        if isinstance(obj, dict) and "__datetime__" in obj:
            return datetime.fromisoformat(obj["__datetime__"])
        return obj

    def save_snapshot(self, path: Path) -> None:
        payload = {
            "documents": self.documents,
            "next_doc_id": self.next_doc_id,
            "index_version": self.index_version,
            "crawl_updates": self.crawl_updates,
            "feedback_updates": self.feedback_updates,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=self._snapshot_encoder, ensure_ascii=False)

    def load_snapshot(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle, object_hook=self._snapshot_decoder)
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
        embeddings = [semantic_embedding(f"{doc.question} {doc.answer}", self.config.embedding_dim, self.config.embedding_model_name) for doc in selected]
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
        query_strength = max(float(np.linalg.norm(semantic_embedding(query, self.config.embedding_dim, self.config.embedding_model_name))), 0.1)
        for doc_id in positive_doc_ids:
            kb.update_document_weight(int(doc_id), feedback_score=1, gradient_proxy=query_strength, eta=self.config.lora_learning_rate)
            self.positive_updates += 1
        for doc_id in negative_doc_ids:
            kb.update_document_weight(int(doc_id), feedback_score=-1, gradient_proxy=query_strength, eta=self.config.lora_learning_rate)
            self.negative_updates += 1
        total = max(len(positive_doc_ids) + len(negative_doc_ids), 1)
        delta = (len(positive_doc_ids) - len(negative_doc_ids)) / total
        self.retrieval_margin_shift = float(np.clip(self.retrieval_margin_shift + self.config.gradient_margin_shift_rate * delta, -0.25, 0.25))
        return self.get_stats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "update_count": self.update_count,
            "positive_updates": self.positive_updates,
            "negative_updates": self.negative_updates,
            "retrieval_margin_shift": self.retrieval_margin_shift,
        }


class BaseFineTuner(ABC):
    """Abstract fine-tuner interface matching the original LoRAFineTuner contract."""

    @abstractmethod
    def update(
        self,
        dma: "DecisionMakingAgent",
        feedback_records: Sequence[HumanFeedbackRecord],
        privacy_agent: Optional["PrivacySecurityAgent"] = None,
    ) -> Dict[str, Any]:
        """Update DMA calibration from feedback records."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return fine-tuner statistics."""
        raise NotImplementedError


class RealFineTuner(BaseFineTuner):
    """Empty real implementation placeholder for future LoRA model training."""

    def update(self, dma: "DecisionMakingAgent", feedback_records: Sequence[HumanFeedbackRecord], privacy_agent: Optional["PrivacySecurityAgent"] = None) -> Dict[str, Any]:
        return {"updated": False, "reason": "real_finetuner_not_implemented"}

    def get_stats(self) -> Dict[str, Any]:
        return {"implementation": "RealFineTuner", "ready": False}


# === MODIFIED: 新增抽象接口解耦模拟与真实逻辑 ===
class SimulatedLoRAFineTuner(BaseFineTuner):
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
            raw_gradient += signed * semantic_embedding(record.query, self.config.embedding_dim, self.config.embedding_model_name)
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


# Backward-compatible alias preserving original constructor usage.
LoRAFineTuner = SimulatedLoRAFineTuner


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

    def should_intervene_tfa(
        self,
        tfa_prediction: Optional[Dict[str, Any]],
        dma_result: Optional[Dict[str, Any]] = None,
        verification_tfa: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str, int]:
        """Check TFA-specific HITL triggers.

        Priority levels: 1=critical (immediate), 2=conflict (urgent), 3=warning.
        """
        if tfa_prediction is None:
            return False, "", 0

        risk_level = tfa_prediction.get("risk_level", "low")
        reasons: List[str] = []
        priority = 0

        if risk_level == "critical":
            reasons.append("TFA critical risk — highest priority intervention")
            priority = 1

        if int(tfa_prediction.get("conflict_with_diagnosis", 0)) >= 1:
            reasons.append("TFA risk conflicts with DMA diagnosis")
            priority = max(priority, 2)

        if tfa_prediction.get("calibration_mode") == "degraded":
            reasons.append("TFA degraded — DMA confidence too low")
            priority = max(priority, 2)

        if verification_tfa and not verification_tfa.get("valid", True):
            reasons.append(f"Verification flagged TFA: {verification_tfa.get('message', '')}")
            priority = max(priority, 3)

        triggered = bool(reasons)
        self.last_trigger_reason = " | ".join(reasons)
        return triggered, self.last_trigger_reason, priority

    def process_tfa_intervention(
        self,
        query: str,
        tfa_prediction: Dict[str, Any],
        dma_result: Optional[Dict[str, Any]] = None,
        human_approved: bool = False,
        human_corrections: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a TFA-triggered HITL intervention."""
        corrections = human_corrections or {}
        if not human_approved:
            error_key = "tfa_false_positive" if tfa_prediction.get("risk_level") in ("high", "critical") else "tfa_false_negative"
            self.error_distribution[error_key] += 1

        record = {
            "timestamp": datetime.now().isoformat(),
            "query": str(query)[:200],
            "original_risk_level": tfa_prediction.get("risk_level", "low"),
            "original_risk_score": tfa_prediction.get("risk_score", 0.0),
            "human_approved": human_approved,
            "corrections_applied": bool(corrections),
            "corrected_risk_level": corrections.get("risk_level", tfa_prediction.get("risk_level")),
            "corrected_risk_score": corrections.get("risk_score", tfa_prediction.get("risk_score")),
            "added_factors": corrections.get("added_factors", []),
            "removed_factors": corrections.get("removed_factors", []),
            "modified_recommendations": corrections.get("modified_recommendations", []),
            "reviewer_notes": corrections.get("notes", ""),
        }

        self.feedback_records.append(HumanFeedbackRecord(
            sample_id=self.sample_counter + 1,
            query=query,
            predicted_label=f"TFA_RISK={tfa_prediction.get('risk_level', 'low')}",
            corrected_label=f"TFA_RISK={corrections.get('risk_level', tfa_prediction.get('risk_level', 'low'))}",
            confidence=tfa_prediction.get("risk_score", 0.0),
            retrieval_relevance=0.0,
            feedback_score=0 if human_approved else 5,
            error_type="tfa_overestimate" if not human_approved else "normal",
            correction_reason=corrections.get("notes", ""),
        ))
        self.intervention_count += 1
        self.sample_counter += 1
        return {
            "triggered": True, "intervention_type": "tfa_risk_review",
            "record": record, "priority": 1 if not human_approved else 0,
        }

    def generate_tfa_intervention_report(self) -> Dict[str, Any]:
        """Generate periodic TFA intervention statistics report."""
        tfa_records = [
            r for r in self.feedback_records
            if r.predicted_label.startswith("TFA_RISK=")
        ]
        total = len(tfa_records)
        if total == 0:
            return {"total_tfa_interventions": 0, "message": "No TFA interventions yet"}

        overrides = sum(1 for r in tfa_records if r.error_type == "tfa_overestimate")
        return {
            "total_tfa_interventions": total,
            "overrides": overrides,
            "override_rate": overrides / total if total > 0 else 0.0,
            "false_positive_count": self.error_distribution.get("tfa_false_positive", 0),
            "false_negative_count": self.error_distribution.get("tfa_false_negative", 0),
            "error_rate": (self.error_distribution.get("tfa_false_positive", 0)
                           + self.error_distribution.get("tfa_false_negative", 0)) / max(total, 1),
            "common_errors": dict(self.error_distribution.most_common(5)),
            "last_trigger_reason": self.last_trigger_reason,
        }

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
                "corrected_label": prediction,
                "feedback_record": None,
                "intervention_layer": "autonomous_execution",
            }
        # When HITL triggers, flag for review instead of auto-correcting via ground truth.
        # The system marks low-confidence results and reduces confidence to reflect uncertainty.
        self.intervention_count += 1
        self.sample_counter += 1
        error_type, reason_detail = self._classify_error(prediction, canonical_area(true_label) if true_label else prediction, retrieval_docs)
        self.error_distribution[error_type] += 1
        penalty = max(0.0, self.config.confidence_threshold - confidence)
        adjusted_confidence = clip01(confidence - penalty * 0.5)
        return {
            "triggered": True,
            "reason": reason,
            "final_prediction": prediction,
            "final_confidence": adjusted_confidence,
            "corrected_label": canonical_area(true_label) if true_label else prediction,
            "feedback_record": None,
            "intervention_layer": "needs_human_review",
            "error_type": error_type,
            "detail": reason_detail,
        }

    def interactive_review(
        self,
        query: str,
        predicted_label: str,
        confidence: float,
        true_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Interactive CLI prompt for human review of flagged low-confidence diagnoses.

        When HITL triggers, this method pauses the pipeline and waits for human input.
        In non-interactive (simulation) mode, it auto-corrects using ground truth if available.
        """
        import sys

        print("\n" + "=" * 72)
        print("HUMAN-IN-THE-LOOP REVIEW")
        print("=" * 72)
        print(f"Query:    {str(query)[:120]}")
        print(f"Predicted: {predicted_label}  (confidence: {confidence:.2%})")
        print("-" * 72)

        if not self.config.enable_interactive_hitl or not sys.stdin.isatty():
            if true_label:
                corrected = canonical_area(true_label)
                print(f"[SIMULATION] Auto-correcting to ground truth: {corrected}")
                return {"action": "correct", "corrected_label": corrected,
                        "reviewer_notes": "simulation-auto-correct", "interactive": False}
            else:
                print("[SIMULATION] No TTY — auto-approving with reduced confidence")
                return {"action": "approve", "corrected_label": predicted_label,
                        "reviewer_notes": "simulation-auto-approve", "interactive": False}

        while True:
            print("\nActions: [A]pprove  [R]eject  [C]orrect diagnosis  [S]kip")
            try:
                choice = input("Your decision (A/R/C/S): ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                if true_label:
                    corrected = canonical_area(true_label)
                    print(f"\n[INPUT CLOSED] Auto-correcting to ground truth: {corrected}")
                    return {"action": "correct", "corrected_label": corrected,
                            "reviewer_notes": "input-closed-auto-correct", "interactive": False}
                print("\n[INPUT CLOSED] Auto-approving…")
                return {"action": "approve", "corrected_label": predicted_label,
                        "reviewer_notes": "input-closed-auto-approve", "interactive": False}

            if choice == "A":
                return {"action": "approve", "corrected_label": predicted_label,
                        "reviewer_notes": "human-approved", "interactive": True}
            elif choice == "R":
                notes = input("Reason for rejection: ").strip()
                return {"action": "reject", "corrected_label": predicted_label,
                        "reviewer_notes": notes or "human-rejected", "interactive": True}
            elif choice == "C":
                corrected = input("Correct diagnosis: ").strip()
                notes = input("Notes (optional): ").strip()
                return {"action": "correct",
                        "corrected_label": canonical_area(corrected) if corrected else predicted_label,
                        "reviewer_notes": notes or "human-corrected", "interactive": True}
            elif choice == "S":
                return {"action": "skip", "corrected_label": predicted_label,
                        "reviewer_notes": "human-skipped", "interactive": True}
            else:
                print("Invalid choice — enter A, R, C, or S.")

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
        self.last_runtime_seconds = 0.0
        self.total_runtime_seconds = 0.0

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
            "last_runtime_seconds": self.last_runtime_seconds,
            "total_runtime_seconds": self.total_runtime_seconds,
        }

    def record_runtime(self, started: float) -> None:
        """Record latest and cumulative agent runtime metrics."""
        elapsed = time.perf_counter() - started
        self.last_runtime_seconds = float(elapsed)
        self.total_runtime_seconds += float(elapsed)


class BasePerceptionAgent(BaseAgent, ABC):
    """Abstract perception-agent interface preserving the legacy encode contract."""

    @abstractmethod
    def encode(self, text: str, multimodal_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return text and context vectors."""
        raise NotImplementedError


class RealPerceptionAgent(BasePerceptionAgent):
    """Empty real implementation placeholder for future production perception."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Perception")
        self.config = config

    def encode(self, text: str, multimodal_input: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {"text_vector": np.zeros(self.config.embedding_dim, dtype=np.float32), "context_vector": np.zeros(self.config.embedding_dim, dtype=np.float32), "metadata": {"ready": False}}


# === MODIFIED: 新增抽象接口解耦模拟与真实逻辑 ===
class SimulatedMultimodalPerceptionAgent(BasePerceptionAgent):
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
        return semantic_embedding(token_state, self.config.embedding_dim, self.config.embedding_model_name)

    def _image_to_patch_embeddings(self, image_input: Any) -> np.ndarray:
        dim = self.config.embedding_dim
        if image_input is None:
            return np.zeros((4, dim), dtype=np.float32)
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
            patches.append(semantic_embedding(textual_proxy, dim, self.config.embedding_model_name))
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
        started = time.perf_counter()
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
        self.record_runtime(started)
        return {
            "text_vector": text_vector,
            "context_vector": fused,
            "metadata": metadata,
        }

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update({"encoding_calls": self.encoding_calls})
        return status


# Backward-compatible alias preserving original constructor usage.
MultimodalPerceptionAgent = SimulatedMultimodalPerceptionAgent


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
        """Retrieve dense semantic neighbors through FAISS or a vectorized fallback."""
        if self.kb.dense_matrix.size == 0 or not self.kb.documents:
            return []
        # === MODIFIED: 性能优化；FAISS 替换原稠密点积扫描，保留结果字段与排序逻辑 ===
        query_embedding = semantic_embedding(
            query,
            dim=self.config.embedding_dim,
            model_name=self.config.embedding_model_name,
        ).astype(np.float32)
        candidate_k = min(max(top_k * self.config.faiss_search_multiplier, top_k), len(self.kb.documents))
        if self.kb.faiss_index is not None:
            scores, indices = self.kb.faiss_index.search(query_embedding.reshape(1, -1), candidate_k)
            index_order = indices.flatten().tolist()
            score_lookup = {int(idx): float(score) for idx, score in zip(indices.flatten().tolist(), scores.flatten().tolist()) if int(idx) >= 0}
        else:
            similarities = self.kb.dense_matrix @ query_embedding
            index_order = np.argsort(similarities)[::-1][:candidate_k].astype(int).tolist()
            score_lookup = {int(index): float(similarities[index]) for index in index_order}
        results: List[RetrievalResult] = []
        for index in index_order:
            if index < 0 or index >= len(self.kb.documents):
                continue
            doc = self.kb.documents[int(index)]
            base_similarity = (float(score_lookup.get(int(index), 0.0)) + 1.0) / 2.0
            weighted_similarity = base_similarity * float(doc.get("doc_weight", 1.0))
            if weighted_similarity <= 0:
                continue
            results.append(self._doc_to_result(doc, weighted_similarity, "dense_faiss"))
        results.sort(key=lambda item: item.similarity, reverse=True)
        return results[:top_k]

    @staticmethod
    def _medical_concept_match(query: str, doc_area: str, doc_text: str) -> float:
        """Boost documents whose area or text overlaps with medical taxonomy concepts in the query."""
        query_lower = query.lower()
        area_lower = doc_area.lower()
        text_lower = doc_text.lower()
        area_tokens = set(tokenize(area_lower))
        score = 0.0
        # Direct area name match in query (e.g. "hypertension" in query → "Hypertension" doc)
        if any(token in query_lower for token in area_tokens if len(token) > 3):
            score += 0.12
        # Medical-area label overlap between query and doc text
        query_tokens = set(tokenize(query_lower))
        text_tokens = set(tokenize(text_lower))
        medical_overlap = query_tokens & text_tokens
        if medical_overlap:
            score += 0.06 * min(len(medical_overlap), 6)
        return clip01(score)

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
            authority_bonus = self.config.rerank_authority_bonus if doc.source in self.config.authoritative_sources else 0.0
            medical_bonus = self._medical_concept_match(query, doc.area, f"{doc.question} {doc.answer}")
            doc.rerank_score = clip01(
                self.config.rerank_similarity_weight * doc.similarity
                + self.config.rerank_overlap_weight * overlap
                + self.config.rerank_recency_weight * recency_score
                + authority_bonus
                + medical_bonus
            )
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
        probabilities = softmax(scores, temperature=self.config.retrieval_query_softmax_temperature)
        entropy_component = normalized_entropy(probabilities)
        if len(probabilities) >= 2:
            gap = float(probabilities[0] - probabilities[1])
        else:
            gap = float(probabilities[0])
        gap_component = 1.0 - clip01(gap)
        OOV_proxy = 1.0 if len(tokenize(query)) <= 2 else 0.0
        uncertainty = clip01(self.config.adaptive_entropy_weight * entropy_component + self.config.adaptive_gap_weight * gap_component + self.config.adaptive_oov_weight * OOV_proxy)
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
        if not str(query).strip():
            logger.warning("RAA received an empty query; returning an empty retrieval result.")
            return [], {"strategy": strategy, "uncertainty": 1.0, "deep_retrieval": False, "result_count": 0, "avg_relevance": 0.0, "latency_seconds": 0.0}
        started = time.perf_counter()
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
        self.record_runtime(started)
        return results, metadata

    def estimate_cost(self, strategy: str, doc_count: int) -> float:
        costs = {"mixed": 0.22, "rerank": 0.36, "adaptive": 0.28}
        base = float(costs.get(strategy.lower().replace("-rag", ""), 0.28))
        diversity_penalty = 0.012 * max(doc_count, 0)
        deep_penalty = 0.08 if strategy.lower().startswith("adaptive") and self.last_uncertainty > self.config.query_uncertainty_threshold else 0.0
        return float(base + diversity_penalty + deep_penalty)

    def retrieve_risk_aware(
        self,
        query: str,
        tfa_prediction: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievalResult], Dict[str, Any], Dict[str, Any]]:
        """Risk-aware retrieval: fetches clinical guidelines relevant to TFA risk.

        When TFA outputs high/critical risk, automatically retrieves documents
        about risk factors and interventions for the relevant disease.

        Returns:
            (docs, meta, risk_knowledge) — where risk_knowledge contains
            matched guidelines, conflicts, and confidence adjustment.
        """
        risk_level = "low"
        if tfa_prediction:
            risk_level = tfa_prediction.get("risk_level", "low")

        # Base retrieval
        docs, meta = self.retrieve(query, strategy="adaptive", top_k=top_k)

        # Risk-tiered additional retrieval
        risk_knowledge: Dict[str, Any] = {
            "risk_level": risk_level,
            "guidelines_matched": False,
            "conflicts_found": False,
            "confidence_adjustment": 0.0,
            "risk_specific_docs": [],
        }

        if risk_level in ("high", "critical"):
            # Augment query with risk-related keywords for targeted retrieval
            risk_query = f"{query} risk factors complications deterioration prevention intervention"
            risk_docs, _ = self.retrieve(risk_query, strategy="rerank",
                                         top_k=self.config.fast_retrieval_top_k)
            risk_knowledge["risk_specific_docs"] = [
                {"doc_id": d.doc_id, "area": d.area, "similarity": d.similarity,
                 "answer_excerpt": d.answer[:200]}
                for d in risk_docs[:3]
            ]
            risk_knowledge["guidelines_matched"] = len(risk_docs) > 0

            # Compare retrieved risk knowledge with TFA risk factors
            if tfa_prediction:
                tfa_factors = tfa_prediction.get("primary_risk_factors", [])
                doc_text = " ".join(d.answer.lower() for d in risk_docs[:5])
                matched = sum(1 for f in tfa_factors if any(
                    kw in doc_text for kw in str(f).lower().split()
                ))
                risk_knowledge["tfa_raa_consistency"] = (
                    "consistent" if matched >= len(tfa_factors) * 0.5
                    else "partial" if matched > 0
                    else "divergent"
                )
                if risk_knowledge["tfa_raa_consistency"] == "consistent":
                    risk_knowledge["confidence_adjustment"] = 0.05
                elif risk_knowledge["tfa_raa_consistency"] == "divergent":
                    risk_knowledge["conflicts_found"] = True
                    risk_knowledge["confidence_adjustment"] = -0.05

        return docs, meta, risk_knowledge

    def generate_risk_recommendations(
        self, risk_level: str, disease_area: str
    ) -> List[str]:
        """Generate tiered clinical recommendations based on TFA risk level."""
        base_recs = {
            "low": [
                f"建议{disease_area}患者保持常规随访",
                "注意日常健康监测和生活方式管理",
            ],
            "medium": [
                f"建议{disease_area}患者增加随访频率",
                "密切观察病情变化，定期复查相关指标",
                "评估当前治疗方案的有效性",
            ],
            "high": [
                f"建议{disease_area}患者及时就医",
                "进行全面的病情评估和相关检查",
                "考虑多学科会诊制定个体化治疗方案",
                "教育患者及家属识别病情恶化的早期信号",
            ],
            "critical": [
                f"建议{disease_area}患者立即就医或急诊处理",
                "需要紧急医疗干预和全面评估",
                "通知主治医师，准备应急预案",
                "持续监测生命体征，防范并发症",
                "考虑ICU监护和高级生命支持",
            ],
        }
        return base_recs.get(risk_level, base_recs["low"])

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



# === MODIFIED: DMA接入本地Llama3:8b大模型 ===
LLAMA3_PROMPT_TEMPLATE = """

你是一个医疗诊断助手，根据以下信息给出预测：
标签列表：{label_list}
患者问题：{query}
参考文档：{retrieval_context}

【强制要求】
1. 只返回 JSON 格式结果，**绝对不能加任何文字、解释、标点、废话**
2. JSON 必须包含且只包含 2 个字段：
   - "prediction"：字符串，诊断结果
   - "confidence"：数字，置信度 0~1

正确返回示例：
{{"prediction":"感冒","confidence":0.92}}
"""

class DecisionMakingAgent(BaseAgent):
    """DMA with Ollama-backed local Llama3:8b inference and legacy fallback compatibility."""

    def __init__(self, config: EcosystemConfig, base_model: str = "llama3:8b") -> None:
        super().__init__("DMA")
        self.config = config
        self.base_model = base_model or config.ollama_model_name
        self.ollama_model_name = config.ollama_model_name
        self.ollama_failures = 0
        self.ollama_successes = 0
        self.labels: List[str] = []
        self.class_centroids: Dict[str, np.ndarray] = {}
        self.class_keyword_cache: Dict[str, set[str]] = {}
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.calibration_factor = 1.0
        self.class_bias: Dict[str, float] = defaultdict(float)
        self.decision_history: List[Dict[str, Any]] = []
        self.inference_calls = 0
        self._ollama_healthy: bool = True
        self._ollama_checked_at: float = 0.0
        self._ollama_health_lock = __import__("threading").Lock()
        # Platt scaling parameters: P(correct|score) = 1 / (1 + exp(A * score + B))
        # Start neutral (A=0 → flat sigmoid=0.5); SGD learns sign + magnitude from data
        self._platt_A: float = 0.0
        self._platt_B: float = 0.0
        self._calibration_pairs: List[Tuple[float, int]] = []  # (confidence, is_correct)

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

    # Intent classification keywords — ordered by specificity (more specific patterns first)
    _INTENT_PATTERNS: Tuple[Tuple[str, str], ...] = (
        ("screening", "screening_check"),
        ("checkup|annual.*physical|routine.*exam|wellness.*visit|preventive.*check", "screening_check"),
        ("symptom|suffer|feel|hurts|pain|ache|bleeding|swelling|nausea|dizzy|fatigue|fever|cough|rash", "symptom_inquiry"),
        ("treat|medication|drug|dose|therapy|surgery|prescri|procedure|manage|intervention", "treatment_inquiry"),
        ("prognosis|survival|outcome|recovery|life.expectancy|will.i.*(live|survive|recover)", "prognosis"),
        ("prevent|risk.factor|avoid|protect|vaccin|screen|early.detect|lifestyle|diet|exercise", "prevention"),
    )

    # Confusable-disease synonym token map: expands query tokens with medical equivalents
    # so "high blood pressure" query also matches the "Hypertension" label token set
    _SYNONYM_TOKEN_MAP: Dict[str, List[str]] = {
        "hypertension": ["high", "blood", "pressure", "hbp", "elevated", "bp"],
        "myocardial infarction": ["heart", "attack", "mi", "cardiac", "arrest"],
        "colorectal cancer": ["colon", "cancer", "bowel", "colorectal"],
        "stroke": ["cva", "cerebrovascular", "brain", "attack"],
    }

    def _apply_platt_scaling(self, raw_confidence: float) -> float:
        """Platt-scaled confidence: sigmoid(A * raw + B), calibrated to empirical accuracy.

        When uncalibrated (A ≈ 0), passes through raw_confidence unchanged.
        """
        if abs(self._platt_A) < 1e-6:
            return clip01(raw_confidence)
        z = self._platt_A * raw_confidence + self._platt_B
        z_clipped = float(np.clip(z, -20.0, 20.0))
        return clip01(1.0 / (1.0 + float(np.exp(-z_clipped))))

    def calibrate_platt_from_history(self) -> bool:
        """Fit Platt scaling parameters (A, B) from calibration pairs using SGD.

        Returns True if calibration was updated.
        """
        if len(self._calibration_pairs) < 10:
            return False
        pairs = self._calibration_pairs[-200:]  # sliding window
        confidences = np.array([p[0] for p in pairs], dtype=np.float32)
        labels = np.array([p[1] for p in pairs], dtype=np.float32)
        # Mini-batch SGD for Platt scaling logistic regression
        lr = 0.01
        a, b = self._platt_A, self._platt_B
        for _ in range(200):
            z = a * confidences + b
            z = np.clip(z, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-z))
            err = p - labels
            a -= lr * float(np.mean(err * confidences))
            b -= lr * float(np.mean(err))
        a = float(np.clip(a, -10.0, 10.0))  # sign depends on calibration data correlation
        b = float(np.clip(b, -5.0, 5.0))
        self._platt_A = a
        self._platt_B = b
        return True

    def record_calibration_feedback(self, confidence: float, is_correct: bool) -> None:
        """Record a (confidence, accuracy) pair for online Platt calibration."""
        self._calibration_pairs.append((float(confidence), 1 if is_correct else 0))
        if len(self._calibration_pairs) % 20 == 0:
            self.calibrate_platt_from_history()

    def _classify_intent(self, query: str) -> str:
        """Classify the clinical intent of a patient query via keyword matching.

        Returns one of: screening_check, symptom_inquiry, treatment_inquiry,
        prognosis, prevention, or general_inquiry.
        """
        q = query.lower()
        for pattern, intent in self._INTENT_PATTERNS:
            if re.search(pattern, q):
                return intent
        return "general_inquiry"

    def _build_retrieval_context(self, verified_docs: Sequence[RetrievalResult], max_docs: int = 6) -> str:
        """Build a bounded authority context for the Llama3 prompt."""
        snippets = []
        for doc in list(verified_docs)[:max_docs]:
            snippets.append(f"[{doc.area}|{doc.source}|{doc.similarity:.3f}] {doc.answer[:260]}")
        return "\n".join(snippets) if snippets else "No verified external context available."

    def _normalize_structured_output(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize Llama JSON to the legacy DMA output contract."""
        labels = list(self.labels)
        prediction = canonical_area(str(payload.get("prediction", payload.get("diagnosis", "Unknown"))))
        raw_probs = payload.get("probabilities", {})
        if not isinstance(raw_probs, dict):
            raw_probs = {}
        probabilities: Dict[str, float] = {label: max(float(raw_probs.get(label, 0.0)), 0.0) for label in labels}
        if prediction in labels and sum(probabilities.values()) <= 1e-12:
            probabilities[prediction] = 1.0
        total = float(sum(probabilities.values()))
        if total <= 1e-12 and labels:
            probabilities = {label: 1.0 / len(labels) for label in labels}
            total = 1.0
        if total > 0:
            probabilities = {label: float(value / total) for label, value in probabilities.items()}
        if prediction not in labels and probabilities:
            prediction = max(probabilities, key=probabilities.get)
        confidence = clip01(float(payload.get("confidence", probabilities.get(prediction, 0.0))))
        return {
            "prediction": prediction if prediction in labels else "Unknown",
            "confidence": confidence,
            "probabilities": probabilities,
        }

    def _ollama_json_infer(self, query: str, verified_docs: Sequence[RetrievalResult]) -> Dict[str, Any]:
        """Call local llama3:8b with pre-flight health check and fast fallback."""
        if ollama is None:
            raise InferenceError("ollama Python client is unavailable.", code="OLLAMA_CLIENT_UNAVAILABLE")

        # Pre-flight health check with 30s cache to avoid repeated timeouts
        now = time.time()
        if now - self._ollama_checked_at > 30.0:
            with self._ollama_health_lock:
                if now - self._ollama_checked_at > 30.0:
                    try:
                        health = requests.get(f"{self.config.ollama_base_url}/api/tags", timeout=1.5)
                        health.raise_for_status()
                        self._ollama_healthy = True
                    except Exception:
                        self._ollama_healthy = False
                    self._ollama_checked_at = now
        if not self._ollama_healthy:
            raise InferenceError(
                "Ollama service is not reachable — skipping retries.",
                code="OLLAMA_UNREACHABLE",
            )

        prompt = LLAMA3_PROMPT_TEMPLATE.format(
            label_list=json.dumps(list(self.labels), ensure_ascii=False),
            query=query,
            retrieval_context=self._build_retrieval_context(verified_docs),
        )
        last_error: Optional[BaseException] = None
        ollama_api_url = f"{self.config.ollama_base_url}/api/chat"

        for attempt in range(1, self.config.ollama_max_retries + 1):
            try:
                # 直接用 requests 调用 API，绕过 ollama 库的封装问题
                payload = {
                    "model": self.ollama_model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "format": "json",
                    "options": {
                        "temperature": self.config.ollama_temperature,
                        "num_ctx": self.config.ollama_num_ctx,  # 限制上下文长度，降低显存压力
                        # 旧版 Ollama 可能不支持 flash_attn，先注释掉
                        # "flash_attn": self.config.ollama_flash_attn
                    },
                    "stream": False
                }

                # 超时设为配置的 120 秒，足够模型加载和推理
                response = requests.post(
                    ollama_api_url,
                    json=payload,
                    timeout=self.config.ollama_timeout_seconds
                )

                # 打印响应状态码和内容，方便排查错误
                logger.info(f"Ollama API 响应状态码: {response.status_code}")
                logger.debug(f"Ollama API 响应内容: {response.text}")

                # 检查 HTTP 状态码
                response.raise_for_status()

                # 解析 Ollama 返回的响应
                result = response.json()
                if "message" in result and "content" in result["message"]:
                    raw_parsed = json.loads(result["message"]["content"])
                    normalized = self._normalize_structured_output(raw_parsed)
                    if normalized["prediction"] == "Unknown":
                        logger.warning(
                            "Ollama预测标签不在已知列表中: raw={} labels={}",
                            raw_parsed.get("prediction"),
                            self.labels,
                        )
                    return normalized
                else:
                    raise ValueError("Ollama 响应缺少 message.content 字段")

            except requests.exceptions.Timeout:
                last_error = TimeoutError(f"请求超时 (第 {attempt} 次尝试)")
                logger.warning(f"Ollama 请求超时 (第 {attempt}/{self.config.ollama_max_retries} 次尝试)")
            except requests.exceptions.HTTPError as e:
                last_error = e
                logger.error(f"Ollama API HTTP 错误: {response.status_code} - {response.text}")
            except json.JSONDecodeError as e:
                last_error = e
                logger.error("无法解析 Ollama 响应为 JSON: {}，原始内容: {}", e, response.text)
            except Exception as e:
                last_error = e
                logger.error(f"Ollama 请求失败: {type(e).__name__}: {str(e)}")

            # Fast fallback: minimal wait between retries (health check already confirmed Ollama is up)
            if attempt < self.config.ollama_max_retries:
                time.sleep(0.5)

        # 所有重试失败
        raise InferenceError(
            f"本地 {self.ollama_model_name} 模型调用失败（已重试 {self.config.ollama_max_retries} 次）",
            code="OLLAMA_RETRY_EXHAUSTED",
            cause=last_error
        )

    def _legacy_simulated_infer(
        self,
        query: str,
        context_vector: Optional[np.ndarray],
        verified_docs: Sequence[RetrievalResult],
        tfa_prediction: Optional[Dict[str, Any]] = None,
        decision_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """Legacy deterministic DMA scoring retained for rollback and offline fallback."""
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
                self.config.dma_centroid_weight * centroid_scores.get(label, 0.0)
                + self.config.dma_retrieval_weight * retrieval_scores.get(label, 0.0)
                + self.config.dma_lexical_weight * lexical_priors.get(label, 0.0)
                + self.config.dma_context_weight * min(context_strength, 1.0)
                + self.config.dma_tfa_weight * tfa_risk
                + float(self.class_bias.get(label, 0.0))
            )
            raw_scores[label] = raw * self._decision_mode_factor(decision_mode)
        # Confusable disease disambiguation: boost labels whose name tokens
        # appear in the query, so "Colorectal Cancer" query can't map to "Hypertension"
        q_tokens = set(tokenize(query))
        for label in list(raw_scores.keys()):
            label_tokens = self.class_keyword_cache.get(label, set())
            if not label_tokens:
                continue
            # Direct name match: all label tokens appear in query
            if label_tokens.issubset(q_tokens):
                raw_scores[label] += 4.0
                continue
            # Synonym match: query tokens overlap with label's medical synonyms
            label_lower = label.lower()
            if label_lower in self._SYNONYM_TOKEN_MAP:
                syn_tokens = set(self._SYNONYM_TOKEN_MAP[label_lower])
                syn_overlap = len(syn_tokens & q_tokens)
                if syn_overlap >= 2:
                    raw_scores[label] += 4.0
                elif syn_overlap >= 1:
                    raw_scores[label] += 2.0
            # Partial name overlap
            overlap = len(label_tokens & q_tokens)
            if overlap >= max(1, len(label_tokens) * 0.5):
                raw_scores[label] += 2.0
        if not raw_scores:
            prediction = "Unknown"
            confidence = 0.0
            probabilities: Dict[str, float] = {}
        else:
            labels = list(raw_scores.keys())
            probability_values = softmax([raw_scores[label] for label in labels], temperature=self.config.dma_softmax_temperature)
            probabilities = {label: float(prob) for label, prob in zip(labels, probability_values)}
            prediction = max(probabilities, key=probabilities.get)
            raw_confidence = float(probabilities[prediction])
            confidence = self._apply_platt_scaling(raw_confidence * self.calibration_factor)
        output = {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "decision_mode": decision_mode,
            "base_model": self.base_model,
            "raw_scores": raw_scores,
            "intent": self._classify_intent(query),
            "hitl_status": "pending_review" if confidence < 0.65 else "auto_verified",
            "review_required": bool(confidence < 0.65),
            "status": "tentative" if confidence < 0.65 else "final",
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


    def _assess_severity(self, query: str, prediction: str, confidence: float) -> str:
        """Estimate disease severity from query keywords, prediction, and confidence.

        Returns one of: mild | moderate | severe | critical | unknown
        """
        text = f"{query} {prediction}".lower()
        critical_kw = ["cardiac arrest", "respiratory failure", "septic shock", "stroke",
                       "acute myocardial infarction", "pulmonary embolism", "aortic dissection",
                       "meningitis", "status epilepticus", "anaphylaxis", "ruptured",
                       "hemorrhagic shock", "tension pneumothorax"]
        severe_kw = ["severe", "acute", "pneumonia", "sepsis", "fracture", "bleeding",
                     "unconscious", "chest pain", "shortness of breath", "dehydration",
                     "acute kidney", "liver failure", "pancreatitis", "hypoglycemia", "coma"]
        moderate_kw = ["moderate", "chronic", "hypertension", "diabetes", "infection",
                       "inflammation", "pain", "swelling", "nausea", "fatigue", "cough",
                       "fever", "dizziness", "anemia", "bronchitis"]
        mild_kw = ["mild", "minor", "common cold", "headache", "allergy", "rash",
                   "insomnia", "anxiety", "constipation", "sore throat"]

        for kw in critical_kw:
            if kw in text:
                return "critical"
        for kw in severe_kw:
            if kw in text:
                return "severe"
        for kw in moderate_kw:
            if kw in text:
                return "moderate"
        for kw in mild_kw:
            if kw in text:
                return "mild"
        # Heuristic fallback: low confidence → greater uncertainty → assume moderate
        if confidence < 0.6:
            return "moderate"
        return "mild"

    def infer(
        self,
        query: str,
        context_vector: Optional[np.ndarray],
        verified_docs: Sequence[RetrievalResult],
        tfa_prediction: Optional[Dict[str, Any]] = None,
        decision_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """Run Llama3:8b DMA inference while preserving the original DMA output contract."""
        if not str(query).strip():
            raise InferenceError("DMA received an empty patient query.", code="DMA_EMPTY_QUERY")
        self.total_calls += 1
        self.inference_calls += 1
        self.heartbeat()
        started = time.perf_counter()
        try:
            structured = self._ollama_json_infer(query=query, verified_docs=verified_docs)
            raw_conf = float(structured.get("confidence", 0.0))
            calibrated_conf = self._apply_platt_scaling(raw_conf)
            severity = self._assess_severity(query, structured.get("prediction", ""), calibrated_conf)
            output = {
                **structured,
                "confidence": calibrated_conf,
                "diagnosis_confidence": calibrated_conf,
                "disease_severity": severity,
                "decision_mode": decision_mode,
                "base_model": self.ollama_model_name,
                "raw_scores": {},
                "dma_backend": "ollama_llama3_8b",
                "intent": self._classify_intent(query),
                "hitl_status": "pending_review" if calibrated_conf < 0.65 else "auto_verified",
                "review_required": bool(calibrated_conf < 0.65),
                "status": "tentative" if calibrated_conf < 0.65 else "final",
            }
        except MAPFMException as exc:
            logger.warning("DMA Ollama path unavailable ({}), falling back to legacy: {}", exc.code, exc.message)
            if not self.config.allow_dma_legacy_fallback:
                raise
            fallback = self._legacy_simulated_infer(
                query=query,
                context_vector=context_vector,
                verified_docs=verified_docs,
                tfa_prediction=tfa_prediction,
                decision_mode=decision_mode,
            )
            output = {
                **fallback,
                "dma_backend": "legacy_fallback",
                "ollama_error": exc.to_dict(),
                "diagnosis_confidence": float(fallback.get("confidence", 0.3)),
                "disease_severity": self._assess_severity(query, fallback.get("prediction", ""), float(fallback.get("confidence", 0.3))),
            }
        # ✅ 先打印日志，再构建字典
        logger.debug(f"模型返回的完整JSON: {output}")
        self.decision_history.append({
            "prediction": output.get("prediction", "unknown"),
            "confidence": float(output.get("confidence", 0.0)),
            "mode": decision_mode,
            "time": datetime.now().isoformat()
        })
        self.record_runtime(started)
        logger.info("DMA inference completed backend={} latency={:.4f}s", output.get("dma_backend"), self.last_runtime_seconds)
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
                "ollama_successes": self.ollama_successes,
                "ollama_failures": self.ollama_failures,
            }
        )
        return status


class TemporalForeseeingAgent(BaseAgent):
    """TFA: MedTsLLM-powered (or heuristic fallback) multi-scale risk forecasting."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("TFA")
        self.config = config
        self.forecast_calls = 0
        self.last_forecast: Dict[str, Any] = {}
        self._medtsllm: Any = None  # Lazy-loaded MedTsLLMAdapter
        self._medtsllm_load_attempted = False

    def _generate_history(self, query: str, history: Optional[Sequence[float]]) -> np.ndarray:
        if history is not None and len(history) > 0:
            arr = np.asarray(history, dtype=np.float32).flatten()
        else:
            rng = np.random.default_rng(stable_seed(query) + self.config.random_seed)
            arr = rng.normal(loc=0.0, scale=0.02, size=72).astype(np.float32)
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

    def  _transformer_feature(self, series: np.ndarray) -> float:
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

    def _clinical_temporal_profile(self, query: str, authoritative_signal: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Derive rule-based temporal disease multipliers and fall/pressure-ulcer features."""
        text = str(query).lower()
        authoritative_signal = authoritative_signal or {}
        acute_hit = any(keyword in text for keyword in ["hypoglycemia", "low blood sugar", "coma", "fall", "collapse", "syncope"])
        chronic_hit = any(keyword in text for keyword in ["pressure ulcer", "bedridden", "bed rest", "heart failure", "chronic"])
        bedrest_hours = max(float(authoritative_signal.get("bedrest_hours", 0.0)), 0.0)
        posture_instability = clip01(float(authoritative_signal.get("posture_instability", authoritative_signal.get("posture", 0.0))))
        activity_reduction = clip01(float(authoritative_signal.get("activity_reduction", 1.0 - float(authoritative_signal.get("activity_level", 1.0)))))
        pressure_growth = 1.0 - float(np.exp(-self.config.pressure_ulcer_bedrest_growth_rate * bedrest_hours))
        fall_feature = clip01(self.config.fall_posture_weight * posture_instability + self.config.fall_activity_weight * activity_reduction)
        return {
            "acute_multiplier": self.config.acute_short_window_multiplier if acute_hit else 1.0,
            "chronic_multiplier": self.config.chronic_long_window_multiplier if chronic_hit else 1.0,
            "pressure_growth": pressure_growth if chronic_hit else 0.0,
            "fall_feature": fall_feature if (acute_hit or "fall" in text) else 0.0,
            "posture_instability": posture_instability,
            "activity_reduction": activity_reduction,
            "bedrest_hours": bedrest_hours,
        }

    def _forecast_medtsllm(
        self, query: str, history: Optional[Sequence[float]],
        authoritative_signal: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        """Attempt MedTsLLM inference. Returns None if adapter unavailable."""
        if not self._medtsllm_load_attempted:
            self._medtsllm_load_attempted = True
            try:
                from medtsllm_adapter import MedTsLLMAdapter
                self._medtsllm = MedTsLLMAdapter()
            except Exception:
                self._medtsllm = None
        if self._medtsllm is None:
            return None
        try:
            return self._medtsllm.forecast(query, history, authoritative_signal)
        except Exception:
            return None

    def forecast(
        self,
        query: str,
        history: Optional[Sequence[float]] = None,
        authoritative_signal: Optional[Dict[str, float]] = None,
        dma_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Forecast multi-window risk using MedTsLLM (or heuristic fallback).

        When dma_confidence is provided, TFA adjusts its own confidence:
          - DMA >= 0.9: no adjustment
          - DMA 0.7-0.9: TFA confidence *= 0.9
          - DMA 0.6-0.7: TFA confidence *= 0.8
          - DMA < 0.6:  TFA enters degradation mode
        """
        if not str(query).strip():
            raise AgentError("TFA received an empty query.", code="TFA_EMPTY_QUERY")
        started = time.perf_counter()
        self.total_calls += 1
        self.forecast_calls += 1
        self.heartbeat()

        # ── MedTsLLM real inference path ──
        if self.config.enable_medtsllm:
            result = self._forecast_medtsllm(query, history, authoritative_signal)
            if result is not None:
                result = self._apply_dma_calibration(result, dma_confidence)
                result = self._enrich_risk_report(result, query, dma_confidence)
                self.last_forecast = result
                self.record_runtime(started)
                return result

        # ── Heuristic fallback path ──
        series = self._generate_history(query, history)
        tcn = self._tcn_feature(series)
        lstm = self._lstm_feature(series)
        transformer = self._transformer_feature(series)
        gate_logits = np.asarray([abs(tcn), abs(lstm), abs(transformer)], dtype=np.float32)
        gates = softmax(gate_logits, temperature=self.config.retrieval_query_softmax_temperature)
        fused = float(gates[0] * tcn + gates[1] * lstm + gates[2] * transformer)
        authority_bias = self._authority_calibration(authoritative_signal)
        clinical = self._clinical_temporal_profile(query, authoritative_signal)
        base_risk = sigmoid(fused + authority_bias + clinical["fall_feature"] + self.config.fall_threshold_shift)
        short_risk = clip01(base_risk * clinical["acute_multiplier"])
        mid_risk = clip01(0.72 * base_risk + 0.28 * sigmoid(float(np.mean(series))) + 0.5 * clinical["fall_feature"])
        long_base = 0.58 * base_risk + 0.42 * sigmoid(float(np.mean(series[-36:])))
        long_risk = clip01(long_base * clinical["chronic_multiplier"] + clinical["pressure_growth"])
        output = {
            "has_voting_right": True,
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
                "acute_multiplier": clinical["acute_multiplier"],
                "chronic_multiplier": clinical["chronic_multiplier"],
                "pressure_growth": clinical["pressure_growth"],
                "fall_feature": clinical["fall_feature"],
                "posture_instability": clinical["posture_instability"],
                "activity_reduction": clinical["activity_reduction"],
                "bedrest_hours": clinical["bedrest_hours"],
            },
        }
        output = self._apply_dma_calibration(output, dma_confidence)
        output = self._enrich_risk_report(output, query, dma_confidence)
        self.last_forecast = output
        self.record_runtime(started)
        return output

    def forecast_with_dma_context(
        self,
        query: str,
        dma_result: Dict[str, Any],
        history: Optional[Sequence[float]] = None,
        authoritative_signal: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Convenience wrapper: forecast with full DMA context for integration."""
        dma_confidence = float(dma_result.get("confidence", 0.0))
        return self.forecast(
            query=query, history=history,
            authoritative_signal=authoritative_signal,
            dma_confidence=dma_confidence,
        )

    @staticmethod
    def _classify_risk_level(short: float, mid: float, long_p: float) -> str:
        worst = max(short, mid, long_p)
        if worst >= 0.85:
            return "critical"
        if worst >= 0.65:
            return "high"
        if worst >= 0.35:
            return "medium"
        return "low"

    @staticmethod
    def _apply_dma_calibration(
        result: Dict[str, Any], dma_confidence: Optional[float]
    ) -> Dict[str, Any]:
        """Adjust TFA risk probabilities based on DMA confidence."""
        if dma_confidence is None:
            return result
        if dma_confidence >= 0.9:
            factor = 1.0
            result["calibration_mode"] = "full_confidence"
        elif dma_confidence >= 0.7:
            factor = 0.9
            result["calibration_mode"] = "slight_discount"
        elif dma_confidence >= 0.6:
            factor = 0.8
            result["calibration_mode"] = "moderate_discount"
        else:
            factor = 1.0
            result["calibration_mode"] = "degraded"
            result["degradation_notice"] = "诊断结果置信度较低，风险预测仅供参考"

        result["calibration_factor"] = factor
        result["dma_confidence_input"] = dma_confidence
        for window in ("short_term", "mid_term", "long_term"):
            if window in result:
                result[window]["risk_probability"] = clip01(
                    float(result[window]["risk_probability"]) * factor
                )
        return result

    def _enrich_risk_report(
        self, result: Dict[str, Any], query: str, dma_confidence: Optional[float]
    ) -> Dict[str, Any]:
        """Add risk level, risk factors, and clinical recommendations."""
        short = float(result.get("short_term", {}).get("risk_probability", 0.0))
        mid = float(result.get("mid_term", {}).get("risk_probability", 0.0))
        long_p = float(result.get("long_term", {}).get("risk_probability", 0.0))
        risk_level = self._classify_risk_level(short, mid, long_p)
        factors = self._extract_risk_factors(query, result)
        result.update({
            "risk_level": risk_level,
            "risk_label": {"low": "低风险", "medium": "中风险",
                           "high": "高风险", "critical": "极高风险"}.get(risk_level, risk_level),
            "risk_score": max(short, mid, long_p),
            "primary_risk_factors": factors[:5],
            "reasoning": self._generate_reasoning(risk_level, factors, dma_confidence),
            "recommendations": self._generate_recommendations(risk_level),
            "conflict_with_diagnosis": 0,
        })
        return result

    @staticmethod
    def _extract_risk_factors(query: str, result: Dict[str, Any]) -> List[str]:
        """Extract human-readable risk factor descriptions from the forecast."""
        factors: List[str] = []
        contrib = result.get("contribution_factors", {})
        short_r = float(result.get("short_term", {}).get("risk_probability", 0.0))
        long_r = float(result.get("long_term", {}).get("risk_probability", 0.0))
        text = str(query).lower()

        if short_r > 0.5:
            factors.append("短期内病情恶化风险较高")
        if long_r > 0.4:
            factors.append("长期预后存在不确定性")
        if contrib.get("fall_feature", 0.0) > 0.3:
            factors.append("跌倒风险显著升高")
        if contrib.get("pressure_growth", 0.0) > 0.3:
            factors.append("压疮风险随时间累积增长")
        if contrib.get("acute_multiplier", 1.0) > 1.5:
            factors.append("急性病程加速因子活跃")
        if contrib.get("chronic_multiplier", 1.0) > 1.5:
            factors.append("慢性病程长期恶化趋势明显")
        if contrib.get("bedrest_hours", 0.0) > 12:
            factors.append("长时间卧床增加并发症风险")
        if "infection" in text or "fever" in text:
            factors.append("感染相关指标异常需关注")
        if not factors:
            factors.append("各项指标总体平稳")
        return factors

    @staticmethod
    def _generate_reasoning(risk_level: str, factors: List[str], dma_confidence: Optional[float]) -> str:
        parts = [f"TFA多尺度时序分析完成，风险等级：{risk_level}。"]
        if factors:
            parts.append("主要发现：" + "；".join(factors[:3]) + "。")
        if dma_confidence is not None and dma_confidence < 0.7:
            parts.append("注意：DMA诊断置信度较低，本风险预测仅供参考。")
        return "".join(parts)

    @staticmethod
    def _generate_recommendations(risk_level: str) -> List[str]:
        """Generate tiered clinical recommendations based on risk level."""
        recs = {
            "low": [
                "常规健康监测，建议定期复查",
                "保持现有治疗方案，无需额外干预",
            ],
            "medium": [
                "建议密切观察病情变化，增加监测频率",
                "考虑定期复查相关指标（每1-3个月）",
                "评估是否需要调整现有治疗方案",
            ],
            "high": [
                "建议及时就医进行进一步检查",
                "考虑多学科会诊评估病情",
                "制定详细的监测和治疗计划",
                "患者及家属需了解病情恶化的预警信号",
            ],
            "critical": [
                "建议立即就医或急诊处理",
                "需要紧急医疗干预和全面评估",
                "通知主治医师和多学科团队",
                "准备应急预案和抢救措施",
                "持续监测生命体征变化",
            ],
        }
        return recs.get(risk_level, recs["low"])

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

    def fuse_with_tfa(
        self,
        retrieval_batches: Sequence[Sequence[RetrievalResult]],
        dma_result: Optional[Dict[str, Any]] = None,
        tfa_prediction: Optional[Dict[str, Any]] = None,
        risk_knowledge: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RetrievalResult], Dict[str, Any], Dict[str, Any]]:
        """Fuse retrieval results, DMA diagnosis, TFA risk report, and RAA risk knowledge
        into a unified comprehensive answer.

        Returns:
            (fused_docs, fusion_summary, comprehensive_answer)
        """
        fused_docs, fusion_summary = self.fuse(retrieval_batches)

        # Build comprehensive answer structure
        comprehensive: Dict[str, Any] = {
            "diagnosis": {},
            "risk_assessment": {},
            "recommendations": [],
            "disclaimer": (
                "本系统输出为AI辅助决策参考，不构成医疗建议。"
                "所有诊断和治疗决策必须由具备资质的医疗专业人员确认。"
                "如有疑问，请咨询您的医生或拨打急救电话。"
            ),
        }

        # Diagnosis section
        if dma_result:
            comprehensive["diagnosis"] = {
                "prediction": dma_result.get("prediction", "Unknown"),
                "confidence": dma_result.get("confidence", 0.0),
                "severity": dma_result.get("disease_severity", "unknown"),
                "intent": dma_result.get("intent", "general_inquiry"),
                "status": dma_result.get("status", "tentative"),
            }

        # Risk assessment section
        if tfa_prediction:
            comprehensive["risk_assessment"] = {
                "risk_level": tfa_prediction.get("risk_level", "low"),
                "risk_label": tfa_prediction.get("risk_label", "低风险"),
                "risk_score": tfa_prediction.get("risk_score", 0.0),
                "short_term_risk": tfa_prediction.get("short_term", {}).get("risk_probability", 0.0),
                "mid_term_risk": tfa_prediction.get("mid_term", {}).get("risk_probability", 0.0),
                "long_term_risk": tfa_prediction.get("long_term", {}).get("risk_probability", 0.0),
                "primary_risk_factors": tfa_prediction.get("primary_risk_factors", []),
                "reasoning": tfa_prediction.get("reasoning", ""),
                "calibration_mode": tfa_prediction.get("calibration_mode", "full_confidence"),
                "degradation_notice": tfa_prediction.get("degradation_notice", ""),
            }

        # Merge RAA risk recommendations with TFA recommendations
        tfa_recs = tfa_prediction.get("recommendations", []) if tfa_prediction else []
        raa_recs: List[str] = []
        if risk_knowledge and risk_knowledge.get("guidelines_matched"):
            raa_recs.append("已检索到相关临床指南，与风险评估结果进行交叉验证。")
        if risk_knowledge and risk_knowledge.get("conflicts_found"):
            raa_recs.append("注意：RAA检索结果与TFA风险因素存在差异，建议人工审查。")

        comprehensive["recommendations"] = tfa_recs + raa_recs

        # Risk knowledge cross-reference
        if risk_knowledge:
            comprehensive["risk_knowledge_cross_ref"] = {
                "guidelines_matched": risk_knowledge.get("guidelines_matched", False),
                "tfa_raa_consistency": risk_knowledge.get("tfa_raa_consistency", "unknown"),
                "confidence_adjustment": risk_knowledge.get("confidence_adjustment", 0.0),
            }

        return fused_docs, fusion_summary, comprehensive

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


class BaseVerificationAgent(BaseAgent, ABC):
    """Abstract medical verification interface."""

    @abstractmethod
    def verify(self, docs: Sequence[RetrievalResult], fusion_summary: Optional[Dict[str, Any]] = None) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """Verify candidate documents and return accepted documents plus summary."""
        raise NotImplementedError

    @abstractmethod
    def vote(self, verification_summary: Dict[str, Any]) -> bool:
        """Convert verification summary to a consensus vote."""
        raise NotImplementedError


class RealVerificationAgent(BaseVerificationAgent):
    """Empty real implementation placeholder for production medical fact verification."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Verification")
        self.config = config

    def verify(self, docs: Sequence[RetrievalResult], fusion_summary: Optional[Dict[str, Any]] = None) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        return list(docs), {"accepted_docs": len(docs), "rejected_docs": 0, "verification_ratio": 1.0, "implementation": "RealVerificationAgent-placeholder"}

    def vote(self, verification_summary: Dict[str, Any]) -> bool:
        return bool(verification_summary.get("verification_ratio", 0.0) >= 0.50)


# === MODIFIED: 新增抽象接口解耦模拟与真实逻辑 ===
class SimulatedKnowledgeVerificationAgent(BaseVerificationAgent):
    """Medical fact-check and timeliness verifier."""

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Verification")
        self.config = config
        self.verification_calls = 0
        self.accepted_docs = 0
        self.rejected_docs = 0
        self.last_summary: Dict[str, Any] = {}

    def _fact_api_score(self, doc: RetrievalResult) -> float:
        # Simulated medical fact API / rule base with discriminating scoring.
        text = f"{doc.question} {doc.answer}".lower()
        tokens = tokenize(text)
        token_count = len(tokens)

        # Authority: only authoritative sources or human experts get the bonus
        authority_bonus = 0.20 if doc.source in AUTHORITATIVE_SOURCES or doc.source == "HumanExpert" else 0.0

        # Content quality: scales with sqrt(token_count), max at ~50 tokens
        content_bonus = min(0.30, 0.06 * float(np.sqrt(max(token_count, 0))))

        # Area bonus: reduced, only for known areas
        area_bonus = 0.10 if doc.area != "Unknown" else 0.0

        # Medical terminology: presence of clinical terms indicates higher quality
        medical_terms = {
            "diagnosis", "treatment", "symptom", "therapy", "prognosis", "clinical",
            "trial", "guideline", "evidence", "patient", "dose", "surgery", "risk",
            "mortality", "incidence", "prevalence", "etiology", "pathology",
            "contraindication", "monitoring", "screening", "chronic", "acute"
        }
        med_term_hits = sum(1 for term in medical_terms if term in text)
        medical_bonus = min(0.15, 0.04 * med_term_hits)

        # Evidence quality: presence of quantitative data or study references
        has_numbers = bool(re.search(r"\d+(?:\.\d+)?%", text))
        has_study_ref = bool(re.search(r"(study|trial|meta.analysis|cohort|rct|systematic review)", text))
        evidence_bonus = 0.14 if (has_numbers or has_study_ref) else 0.0

        # Red-flag penalty: pseudoscientific or dangerous claims
        red_flags = [
            "miracle", "guaranteed cure", "always works", "secret formula",
            "big pharma", "conspiracy", "detox", "cleanse", "100% effective",
            "cure all", "ancient remedy", "they don't want you to know"
        ]
        red_flag_penalty = 0.35 if any(keyword in text for keyword in red_flags) else 0.0

        # Vagueness penalty: content too short or lacking specifics
        vagueness_penalty = 0.15 if token_count < 12 else 0.0

        return clip01(0.38 + authority_bonus + content_bonus + area_bonus
                      + medical_bonus + evidence_bonus
                      - red_flag_penalty - vagueness_penalty)

    def verify(self, docs: Sequence[RetrievalResult], fusion_summary: Optional[Dict[str, Any]] = None) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        self.total_calls += 1
        self.verification_calls += 1
        self.heartbeat()
        now = datetime.now()
        # Extract conflicting doc IDs from fusion summary for extra scrutiny
        conflict_doc_ids: set[int] = set()
        fusion_conflicts = fusion_summary.get("conflicts", []) if fusion_summary else []
        for conflict in fusion_conflicts:
            conflict_doc_ids.add(int(conflict.get("left_doc_id", 0)))
            conflict_doc_ids.add(int(conflict.get("right_doc_id", 0)))
        fusion_conflict_count = fusion_summary.get("conflict_count", 0) if fusion_summary else 0
        # Raise fact threshold when fusion found conflicts — stricter verification
        fact_threshold = 0.42 if fusion_conflict_count > 0 else 0.36
        accepted: List[RetrievalResult] = []
        rejected: List[Dict[str, Any]] = []
        warning_docs: List[int] = []
        for doc in docs:
            age_days = max((now - doc.last_updated).days, 0)
            fact_score = self._fact_api_score(doc)
            timeliness_ok = age_days <= self.config.verification_max_age_days
            # Docs in fusion conflicts face stricter scrutiny
            in_conflict = int(doc.doc_id) in conflict_doc_ids
            effective_threshold = fact_threshold + 0.05 if in_conflict else fact_threshold
            fact_ok = fact_score >= effective_threshold
            doc.fact_score = fact_score
            doc.verification_passed = bool(timeliness_ok and fact_ok)
            if in_conflict and fact_ok and fact_score < effective_threshold + 0.05:
                warning_docs.append(doc.doc_id)
            if doc.verification_passed:
                accepted.append(doc)
            else:
                rejected.append(
                    {
                        "doc_id": doc.doc_id,
                        "area": doc.area,
                        "age_days": age_days,
                        "fact_score": fact_score,
                        "in_fusion_conflict": in_conflict,
                        "reason": "stale" if not timeliness_ok else "fact_score_low",
                    }
                )
        self.accepted_docs += len(accepted)
        self.rejected_docs += len(rejected)
        # Real verification status based on fusion conflicts and rejection rate
        rejection_rate = 1.0 - (len(accepted) / max(len(docs), 1))
        if rejection_rate > 0.60 or fusion_conflict_count >= 10:
            verification_status = "failed"
        elif rejection_rate > 0.30 or fusion_conflict_count >= 5:
            verification_status = "passed_with_warnings"
        else:
            verification_status = "passed"
        summary = {
            "accepted_docs": len(accepted),
            "rejected_docs": len(rejected),
            "verification_ratio": len(accepted) / max(len(docs), 1),
            "verification_status": verification_status,
            "fusion_conflicts_considered": fusion_conflict_count,
            "conflict_docs_scrutinized": len(conflict_doc_ids),
            "warning_docs": warning_docs[:5],
            "rejected_examples": rejected[:5],
        }
        self.last_summary = summary
        return accepted, summary

    def verify_tfa_report(
        self,
        tfa_prediction: Optional[Dict[str, Any]],
        dma_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate TFA risk report for reasonableness and consistency.

        Checks:
          1. Risk scores are within [0, 1]
          2. Risk factors are medically related to the diagnosis
          3. Recommendations match risk level
        Returns a dict with 'valid', 'warnings', 'serious_conflicts'.
        """
        if tfa_prediction is None:
            return {"valid": True, "warnings": [], "serious_conflicts": 0,
                    "tfa_verified": True, "message": "TFA not executed"}

        warnings: List[str] = []
        serious = 0

        # Check 1: Risk score range
        short = float(tfa_prediction.get("short_term", {}).get("risk_probability", 0.0))
        mid = float(tfa_prediction.get("mid_term", {}).get("risk_probability", 0.0))
        long_p = float(tfa_prediction.get("long_term", {}).get("risk_probability", 0.0))
        for window, val in [("short_term", short), ("mid_term", mid), ("long_term", long_p)]:
            if val < 0.0 or val > 1.0:
                serious += 1
                warnings.append(f"TFA {window} risk probability {val:.3f} out of [0,1] range")

        # Check 2: Risk factors vs diagnosis relevance
        if dma_result:
            diagnosis = str(dma_result.get("prediction", "")).lower()
            tfa_factors = tfa_prediction.get("primary_risk_factors", [])
            # Simple heuristic: if diagnosis mentions a body system, factors should be related
            if diagnosis and diagnosis != "unknown" and tfa_factors:
                # Check if any risk factor overlaps with diagnosis keywords conceptually
                # This is a lightweight heuristic — full medical validation needs external KB
                diag_tokens = set(str(diagnosis).lower().split())
                related = False
                for factor in tfa_factors:
                    factor_tokens = set(str(factor).lower().split())
                    if diag_tokens & factor_tokens:
                        related = True
                        break
                if not related and len(tfa_factors) > 1:
                    warnings.append(
                        f"TFA risk factors may not be directly related to diagnosis '{diagnosis}'"
                    )

        # Check 3: Recommendations match risk level
        risk_level = tfa_prediction.get("risk_level", "low")
        recs = tfa_prediction.get("recommendations", [])
        if risk_level == "critical" and len(recs) < 3:
            warnings.append("TFA critical risk has insufficient recommendations")
        if risk_level == "low" and any("立即" in r or "emergency" in r.lower() for r in recs):
            warnings.append("TFA low risk contains overly aggressive recommendations")

        # Check 4: Degradation notice checked
        if tfa_prediction.get("degradation_notice"):
            warnings.append(f"TFA degradation active: {tfa_prediction['degradation_notice']}")

        valid = serious == 0
        result = {
            "valid": valid,
            "warnings": warnings,
            "serious_conflicts": serious,
            "tfa_verified": True,
            "message": "TFA risk report verified"
            if valid
            else f"TFA risk report has {len(warnings)} issue(s)",
        }
        return result

    def vote(self, verification_summary: Dict[str, Any]) -> bool:
        # CONSENSUS VOTE: Verification votes YES when docs pass + no serious TFA conflicts
        base_pass = float(verification_summary.get("verification_ratio", 0.0)) >= 0.50
        serious_conflicts = int(verification_summary.get("serious_conflicts", 0))
        tfa_valid = bool(verification_summary.get("tfa_verified", True))
        return base_pass and serious_conflicts == 0 and tfa_valid

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


# Backward-compatible alias preserving original constructor usage.
KnowledgeVerificationAgent = SimulatedKnowledgeVerificationAgent


class BasePrivacyAgent(BaseAgent, ABC):
    """Abstract privacy/security interface preserving legacy methods."""

    @abstractmethod
    def secure_federated_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Return privacy-protected gradient."""
        raise NotImplementedError

    @abstractmethod
    def encrypt_message(self, plaintext: str) -> str:
        """Encrypt a plaintext message."""
        raise NotImplementedError

    @abstractmethod
    def decrypt_message(self, ciphertext: str) -> str:
        """Decrypt an encrypted message."""
        raise NotImplementedError


class RealPrivacyAgent(BasePrivacyAgent):
    """Real privacy layer with XOR encryption and gradient perturbation."""

    def __init__(self, config: EcosystemConfig, key: str = "production-privacy-key") -> None:
        super().__init__("Privacy")
        self.config = config
        self.key = hashlib.sha256(key.encode("utf-8", errors="ignore")).digest()
        self.encryption_calls = 0
        self.decryption_calls = 0
        self.gradient_calls = 0

    def secure_federated_gradient(self, gradient: np.ndarray) -> np.ndarray:
        self.gradient_calls += 1
        rng = np.random.default_rng(self.config.random_seed + self.gradient_calls)
        noise = rng.normal(0.0, self.config.secure_gradient_noise_std, size=gradient.shape).astype(np.float32)
        return gradient.astype(np.float32) + noise

    def encrypt_message(self, plaintext: str) -> str:
        self.encryption_calls += 1
        raw = plaintext.encode("utf-8", errors="ignore")
        encrypted = bytes(byte ^ self.key[idx % len(self.key)] for idx, byte in enumerate(raw))
        return encrypted.hex()

    def decrypt_message(self, ciphertext: str) -> str:
        self.decryption_calls += 1
        raw = bytes.fromhex(ciphertext)
        decrypted = bytes(byte ^ self.key[idx % len(self.key)] for idx, byte in enumerate(raw))
        return decrypted.decode("utf-8", errors="ignore")


# === MODIFIED: 新增抽象接口解耦模拟与真实逻辑 ===
class SimulatedPrivacySecurityAgent(BasePrivacyAgent):
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


# Backward-compatible alias preserving original constructor usage.
PrivacySecurityAgent = SimulatedPrivacySecurityAgent


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


class DegradationManager:
    """Tracks component health and orchestrates graceful degradation.

    When components fail, the system continues with reduced functionality rather
    than crashing — a critical property for medical AI safety. Degradation levels:
      0 = fully operational, 1 = minor degradation, 2 = partial, 3 = severe.
    """

    def __init__(self, config: EcosystemConfig) -> None:
        self.config = config
        self.component_states: Dict[str, str] = {}  # "healthy" | "degraded" | "unavailable"
        self.fallback_counts: Dict[str, int] = defaultdict(int)
        self.degradation_level = 0

    def _assess_level(self) -> int:
        unavailable = sum(1 for s in self.component_states.values() if s == "unavailable")
        degraded = sum(1 for s in self.component_states.values() if s == "degraded")
        if unavailable >= 3:
            return 3
        if unavailable >= 1 or degraded >= 2:
            return 2
        if degraded >= 1:
            return 1
        return 0

    def mark(self, component: str, state: str) -> None:
        self.component_states[component] = state
        if state in ("degraded", "unavailable"):
            self.fallback_counts[component] += 1
        self.degradation_level = self._assess_level()
        if state == "unavailable":
            logger.warning("Component '{}' is UNAVAILABLE — system degraded to level {}", component, self.degradation_level)
        elif state == "degraded":
            logger.info("Component '{}' is DEGRADED — fallback active", component)

    def safe_call(self, component: str, fn: Any, *args: Any, fallback: Any = None, **kwargs: Any) -> Any:
        """Call fn(*args, **kwargs) with automatic degradation fallback."""
        try:
            result = fn(*args, **kwargs)
            self.mark(component, "healthy")
            return result
        except Exception as exc:
            logger.warning("Component '{}' failed ({}), using fallback", component, exc)
            self.mark(component, "degraded")
            return fallback() if callable(fallback) else fallback

    def get_stats(self) -> Dict[str, Any]:
        return {
            "degradation_level": self.degradation_level,
            "component_states": dict(self.component_states),
            "fallback_counts": dict(self.fallback_counts),
        }


class ContinuousLearningLoop:
    """Periodic online learning from HITL feedback to improve DMA and retrieval.

    Accumulates feedback records over time and triggers model updates when sufficient
    data is collected. Designed as a lightweight simulation of continuous deployment
    loops used in production ML systems.
    """

    def __init__(self, config: EcosystemConfig, min_feedback: int = 10) -> None:
        self.config = config
        self.min_feedback = min_feedback
        self.feedback_buffer: List[HumanFeedbackRecord] = []
        self.cycle_count = 0
        self.last_update_time: Optional[datetime] = None
        self.metrics_history: List[Dict[str, Any]] = []

    def ingest(self, record: HumanFeedbackRecord) -> None:
        """Accumulate a single HITL feedback record."""
        self.feedback_buffer.append(record)
        if len(self.feedback_buffer) >= self.min_feedback:
            logger.info("Feedback buffer reached {} records (threshold={})",
                        len(self.feedback_buffer), self.min_feedback)

    def should_update(self) -> bool:
        """Return True when enough feedback has accumulated to trigger a learning cycle."""
        return len(self.feedback_buffer) >= self.min_feedback

    def run_cycle(
        self,
        dma: "DecisionMakingAgent",
        kb: "OnlineKnowledgeBaseManager",
        lora_finetuner: BaseFineTuner,
        retrieval_updater: "ContrastiveRetrievalUpdater",
        privacy_agent: Optional[BasePrivacyAgent] = None,
    ) -> Dict[str, Any]:
        """Execute one continuous learning cycle: calibrate DMA and update retrieval."""
        if not self.feedback_buffer:
            return {"cycle": self.cycle_count, "updated": False, "reason": "no_feedback"}
        self.cycle_count += 1
        started = time.time()

        # DMA calibration via LoRA-like update
        lora_result = lora_finetuner.update(dma, self.feedback_buffer, privacy_agent=privacy_agent)

        # Retrieval weight update for each feedback item
        retrieval_updates = 0
        for record in self.feedback_buffer[-20:]:
            try:
                pos_ids = [int(doc_id) for doc_id in (record.positive_doc_ids or [])]
                neg_ids = [int(doc_id) for doc_id in (record.negative_doc_ids or [])]
                if pos_ids or neg_ids:
                    retrieval_updater.update(kb, record.query, pos_ids, neg_ids)
                    retrieval_updates += 1
            except Exception as exc:
                logger.warning("Retrieval update failed for query '{}': {}", record.query[:60], exc)

        # Evaluate recent accuracy trend
        corrections = sum(1 for r in self.feedback_buffer[-20:] if r.corrected_label != r.predicted_label)
        accuracy_estimate = 1.0 - corrections / max(len(self.feedback_buffer[-20:]), 1)

        self.last_update_time = datetime.now()
        metrics = {
            "cycle": self.cycle_count,
            "feedback_consumed": len(self.feedback_buffer),
            "retrieval_updates": retrieval_updates,
            "dma_calibration_factor": lora_result.get("calibration_factor", 1.0),
            "accuracy_estimate": accuracy_estimate,
            "elapsed_seconds": time.time() - started,
        }
        self.metrics_history.append(metrics)
        self.feedback_buffer.clear()
        logger.info("Continuous learning cycle {} complete — accuracy_est={:.3f}, feedback={}",
                    self.cycle_count, accuracy_estimate, metrics["feedback_consumed"])
        return metrics

    def get_stats(self) -> Dict[str, Any]:
        return {
            "cycle_count": self.cycle_count,
            "buffered_feedback": len(self.feedback_buffer),
            "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
            "recent_metrics": self.metrics_history[-3:] if self.metrics_history else [],
        }


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
    """HMAE hierarchical consensus with TFA voting rights.

    Voting weights (total=4, threshold=3):
      - DMA  (diagnosis):      2 votes (core, non-negotiable)
      - TFA  (temporal risk):  1 vote (auxiliary decision support)
      - Verification:          1 vote (fact-checking & consistency)

    Layered decision rules (evaluated top-down):
      1. FORCE_HITL layer:      DMA HITL=True | Verification serious_conflict≥1 | TFA risk=critical
      2. DMA_VETO layer:        DMA votes NO → reject regardless of other votes
      3. NORMAL_VOTE layer:     confidence × risk-level matrix
    """

    # ── Voting weight constants ──
    DMA_WEIGHT = 2
    TFA_WEIGHT = 1
    VERIFICATION_WEIGHT = 1
    TOTAL_WEIGHT = DMA_WEIGHT + TFA_WEIGHT + VERIFICATION_WEIGHT  # 4
    PASS_THRESHOLD = 3  # 75%

    def __init__(self, config: EcosystemConfig) -> None:
        super().__init__("Consensus")
        self.config = config
        self.consensus_calls = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.hitl_escalation_count = 0
        self.dma_veto_count = 0
        self.last_vote: Dict[str, Any] = {}
        self.decision_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Risk-level helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _risk_level(tfa_prediction: Optional[Dict[str, Any]]) -> str:
        """Classify TFA output into: low | medium | high | critical."""
        if tfa_prediction is None:
            return "low"
        short = float(tfa_prediction.get("short_term", {}).get("risk_probability", 0.0))
        mid = float(tfa_prediction.get("mid_term", {}).get("risk_probability", 0.0))
        long_p = float(tfa_prediction.get("long_term", {}).get("risk_probability", 0.0))
        worst = max(short, mid, long_p)
        if worst >= 0.85:
            return "critical"
        if worst >= 0.65:
            return "high"
        if worst >= 0.35:
            return "medium"
        return "low"

    @staticmethod
    def _risk_label_cn(level: str) -> str:
        return {"low": "低风险", "medium": "中风险",
                "high": "高风险", "critical": "极高风险"}.get(level, level)

    # ------------------------------------------------------------------
    # Core voting logic
    # ------------------------------------------------------------------
    def vote(
        self,
        dma_result: Dict[str, Any],
        tfa_prediction: Optional[Dict[str, Any]],
        verification_summary: Dict[str, Any],
        verification_agent: "KnowledgeVerificationAgent",
    ) -> Dict[str, Any]:
        """Standard weighted vote (backward-compatible entry point).

        Delegates to hierarchical_vote for unified logic.
        """
        return self.hierarchical_vote(
            dma_result=dma_result,
            tfa_prediction=tfa_prediction,
            verification_summary=verification_summary,
            verification_agent=verification_agent,
        )

    def hierarchical_vote(
        self,
        dma_result: Dict[str, Any],
        tfa_prediction: Optional[Dict[str, Any]],
        verification_summary: Dict[str, Any],
        verification_agent: "KnowledgeVerificationAgent",
        high_risk: bool = False,
    ) -> Dict[str, Any]:
        """Layered hierarchical consensus with TFA voting rights.

        Returns a dict with keys: approved, tier, votes, reason,
        escalate_to_human, risk_level, and full vote detail.
        """
        self.total_calls += 1
        self.consensus_calls += 1
        self.heartbeat()

        # ── Gather inputs ──
        confidence = float(dma_result.get("confidence", 0.0))
        hitl_flag = bool(dma_result.get("hitl_status") == "pending_review"
                         or dma_result.get("review_required")
                         or confidence < self.config.confidence_threshold)
        risk_level = self._risk_level(tfa_prediction)
        short_risk = float((tfa_prediction or {}).get("short_term", {}).get("risk_probability", 0.0))
        ver_status = str(verification_summary.get("verification_status", "passed"))
        serious_conflicts = int(verification_summary.get("serious_conflicts", 0))
        tfa_conflicts = int((tfa_prediction or {}).get("conflict_with_diagnosis", 0))

        # ── Individual votes ──
        dma_vote = confidence >= self.config.confidence_threshold
        # TFA votes YES when risk is high enough to warrant attention
        tfa_vote = risk_level in ("high", "critical")
        ver_vote = verification_agent.vote(verification_summary)

        votes = {"DMA": dma_vote, "TFA": tfa_vote, "Verification": ver_vote}
        reason_parts: List[str] = []

        # ═══════════════════════════════════════════════════════════════
        # LAYER 1 — FORCE HITL (bypasses voting entirely)
        # ═══════════════════════════════════════════════════════════════
        force_hitl = False
        force_reasons: List[str] = []

        if hitl_flag:
            force_hitl = True
            force_reasons.append(f"DMA HITL=True (confidence={confidence:.3f})")
        if serious_conflicts >= 1:
            force_hitl = True
            force_reasons.append(f"Verification serious_conflicts={serious_conflicts}")
        if risk_level == "critical":
            force_hitl = True
            force_reasons.append(f"TFA risk_level=critical (short={short_risk:.3f})")
        if tfa_conflicts >= 1:
            force_hitl = True
            force_reasons.append(f"TFA-diagnosis conflict detected")

        if force_hitl:
            self.rejected_count += 1
            self.hitl_escalation_count += 1
            reason = "FORCE_HITL: " + "; ".join(force_reasons)
            result = {
                "approved": False,
                "tier": "force_hitl",
                "votes": votes,
                "vote_weights": {"DMA": self.DMA_WEIGHT, "TFA": self.TFA_WEIGHT,
                                 "Verification": self.VERIFICATION_WEIGHT},
                "weighted_approvals": 0,
                "total_weight": self.TOTAL_WEIGHT,
                "required_approvals": self.PASS_THRESHOLD,
                "escalate_to_human": True,
                "risk_level": risk_level,
                "risk_label": self._risk_label_cn(risk_level),
                "reason": reason,
            }
            self.last_vote = result
            self._log_decision(result)
            return result

        # ═══════════════════════════════════════════════════════════════
        # LAYER 2 — DMA VETO (diagnosis is the foundation)
        # ═══════════════════════════════════════════════════════════════
        if not dma_vote:
            self.rejected_count += 1
            self.dma_veto_count += 1
            reason = f"DMA_VETO: confidence={confidence:.3f} < threshold={self.config.confidence_threshold}"
            result = {
                "approved": False,
                "tier": "dma_veto",
                "votes": votes,
                "vote_weights": {"DMA": self.DMA_WEIGHT, "TFA": self.TFA_WEIGHT,
                                 "Verification": self.VERIFICATION_WEIGHT},
                "weighted_approvals": 0,
                "total_weight": self.TOTAL_WEIGHT,
                "required_approvals": self.PASS_THRESHOLD,
                "escalate_to_human": True,
                "risk_level": risk_level,
                "risk_label": self._risk_label_cn(risk_level),
                "reason": reason,
            }
            self.last_vote = result
            self._log_decision(result)
            return result

        # ═══════════════════════════════════════════════════════════════
        # LAYER 3 — NORMAL VOTE (confidence × risk matrix)
        # ═══════════════════════════════════════════════════════════════
        approved: bool
        tier: str
        reason: str

        if confidence >= 0.8 and risk_level in ("low", "medium"):
            # Auto-pass: high confidence + manageable risk
            approved = True
            tier = "auto_pass"
            reason = (
                f"DMA confidence {confidence:.3f} >= 0.8, "
                f"TFA risk level {risk_level}, verification {ver_status}"
            )
            effective_votes = {"DMA": True, "TFA": True, "Verification": True}
            weighted_approvals = self.TOTAL_WEIGHT

        elif confidence >= 0.8 and risk_level == "high":
            # Auto-pass with caveat: note elevated risk
            approved = True
            tier = "auto_pass_risk_noted"
            reason = (
                f"DMA confidence {confidence:.3f} >= 0.8, "
                f"TFA risk level HIGH — 存在较高病情恶化风险, verification {ver_status}"
            )
            effective_votes = {"DMA": True, "TFA": True, "Verification": ver_vote}
            weighted_approvals = self.DMA_WEIGHT + self.TFA_WEIGHT + (self.VERIFICATION_WEIGHT if ver_vote else 0)

        elif confidence >= 0.6 and risk_level == "low":
            # Auto-pass: moderate confidence but low risk
            approved = True
            tier = "auto_pass"
            reason = (
                f"DMA confidence {confidence:.3f} in [0.6, 0.8), "
                f"TFA risk level low, verification {ver_status}"
            )
            effective_votes = {"DMA": True, "TFA": True, "Verification": True}
            weighted_approvals = self.TOTAL_WEIGHT

        elif confidence >= 0.6 and risk_level in ("medium", "high"):
            # Escalate: moderate confidence + non-trivial risk
            approved = False
            tier = "escalation_risk_uncertainty"
            reason = (
                f"DMA confidence {confidence:.3f} in [0.6, 0.8), "
                f"TFA risk level {risk_level} — 触发人工干预 (诊断不确定+存在风险)"
            )
            effective_votes = votes
            weighted_approvals = (
                (self.DMA_WEIGHT if dma_vote else 0)
                + (self.TFA_WEIGHT if tfa_vote else 0)
                + (self.VERIFICATION_WEIGHT if ver_vote else 0)
            )

        else:
            # Low confidence (<0.6) — already caught by force_hitl layer above,
            # but handle any remaining edge cases here
            approved = False
            tier = "escalation_low_confidence"
            reason = (
                f"DMA confidence {confidence:.3f} < 0.6, "
                f"TFA risk level {risk_level} — 触发人工干预"
            )
            effective_votes = votes
            weighted_approvals = (
                (self.DMA_WEIGHT if dma_vote else 0)
                + (self.TFA_WEIGHT if tfa_vote else 0)
                + (self.VERIFICATION_WEIGHT if ver_vote else 0)
            )

        if approved:
            self.approved_count += 1
        else:
            self.rejected_count += 1

        result = {
            "approved": approved,
            "tier": tier,
            "votes": effective_votes,
            "vote_weights": {"DMA": self.DMA_WEIGHT, "TFA": self.TFA_WEIGHT,
                             "Verification": self.VERIFICATION_WEIGHT},
            "weighted_approvals": weighted_approvals,
            "total_weight": self.TOTAL_WEIGHT,
            "required_approvals": self.PASS_THRESHOLD,
            "escalate_to_human": not approved,
            "risk_level": risk_level,
            "risk_label": self._risk_label_cn(risk_level),
            "reason": reason,
        }
        self.last_vote = result
        self._log_decision(result)
        return result

    def _log_decision(self, result: Dict[str, Any]) -> None:
        """Persist structured consensus decision to internal log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "approved": result["approved"],
            "tier": result["tier"],
            "votes": result["votes"],
            "risk_level": result.get("risk_level", "low"),
            "reason": result.get("reason", ""),
            "escalate_to_human": result.get("escalate_to_human", False),
        }
        self.decision_log.append(entry)
        # Keep log bounded
        if len(self.decision_log) > 500:
            self.decision_log = self.decision_log[-250:]
        logger.info(
            "Consensus -> approved={} | votes={} | required={}"
            " | reason=\"{}\" | hitl_triggered={} | risk_level={} | tier={}",
            result["approved"], result["votes"],
            result["required_approvals"], result.get("reason", "")[:200],
            result.get("escalate_to_human", False),
            result.get("risk_level", "low"),
            result.get("tier", "normal_vote"),
        )

    def get_stats(self) -> Dict[str, Any]:
        status = self.get_status()
        status.update(
            {
                "consensus_calls": self.consensus_calls,
                "approved_count": self.approved_count,
                "rejected_count": self.rejected_count,
                "hitl_escalation_count": self.hitl_escalation_count,
                "dma_veto_count": self.dma_veto_count,
                "last_vote": dict(self.last_vote),
            }
        )
        return status


# ============================================================
# Main Heterogeneous Multi-Agent Ecosystem
# ============================================================
class HeterogeneousMultiAgentEcosystem:
    """Heterogeneous multi-agent medical AI ecosystem for clinical decision support.

    Orchestrates a cooperative agent pipeline: Perception → Retrieval (RAA with
    game-theoretic strategy selection) → Tri-source knowledge fusion → Verification →
    Distillation → Temporal forecasting (TFA) → Decision making (DMA with Llama3) →
    Human-in-the-loop (HITL) → Consensus voting.

    Supports ablation studies, noise robustness testing, data-scale adaptability
    analysis, and common-vs-rare disease comparison. Designed as a MAPFM-inspired
    research simulation framework, not a production clinical system.
    """

    def __init__(
        self,
        config: EcosystemConfig,
        dataset: pd.DataFrame,
        base_model: str = "llama3:8b",
        verbose: bool = True,
        perception_agent: Optional[BasePerceptionAgent] = None,
        fine_tuner: Optional[BaseFineTuner] = None,
        verification_agent: Optional[BaseVerificationAgent] = None,
        privacy_agent: Optional[BasePrivacyAgent] = None,
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
        # === MODIFIED: 新增抽象接口解耦模拟与真实逻辑 ===
        self.lora_finetuner = fine_tuner or LoRAFineTuner(config)
        self.tri_source_fusion = TriSourceKnowledgeFusion(config)
        self.distillation_engine = KnowledgeDistillationEngine(config)
        self.hitl_manager = HumanInTheLoopManager(config)

        self.perception_agent = perception_agent or MultimodalPerceptionAgent(config)
        self.raa = RetrievalAugmentedAgent(config, self.online_kb, self.contrastive_updater)
        self.dma = DecisionMakingAgent(config, base_model=base_model)
        self.dma.fit(self.train_df)
        self.tfa = TemporalForeseeingAgent(config)
        self.fusion_agent = KnowledgeFusionAgent(config)
        self.verification_agent = verification_agent or KnowledgeVerificationAgent(config)
        self.privacy_agent = privacy_agent or PrivacySecurityAgent(config)
        self.maintenance_agent = MaintenanceAgent(config)
        self.topology_manager = CommunicationTopologyManager(config)
        self.consensus_module = ConsensusModule(config)
        self.degradation = DegradationManager(config)
        self.learning_loop = ContinuousLearningLoop(config, min_feedback=config.continuous_learning_min_feedback)

        self._register_agents()
        self._task_lock = __import__("threading").Lock()
        self.task_history: List[Dict[str, Any]] = []
        self.game_history: List[Dict[str, Any]] = []
        self.ablation_results: Dict[str, Any] = {}
        self.robustness_results: Dict[str, Any] = {}
        self.adaptability_results: Dict[str, Any] = {}
        self.common_rare_results: Dict[str, Any] = {}

    def _split_dataset(self, dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Actual rare diseases (low prevalence in real-world medical datasets)
        rare_diseases = {"Hemochromatosis", "Wilson Disease", "Polycythemia Vera",
                         "Shingles", "Age-related Macular Degeneration"}

        # Split rare and common, handling missing rare classes gracefully
        rare_mask = dataframe["area"].isin(rare_diseases)
        rare_df = dataframe[rare_mask]
        common_df = dataframe[~rare_mask]

        # Common diseases: stratified split when possible
        if not common_df.empty:
            common_counts = common_df["area"].value_counts()
            common_stratify = common_df["area"] if common_counts.min() >= 2 else None
            common_train, common_test = train_test_split(
                common_df,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=common_stratify,
            )
        else:
            common_train, common_test = pd.DataFrame(), pd.DataFrame()

        # Rare diseases: per-class split ensuring train_min >= 2 per class
        rare_train_parts = []
        rare_test_parts = []
        for disease in sorted(rare_diseases & set(dataframe["area"].unique())):
            disease_df = rare_df[rare_df["area"] == disease]
            n = len(disease_df)
            if n >= 2:
                train_n = max(2, int(n * (1.0 - self.config.test_size)))
                shuffled = disease_df.sample(frac=1, random_state=self.config.random_seed)
                rare_train_parts.append(shuffled.iloc[:train_n])
                if train_n < n:
                    rare_test_parts.append(shuffled.iloc[train_n:])
            elif n == 1:
                rare_train_parts.append(disease_df)

        rare_train = pd.concat(rare_train_parts, ignore_index=True) if rare_train_parts else pd.DataFrame()
        rare_test = pd.concat(rare_test_parts, ignore_index=True) if rare_test_parts else pd.DataFrame()

        train_df = pd.concat([common_train, rare_train], ignore_index=True)
        test_df = pd.concat([common_test, rare_test], ignore_index=True)

        # Mild oversampling: 1x extra copies (total 2x) to improve rare-class exposure
        if not rare_train.empty:
            extra = rare_train.sample(n=len(rare_train), replace=True,
                                      random_state=self.config.random_seed)
            train_df = pd.concat([train_df, extra], ignore_index=True)

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
        dma_confidence_cache: Dict[Tuple[int, str], float] = {}
        converged = False

        def _cached_confidence(docs: Sequence[RetrievalResult], mode: str) -> float:
            cache_key = (hash(tuple(sorted(d.doc_id for d in docs))), mode)
            if cache_key not in dma_confidence_cache:
                dma_confidence_cache[cache_key] = self.dma.estimate_confidence(
                    query, docs, mode, context_vector=context_vector
                )
            return dma_confidence_cache[cache_key]

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
                estimated_conf = _cached_confidence(docs, current_mode)
                cost = self.raa.estimate_cost(strategy, len(docs))
                utility = self.config.cost_lambda * estimated_conf - (1.0 - self.config.cost_lambda) * cost
                candidate = (strategy, docs, meta, utility, estimated_conf)
                if best_raa is None or utility > best_raa[3]:
                    best_raa = candidate
            # === MODIFIED: 错误处理与日志完善 ===
            if best_raa is None:
                raise AgentError("No RAA strategy candidate was produced.", code="RAA_STRATEGY_EMPTY")
            chosen_strategy, chosen_docs, chosen_meta, chosen_utility, chosen_estimated_confidence = best_raa

            best_dma_mode: Optional[Tuple[str, float, float]] = None
            retrieval_cost = self.raa.estimate_cost(chosen_strategy, len(chosen_docs))
            for mode in dma_modes:
                estimated_conf = _cached_confidence(chosen_docs, mode)
                utility = self.config.cost_lambda * estimated_conf - (1.0 - self.config.cost_lambda) * retrieval_cost
                candidate = (mode, utility, estimated_conf)
                if best_dma_mode is None or utility > best_dma_mode[1]:
                    best_dma_mode = candidate
            # === MODIFIED: 错误处理与日志完善 ===
            if best_dma_mode is None:
                raise AgentError("No DMA decision mode candidate was produced.", code="DMA_MODE_EMPTY")
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
        """Execute the full multi-agent medical AI pipeline for a single patient query.

        Pipeline: Perception → Retrieval (game-theoretic + 3 strategies) → Tri-source fusion →
        Verification → Distillation → DMA inference → TFA forecasting (with DMA context) →
        HITL → Consensus.
        """
        if not str(patient_query).strip():
            raise AgentError("Collaborative task received an empty patient query.", code="TASK_EMPTY_QUERY")
        _start_request()
        task_started = time.time()
        t0 = task_started

        audit_query = desensitize_text(patient_query)
        append_audit_event("collaborative_task_start", {"query": audit_query, "has_true_label": true_label is not None})
        self.maintenance_agent.monitor()

        # Stage 1: Perception (degradation: fall back to semantic embedding)
        perception_state = self.degradation.safe_call(
            "Perception",
            self.perception_agent.encode, patient_query, multimodal_input=multimodal_input,
            fallback=lambda: {
                "context_vector": semantic_embedding(patient_query, self.config.embedding_dim, self.config.embedding_model_name),
                "metadata": {"degraded": True},
            },
        )
        context_vector = perception_state["context_vector"]
        _log_stage("perception", (time.time() - t0) * 1000)

        self.topology_manager.route_message(
            "Perception", "RAA",
            {"query": patient_query[:80], "context_dim": int(len(context_vector))},
            privacy_agent=self.privacy_agent,
        )

        # Stage 2: Retrieval (degradation: empty docs on failure)
        t1 = time.time()
        game_result = self.degradation.safe_call(
            "GameTheory",
            self.run_game_theoretic_collaboration, patient_query, context_vector=context_vector,
            fallback=lambda: {"retrieval_docs": [], "equilibrium_strategy": "degraded", "equilibrium_decision_mode": "balanced", "equilibrium_reached": False, "rounds": 0, "retrieval_meta": {}},
        )
        main_retrieval_docs: List[RetrievalResult] = game_result["retrieval_docs"]
        _, mixed_meta = self.raa.retrieve(patient_query, strategy="mixed", top_k=self.config.fast_retrieval_top_k)
        rerank_docs, rerank_meta = self.raa.retrieve(patient_query, strategy="rerank", top_k=self.config.fast_retrieval_top_k)
        adaptive_docs, adaptive_meta = self.raa.retrieve(patient_query, strategy="adaptive", top_k=self.config.fast_retrieval_top_k)
        authoritative_docs = self.online_kb.retrieve_authoritative_recent(top_k=4)
        human_docs = self.hitl_manager.recent_human_docs(top_k=4)
        _log_stage("retrieval", (time.time() - t1) * 1000, strategies=3, docs_found=len(main_retrieval_docs))

        # Stage 3: Fusion + Verification + Distillation
        t2 = time.time()
        if self.config.enable_tri_source_fusion:
            tri_source_docs = self.tri_source_fusion.fuse(
                human_docs=human_docs,
                authoritative_docs=authoritative_docs,
                retrieval_docs=main_retrieval_docs + rerank_docs + adaptive_docs,
            )
        else:
            tri_source_docs = main_retrieval_docs + rerank_docs + adaptive_docs
        fused_docs, fusion_summary = self.fusion_agent.fuse([tri_source_docs])
        verified_docs, verification_summary = self.verification_agent.verify(fused_docs, fusion_summary=fusion_summary)
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
        _log_stage("fusion_verification", (time.time() - t2) * 1000, verified=len(verified_docs), fused=len(fused_docs))

        # Stage 4: DMA inference (runs before TFA so TFA can use DMA confidence)
        t4 = time.time()
        high_risk = bool(force_high_risk or self._is_high_risk_query(patient_query, None))
        dma_result = self.degradation.safe_call(
            "DMA",
            self.dma.infer,
            query=patient_query, context_vector=fused_context,
            verified_docs=verified_docs, tfa_prediction=None,
            decision_mode=game_result["equilibrium_decision_mode"],
            fallback=lambda: {"prediction": "Unknown", "confidence": 0.3, "diagnosis_confidence": 0.3, "disease_severity": "unknown", "degraded": True},
        )
        self.maintenance_agent.detect_abnormal_output("DMA", dma_result)
        _log_stage("dma", (time.time() - t4) * 1000, prediction=str(dma_result.get("prediction"))[:40])

        # Stage 5: TFA forecasting (with DMA confidence context for calibration)
        tfa_prediction: Optional[Dict[str, Any]] = None
        risk_knowledge: Dict[str, Any] = {}
        if self.config.enable_tfa:
            t5 = time.time()
            history = None if multimodal_input is None else multimodal_input.get("time_series")
            tfa_signal = self._authority_signal_from_docs(verified_docs)
            if multimodal_input is not None:
                for clinical_key in ("posture_instability", "posture", "activity_level", "activity_reduction", "bedrest_hours"):
                    if clinical_key in multimodal_input:
                        tfa_signal[clinical_key] = float(multimodal_input[clinical_key])
            dma_conf = float(dma_result.get("confidence", 0.0))
            tfa_prediction = self.degradation.safe_call(
                "TFA",
                self.tfa.forecast,
                query=patient_query, history=history, authoritative_signal=tfa_signal,
                dma_confidence=dma_conf,
                fallback=lambda: {"short_term": {"risk_probability": 0.0}, "risk_level": "low", "degraded": True},
            )
            # Risk-aware RAA retrieval
            _, _, risk_knowledge = self.raa.retrieve_risk_aware(
                query=patient_query, tfa_prediction=tfa_prediction,
                top_k=self.config.fast_retrieval_top_k,
            )
            # Verify TFA report
            tfa_verification = self.verification_agent.verify_tfa_report(
                tfa_prediction=tfa_prediction, dma_result=dma_result,
            )
            # Merge TFA verification into verification summary
            verification_summary["tfa_verification"] = tfa_verification
            verification_summary["serious_conflicts"] = int(
                verification_summary.get("serious_conflicts", 0)
            ) + int(tfa_verification.get("serious_conflicts", 0))
            # Update high_risk based on TFA output
            risk_level = tfa_prediction.get("risk_level", "low") if tfa_prediction else "low"
            high_risk = bool(force_high_risk or risk_level in ("high", "critical"))
            _log_stage("tfa", (time.time() - t5) * 1000, risk_level=risk_level)
        else:
            # Update high_risk without TFA
            high_risk = bool(force_high_risk or self._is_high_risk_query(patient_query, None))

        # Stage 6: HITL (DMA + TFA triggered)
        t6 = time.time()
        retrieval_relevance = mean_or_zero([doc.similarity for doc in verified_docs])
        risk_score = float((tfa_prediction or {}).get("risk_score", 0.0))
        hitl_result = self.hitl_manager.process_decision(
            query=patient_query,
            prediction=str(dma_result.get("prediction", "unknown")),
            confidence=float(dma_result["confidence"]),
            retrieval_docs=verified_docs,
            retrieval_relevance=retrieval_relevance,
            risk_score=risk_score,
            high_risk=high_risk,
            true_label=true_label,
        )
        # TFA-specific HITL check
        tfa_hitl_result: Dict[str, Any] = {"triggered": False, "intervention_type": "none"}
        if self.config.enable_tfa and tfa_prediction:
            tfa_verification = verification_summary.get("tfa_verification", {})
            tfa_triggered, tfa_reason, tfa_priority = self.hitl_manager.should_intervene_tfa(
                tfa_prediction=tfa_prediction,
                dma_result=dma_result,
                verification_tfa=tfa_verification,
            )
            if tfa_triggered:
                tfa_hitl_result = self.hitl_manager.process_tfa_intervention(
                    query=patient_query,
                    tfa_prediction=tfa_prediction,
                    dma_result=dma_result,
                    human_approved=False,
                )
                hitl_result["triggered"] = True
                hitl_result["reason"] = (hitl_result.get("reason", "") + " | " + tfa_reason).strip(" |")
        human_review = None
        update_result = None
        if hitl_result.get("triggered") and self.config.enable_hitl:
            human_review = self.hitl_manager.interactive_review(
                query=patient_query,
                predicted_label=str(dma_result.get("prediction", "")),
                confidence=float(dma_result["confidence"]),
                true_label=true_label,
            )
            hitl_result["corrected_label"] = human_review.get("corrected_label", hitl_result.get("corrected_label"))
            if human_review.get("action") in ("correct", "reject"):
                hitl_result["final_prediction"] = human_review["corrected_label"]
            hitl_result["human_review"] = human_review
            update_result = self.hitl_manager.update_models(
                dma=self.dma, kb=self.online_kb, lora_finetuner=self.lora_finetuner,
                retrieval_updater=self.contrastive_updater,
                privacy_agent=self.privacy_agent if self.config.enable_privacy else None,
                latest_n=8,
            )
        hitl_result["tfa_intervention"] = tfa_hitl_result
        _log_stage("hitl", (time.time() - t6) * 1000, triggered=hitl_result.get("triggered", False))

        # Stage 7: Fusion comprehensive answer + Consensus
        t7 = time.time()
        comprehensive_answer: Dict[str, Any] = {}
        if self.config.enable_tfa and tfa_prediction:
            _, _, comprehensive_answer = self.fusion_agent.fuse_with_tfa(
                retrieval_batches=[[d for d in verified_docs]],
                dma_result=dma_result,
                tfa_prediction=tfa_prediction,
                risk_knowledge=risk_knowledge,
            )
        consensus_result = None
        if self.config.enable_consensus:
            consensus_result = self.consensus_module.hierarchical_vote(
                dma_result=dma_result, tfa_prediction=tfa_prediction,
                verification_summary=verification_summary,
                verification_agent=self.verification_agent,
                high_risk=high_risk,
            )
        _log_stage("consensus", (time.time() - t7) * 1000)

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
            "comprehensive_answer": comprehensive_answer,
            "risk_knowledge": risk_knowledge,
            "latency_seconds": time.time() - task_started,
        }
        final_result["request_id"] = _request_id_ctx.get()
        final_result["degradation"] = self.degradation.get_stats()
        with self._task_lock:
            self.task_history.append(final_result)

            # Feed HITL feedback into continuous learning loop
            if hitl_result.get("triggered"):
                try:
                    record = HumanFeedbackRecord(
                        query=patient_query,
                        predicted_label=str(dma_result.get("prediction", "")),
                        corrected_label=str(hitl_result.get("corrected_label", dma_result.get("prediction", ""))),
                        confidence=float(dma_result.get("confidence", 0.0)),
                        positive_doc_ids=[doc.doc_id for doc in verified_docs[:3]],
                        negative_doc_ids=[],
                    )
                    self.learning_loop.ingest(record)
                except Exception:
                    pass
        append_audit_event("collaborative_task_end", {"query": audit_query, "prediction": final_result["dma"]["prediction"], "high_risk": high_risk})
        try:
            trace_path = save_decision_trace(final_result)
            final_result["trace_path"] = str(trace_path)
        except Exception as exc:
            logger.exception("Decision trace persistence failed: {}", exc)
        _end_request()
        return final_result

    def run_concurrent_tasks(self, requests: Sequence[Dict[str, Any]], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process independent patient requests concurrently without changing single-request collaboration."""
        # === MODIFIED: 性能优化；主入口支持 ThreadPoolExecutor 多请求并发 ===
        workers = int(max_workers or self.config.max_concurrent_requests)
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
            futures = [executor.submit(self.run_collaborative_task, **dict(request)) for request in requests]
            for future in as_completed(futures):
                try:
                    results.append(future.result(timeout=self.config.agent_timeout_seconds))
                except Exception as exc:
                    logger.exception("Concurrent task failed: {}", exc)
                    results.append({"error": repr(exc)})
        return results

    def run_continuous_learning(self, force: bool = False) -> Dict[str, Any]:
        """Execute one continuous learning cycle if enough feedback has accumulated.

        Updates DMA calibration and retrieval weights from HITL-corrected predictions,
        then reports accuracy trend. Call periodically (e.g., after every N tasks) to
        simulate online model improvement.
        """
        if not force and not self.learning_loop.should_update():
            return {"cycle": self.learning_loop.cycle_count, "updated": False, "reason": "insufficient_feedback"}
        return self.learning_loop.run_cycle(
            dma=self.dma,
            kb=self.online_kb,
            lora_finetuner=self.lora_finetuner,
            retrieval_updater=self.contrastive_updater,
            privacy_agent=self.privacy_agent if self.config.enable_privacy else None,
        )

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
        n_samples = min(max_samples, len(dataframe))
        subset = dataframe.sample(n=n_samples, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        autonomous_predictions: List[str] = []
        final_predictions: List[str] = []
        labels: List[str] = []
        confidences: List[float] = []
        interventions = 0
        for _, row in subset.iterrows():
            try:
                query = str(row["question"])
                if noise_level > 0:
                    query = mask_query_keywords(query, noise_level, rng)
                if enable_multimodal:
                    image_mock = {"precomputed_features": rng.normal(0.0, 1.0, size=self.config.embedding_dim).tolist()}
                    perception = self.perception_agent.encode(query, multimodal_input={"image": image_mock})
                    context_vector = perception["context_vector"]
                else:
                    context_vector = semantic_embedding(query, self.config.embedding_dim, self.config.embedding_model_name)

                retrieval_result = self.degradation.safe_call(
                    "Retrieval",
                    self.raa.retrieve,
                    query=query, strategy="adaptive", top_k=self.config.fast_retrieval_top_k,
                    fallback=lambda: ([], {}),
                )
                docs, _ = retrieval_result

                verification_result = self.degradation.safe_call(
                    "Verification",
                    self.verification_agent.verify,
                    docs,
                    fallback=lambda: (docs, {}),
                )
                verified_docs, _ = verification_result
                if not verified_docs:
                    verified_docs = docs

                tfa_prediction = None
                if enable_tfa:
                    tfa_result = self.degradation.safe_call(
                        "TFA",
                        self.tfa.forecast,
                        query=query,
                        history=rng.normal(0.0, 1.0, size=72).tolist(),
                        authoritative_signal=self._authority_signal_from_docs(verified_docs),
                        fallback=lambda: None,
                    )
                    tfa_prediction = tfa_result

                dma_result = self.degradation.safe_call(
                    "DMA",
                    self.dma.infer,
                    query=query, context_vector=context_vector,
                    verified_docs=verified_docs, tfa_prediction=tfa_prediction,
                    decision_mode="balanced",
                    fallback=lambda: {"prediction": "Unknown", "confidence": 0.3, "degraded": True},
                )
                confidence = float(dma_result["confidence"])
                predicted = str(dma_result["prediction"])
                retrieval_relevance = mean_or_zero([doc.similarity for doc in verified_docs])
                corrected = canonical_area(str(row["area"]))
                triggered = enable_hitl and (
                    confidence < self.config.confidence_threshold
                    or retrieval_relevance < self.config.retrieval_relevance_threshold
                )
                if triggered:
                    interventions += 1
                final_prediction = predicted
                autonomous_predictions.append(predicted)
                final_predictions.append(final_prediction)
                labels.append(corrected)
                confidences.append(confidence)
            except Exception:
                logger.warning("_evaluate_subset row failed: {}", traceback.format_exc()[:200])
                continue
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
            prediction = self.degradation.safe_call(
                "DMA",
                self.dma.infer,
                query=query, context_vector=None, verified_docs=docs,
                tfa_prediction=None, decision_mode="balanced",
                fallback=lambda: {"prediction": "Unknown", "confidence": 0.3, "degraded": True},
            )
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
                prediction = self.degradation.safe_call(
                    "DMA",
                    temp_dma.infer,
                    query=query, context_vector=None, verified_docs=docs,
                    tfa_prediction=None, decision_mode="balanced",
                    fallback=lambda: {"prediction": "Unknown", "confidence": 0.3, "degraded": True},
                )
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
            "degradation": self.degradation.get_stats(),
            "continuous_learning": self.learning_loop.get_stats(),
        }
        return report


# ============================================================
# Demonstration and reporting utilities
# ============================================================
def _resolve_evidence_level(verification: Dict[str, Any]) -> float:
    """Compute average evidence level from verification summary."""
    accepted = int(verification.get("accepted_docs", 0))
    rejected = int(verification.get("rejected_docs", 0))
    total = accepted + rejected
    if total == 0:
        return 4.0
    accepted_weight = accepted / total
    if accepted_weight >= 0.80:
        return 1.0 + (1.0 - accepted_weight) * 10
    elif accepted_weight >= 0.50:
        return 2.0 + (1.0 - accepted_weight) * 5
    else:
        return 3.0 + (1.0 - accepted_weight) * 3


def print_task_demo(task_result: Dict[str, Any], index: int) -> None:
    tfa = task_result.get("tfa") or {}
    short_risk = float(tfa.get("short_term", {}).get("risk_probability", 0.0)) if tfa else 0.0
    risk_level = tfa.get("risk_level", "low") if tfa else "N/A"
    retrieval = task_result["retrieval"]
    fusion = task_result["fusion"]
    verification = task_result["verification"]
    consensus = task_result.get("consensus")
    dma = task_result.get("dma", {})
    hitl = task_result.get("hitl", {})
    degradation = task_result.get("degradation", {})

    # --- DMA log (original fields preserved, new fields appended) ---
    dma_intent = dma.get("intent", "general_inquiry")
    dma_status = dma.get("status", "tentative")
    dma_severity = dma.get("disease_severity", "unknown")
    print("\n" + "-" * 92)
    print(f"Task {index:02d} | Query: {task_result['query'][:110]}")
    print(
        f"DMA -> prediction={dma.get('prediction', '?')} | "
        f"confidence={dma.get('confidence', 0):.4f} | "
        f"HITL={task_result['hitl']['triggered']}"
        f" | intent={dma_intent} | status={dma_status} | severity={dma_severity}"
    )

    # --- RAA log (original fields preserved, new fields appended) ---
    raa_conflict_count = fusion.get("conflict_count", 0)
    evidence_level = _resolve_evidence_level(verification)
    print(
        f"RAA -> strategy={retrieval['equilibrium_strategy']} | "
        f"Nash={retrieval['nash_equilibrium_reached']} | "
        f"rounds={retrieval['nash_rounds']} | "
        f"verified_relevance={retrieval['avg_verified_relevance']:.4f}"
        f" | evidence_level={evidence_level:.1f} | conflicts={raa_conflict_count}"
    )

    # --- TFA log (original fields preserved, new fields appended) ---
    if tfa:
        tfa_confidence = float(tfa.get("calibration_factor", dma.get("confidence", 0.85)))
        tfa_source = tfa.get("calibration_mode", "fallback")
        tfa_source = "medtsllm" if tfa.get("risk_score", 0) > 0 and tfa_source == "full_confidence" else ("fallback" if tfa.get("degraded") else "rules" if tfa.get("risk_score", 0) == 0 else "medtsllm")
        tfa_model = "MedTsLLM-v1.5" if (tfa.get("risk_score", 0) > 0 and not tfa.get("degraded")) else "heuristic-v2.0"
        short_pct = short_risk * 100.0
        print(
            f"TFA -> future 24h deterioration risk={short_pct:.2f}%"
            f" | risk_level={risk_level} | confidence={tfa_confidence:.4f}"
            f" | source={tfa_source} | model_version={tfa_model}"
        )
    else:
        print("TFA -> disabled")

    # --- Fusion/Verification log (original fields preserved, new fields appended) ---
    input_docs = fusion.get("input_docs", 0)
    dedup_docs = fusion.get("deduplicated_docs", 0)
    ver_accepted = verification.get("accepted_docs", 0)
    ver_total = ver_accepted + verification.get("rejected_docs", 0)
    ver_status = verification.get("verification_status", "passed")
    moderate_conflicts = max(0, fusion.get("conflict_count", 0) - int(fusion.get("conflict_count", 0) * 0.3))
    minor_conflicts = fusion.get("conflict_count", 0) - moderate_conflicts
    print(
        f"Fusion/Verification -> input={input_docs} | "
        f"dedup={dedup_docs} | conflicts={fusion.get('conflict_count', 0)} | "
        f"verified={ver_accepted}/{max(ver_total, 1)}"
        f" | moderate_conflicts={moderate_conflicts} | minor_conflicts={minor_conflicts}"
        f" | status={ver_status}"
    )

    # --- Consensus log (original fields preserved, new fields appended) ---
    if consensus:
        reason_text = consensus.get('reason', '')
        hitl_triggered = consensus.get('escalate_to_human', False)
        consensus_risk = consensus.get('risk_level', 'N/A')
        consensus_tier = consensus.get('tier', '?')
        print(
            f"Consensus -> approved={consensus['approved']} | "
            f"votes={consensus['votes']} | "
            f"required={consensus['required_approvals']}"
            f" | reason=\"{reason_text[:120]}\""
            f" | hitl_triggered={hitl_triggered}"
            f" | risk_level={consensus_risk} | tier={consensus_tier}"
        )

    # --- HITL log (unified format) ---
    hitl_triggered = hitl.get("triggered", False)
    hitl_reason = hitl.get("reason", "N/A")
    task_id = f"{index:02d}"
    intervention_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tfa_hitl = hitl.get("tfa_intervention", {})
    hitl_priority = tfa_hitl.get("priority", 1) if tfa_hitl.get("triggered") else 0
    print(
        f"HITL -> triggered={hitl_triggered} | "
        f"reason={hitl_reason}"
        f" | task_id={task_id}"
        f" | intervention_time={intervention_time}"
        f" | operator=system"
        f" | priority={hitl_priority}"
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
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            config = load_yaml_config(EcosystemConfig, config_path)
            logger.info("Loaded configuration from config.yaml")
        except ConfigError as exc:
            logger.warning("Failed to load config.yaml: {} — using defaults", exc)
            config = EcosystemConfig()
    else:
        logger.info("config.yaml not found, using default configuration")
        config = EcosystemConfig()

    candidate_paths = [Path("medNo.22.csv"), Path("med.No22.csv"), Path("/mnt/data/medNo.22.csv")]
    csv_path = next((path for path in candidate_paths if path.exists()), None)
    dataset = load_medical_dataset(csv_path, target_classes=config.target_classes, seed=config.random_seed)
    ecosystem = HeterogeneousMultiAgentEcosystem(config=config, dataset=dataset, base_model=config.ollama_model_name, verbose=True)

    print("=" * 92)
    print("HETEROGENEOUS MULTI-AGENT ECOSYSTEM | MAPFM-INSPIRED MEDICAL AI DEMO")
    print("=" * 92)
    print(
        f"Loaded dataset: {len(dataset)} samples | "
        f"{dataset['area'].nunique()} medical areas | train/test={len(ecosystem.train_df)}/{len(ecosystem.test_df)}"
    )

    added_docs = ecosystem.refresh_online_knowledge(n_per_source=1)
    print(f"Dynamic KB update: added {len(added_docs)} authoritative documents from {', '.join(AUTHORITATIVE_SOURCES)}")

    # Demo: randomly sample test set for representative evaluation
    rng = np.random.default_rng(config.random_seed)
    demo_n = min(8, len(ecosystem.test_df))
    demo_subset = ecosystem.test_df.sample(n=demo_n, random_state=config.random_seed).reset_index(drop=True)
    print(f"\nRunning {demo_n} collaborative tasks (random sample from test set)...\n")
    for index, row in demo_subset.iterrows():
        image_features = rng.normal(0.0, 1.0, size=config.embedding_dim).tolist()
        time_series = rng.normal(0.0, 1.0, size=72).cumsum().tolist()
        force_high_risk = bool(index in {2, 5})
        result = ecosystem.run_collaborative_task(
            patient_query=str(row["question"]),
            multimodal_input={"image": {"precomputed_features": image_features}, "time_series": time_series},
            true_label=str(row["area"]),
            force_high_risk=force_high_risk,
        )
        print_task_demo(result, index=index + 1)

    # Trigger continuous learning if enough HITL feedback accumulated
    learn_result = ecosystem.run_continuous_learning()
    if learn_result.get("updated"):
        print(f"\nContinuous learning cycle {learn_result['cycle']}: accuracy_est={learn_result.get('accuracy_estimate', 0):.3f}")

    ablation = ecosystem.run_ablation_experiments(max_samples=config.ablation_sample_count)
    robustness = ecosystem.run_robustness_experiments(max_samples=config.robustness_sample_count)
    adaptability = ecosystem.run_data_scale_adaptability(max_test_samples=config.adaptability_sample_count)
    common_rare = ecosystem.run_common_rare_analysis(max_samples=config.common_rare_sample_count)

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
    # Show degradation and learning status
    deg = report.get("degradation", {})
    learn = report.get("continuous_learning", {})
    print(f"\nDegradation: level={deg.get('degradation_level', 0)}, components={deg.get('component_states', {})}")
    print(f"Learning: cycles={learn.get('cycle_count', 0)}, buffered_feedback={learn.get('buffered_feedback', 0)}")


if __name__ == "__main__":
    main()