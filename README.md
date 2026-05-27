# MAPFM — Heterogeneous Multi-Agent Medical AI Ecosystem

MAPFM (Multi-Agent Perception-Fusion Medical) 是一个**异构多智能体医疗 AI 临床决策支持系统**，模拟从患者问诊输入到诊断输出、风险评估、人工审核、共识投票的完整临床辅助决策流程。

## Architecture

```
Patient Query
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1  Perception Agent                                     │
│           Text encoder + Image ViT patching → Cross-Attention  │
│           Fusion → Context Vector                              │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 2  RAA (Retrieval Augmented Agent)                      │
│           ┌─ Mixed-RAG:   TF-IDF + MiniLM/FAISS fusion        │
│           ├─ Rerank-RAG:  + similarity + recency + authority   │
│           └─ Adaptive-RAG: uncertainty-gated deep/shallow      │
│           Nash Game-Theoretic strategy selection with DMA      │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 3  Tri-Source Fusion → Knowledge Fusion → Verification  │
│           Human(0.95) + Authoritative(0.90) + Retrieval(0.75) │
│           SimHash dedup + conflict detection + fact-checking   │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 4  DMA (Decision Making Agent)                          │
│           Primary:  Ollama Llama3:8b (local LLM)              │
│           Fallback: TF-IDF centroid + retrieval + lexical     │
│           Platt scaling calibration + disease disambiguation   │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 5  TFA (Temporal Foreseeing Agent)                      │
│           MedTsLLM-v1.5 / Heuristic TCN+LSTM+Transformer      │
│           Short(24h) / Mid(30d) / Long(12m) risk windows      │
│           DMA-confidence-aware calibration                    │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 6  HITL (Human-in-the-Loop)                             │
│           DMA low confidence / TFA critical risk / conflicts  │
│           Interactive review → Model & retrieval updates      │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 7  Hierarchical Consensus + Comprehensive Answer        │
│           FORCE_HITL → DMA_VETO → NORMAL_VOTE (3-layer)      │
│           DMA(2票) + TFA(1票) + Verification(1票)             │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
                 Final Report
```

## Agents

| Agent | Role | Key Capability |
|---|---|---|
| **Perception** | 多模态感知编码 | Text + Image cross-attention fusion, ViT-style patching |
| **RAA** | 检索增强 | 3 RAG strategies, FAISS dense index, game-theoretic selection |
| **DMA** | 诊断决策 | Ollama Llama3:8b primary + legacy fallback, Platt calibration |
| **TFA** | 时序风险预测 | MedTsLLM / TCN+LSTM+Transformer, multi-window risk forecasting |
| **Fusion** | 知识融合去重 | SimHash dedup, conflict detection, area consensus |
| **Verification** | 事实核查 | Fact scoring, timeliness check, TFA report validation |
| **Consensus** | 层次化共识 | 3-layer voting: FORCE_HITL → DMA_VETO → NORMAL_VOTE |
| **Privacy** | 隐私安全 | XOR encryption, differential privacy gradients |
| **Maintenance** | 健康监控 | Heartbeat, fault isolation, abnormal output detection |
| **Topology** | 通信拓扑 | Star/Mesh routing, encrypted messaging, retry queue |

## Integrated Modules

| Module | Function |
|---|---|
| `OnlineKnowledgeBaseManager` | Dynamic KB with TF-IDF + FAISS dual index, authoritative crawl simulation |
| `TriSourceKnowledgeFusion` | Weighted fusion of human/authoritative/retrieval knowledge sources |
| `KnowledgeDistillationEngine` | Compress verified docs into context vectors + distilled text |
| `HumanInTheLoopManager` | 3-level HITL: autonomous → intervention → parameter update |
| `ContrastiveRetrievalUpdater` | Positive/negative sample contrastive feedback for retrieval weights |
| `ContinuousLearningLoop` | Online learning triggered at ≥10 feedback records |
| `DegradationManager` | 4-level graceful degradation with automatic `safe_call` fallback |

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) (optional, for LLM inference)

### Setup

```bash
# Clone and enter the project
cd MAPFM_medical_AI_repaired_complete

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull the LLM model (optional — legacy fallback works without it)
ollama pull llama3:8b
```

### Run Demo

```bash
python mapfm_ecosystem_repaired.py
```

This runs 8 collaborative tasks on random test samples, prints per-task diagnostics (DMA, RAA, TFA, Fusion, Consensus, HITL), then executes ablation, robustness, data-scale adaptability, and common-vs-rare disease experiments.

### Run FastAPI Server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Run Tests

```bash
pytest -q
```

## Configuration

All hard-coded parameters are externalized into `SystemConfig` and `EcosystemConfig` dataclasses. Override via `config.yaml`:

```yaml
random_seed: 42
confidence_threshold: 0.62
ollama_model_name: "llama3:8b"
ollama_timeout_seconds: 10.0
enable_tfa: true
enable_hitl: true
enable_consensus: true
consensus_threshold: 0.667
nash_max_rounds: 6
# ... 60+ configurable parameters
```

Key configuration groups:
- **Model**: `ollama_model_name`, `ollama_temperature`, `ollama_num_ctx`
- **Thresholds**: `confidence_threshold`, `retrieval_relevance_threshold`, `high_risk_tfa_threshold`
- **Weights**: `dma_centroid_weight`, `dma_retrieval_weight`, `dma_lexical_weight`, `dma_tfa_weight`
- **Feature flags**: `enable_tfa`, `enable_hitl`, `enable_consensus`, `enable_multimodal`, `enable_privacy`

## Experiments

The system includes four built-in experiment suites:

| Experiment | Description |
|---|---|
| **Ablation** | Full ecosystem vs. no-HITL / no-TFA / no-Multimodal |
| **Noise Robustness** | Performance at 0% / 10% / 30% keyword masking noise |
| **Data-Scale Adaptability** | Accuracy vs. training data fraction (20%–100%) |
| **Common vs. Rare** | F1 disparity between high-prevalence and low-prevalence diseases |

## File Structure

```
├── mapfm_ecosystem_repaired.py   # Main ecosystem (~4650 lines)
├── utils.py                      # Embedding, SimHash, tokenization, etc.
├── exceptions.py                 # MAPFMException hierarchy (6 subclasses)
├── logging_config.py             # Structured Loguru logging
├── config_loader.py              # YAML config with hot-reload
├── desensitizer.py               # PII masking
├── audit_logger.py               # JSONL audit trail
├── traceability.py               # Decision trace persistence
├── redis_bus.py                  # Optional Redis message queue
├── api.py                        # FastAPI REST endpoints
├── config.yaml                   # Deployable configuration defaults
├── requirements.txt              # Python dependencies
├── tests/
│   ├── test_dma.py               # DMA fallback contract test
│   ├── test_retrieval.py         # Retrieval strategy tests
│   ├── test_utils.py             # Utility function tests
│   └── test_exceptions.py        # Exception hierarchy tests
├── examples/
│   └── run_single_request.py     # Single-request example
├── original/                     # Original source archive
└── logs/                         # Runtime logs
```

## Design Principles

### LLM + Traditional ML Hybrid
DMA runs Llama3:8b as the primary inference engine. When Ollama is unreachable, the system falls back to a deterministic TF-IDF centroid + retrieval-weighted classifier — no external dependency required at runtime.

### Graceful Degradation
Every component call is wrapped via `DegradationManager.safe_call()`. Single-point failures (Ollama crash, FAISS index corruption, TFA timeout) trigger automatic fallbacks; the pipeline continues with reduced functionality rather than aborting.

### Safety by Design
- **3-layer consensus voting** prevents over-reliance on any single agent
- **HITL** triggers on low confidence, critical risk, verification conflicts, or TFA-diagnosis divergence
- **Platt scaling** calibrates LLM confidence against empirical correctness
- **Chinese risk labels and recommendations** for clinical readability

### Privacy-Aware
Inter-agent messages are XOR-encrypted. Federated gradient updates include differential privacy noise (`secure_gradient_noise_std`). PII desensitization runs on every query before audit logging.

### Research-Ready
Configurable experiment suites (ablation, robustness, adaptability, common-vs-rare), structured request tracing with correlation IDs, per-stage latency logging, and JSONL audit trails make the system suitable for medical AI research and simulation.

## Compatibility Notes

- **Ollama** is the preferred DMA runtime. Set `allow_dma_legacy_fallback=true` (default) for offline environments.
- **MiniLM** (`sentence-transformers/all-MiniLM-L6-v2`) is lazy-loaded. Falls back to deterministic hash embeddings if unavailable.
- **FAISS** is optional — dense retrieval falls back to NumPy dot-product scan.
- The original source from `GU YUANJIE.docx` is preserved in `original/`.
- `# === MODIFIED:` annotations mark all changes from the original code.
