# MAPFM HMAE Architecture Upgrade — TFA Deep Integration

## Overview

This document describes the architectural optimization of the Heterogeneous Multi-Agent Ecosystem (HMAE) for medical AI, centered on deep integration of the refactored Temporal Foreseeing Agent (TFA) with all core modules.

**Version:** 2.0.0  
**Date:** 2026-05-27  
**Principle:** Diagnosis-first, TFA as auxiliary decision support, safety-first

---

## 1. Multimodal Data Flow Architecture

### 1.1 Data Input Layer

New multimodal endpoints in [api.py](api.py):

| Endpoint | Method | Input | Description |
|---|---|---|---|
| `/diagnose/multimodal` | POST | Multipart form | Full multimodal: CSV + TXT + images |
| `/diagnose/multimodal/simple` | POST | Multipart form | CSV + TXT only (no images) |
| `/diagnose` | POST | JSON | Text-only (backward compatible) |
| `/system/health` | GET | — | System health + degradation status |

Supported file types:
- **CSV**: Vital signs (HR, BP, Temp, RR, SpO2 with timestamps), Lab results (test name, value, unit, reference range)
- **TXT**: Clinical notes (structured `[timestamp] type` blocks or plain text)
- **JPG/PNG**: Medical imaging (X-ray, CT, MRI, ultrasound)

### 1.2 Preprocessing Pipeline

Implemented in [utils/multimodal_utils.py](utils/multimodal_utils.py):

```
Raw Files → Data Cleaning → Normalization → Desensitization → MultimodalPatientData
```

- **Data Cleaning**: Remove duplicates, outliers beyond clinical ranges, rows with >50% missing
- **Normalization**: Unified timestamp parsing (ISO 8601, Unix, flexible formats), unit standardization, vital sign normalization to [0,1]
- **Desensitization**: Regex-based PII stripping (names, SSN, MRN, email, phone, IP)
- **Output**: `MultimodalPatientData` dataclass (type-safe, validated)

### 1.3 Data Storage

- Temporary directory created per request (`tempfile.mkdtemp`)
- Images copied to temp dir for safe processing
- Automatic cleanup via `cleanup_temp_dir()` after request completion

### 1.4 Internal Data Transfer

All inter-module communication uses `MultimodalPatientData` objects (not raw dicts/JSON). Key methods:
- `vital_signs_array()` → (N, 6) float array
- `vital_signs_time_series()` → 1-D normalized series for TFA

---

## 2. DMA-TFA Integration

### 2.1 DMA Output Extensions

New fields in DMA.infer() output:
- `diagnosis_confidence`: float (same as confidence, semantic alias)
- `disease_severity`: string (`mild` | `moderate` | `severe` | `critical` | `unknown`)

Severity is determined by keyword matching against the query and prediction, with a confidence-based fallback.

### 2.2 TFA Confidence Calibration

TFA adjusts its prediction confidence based on DMA diagnostic confidence:

| DMA Confidence | TFA Adjustment | Mode |
|---|---|---|
| ≥ 0.9 | ×1.0 (unchanged) | `full_confidence` |
| 0.7–0.9 | ×0.9 (slight discount) | `slight_discount` |
| 0.6–0.7 | ×0.8 (moderate discount) | `moderate_discount` |
| < 0.6 | Degradation mode | `degraded` |

In degradation mode, TFA adds the notice: "诊断结果置信度较低，风险预测仅供参考"

### 2.3 Pipeline Reordering

Pipeline changed from `TFA → DMA` to `DMA → TFA` so TFA can use DMA confidence for calibration. DMA.infer() now runs without TFA context; TFA receives DMA confidence as input.

### 2.4 Diagnosis-Risk Cross-Validation

When TFA risk factors appear unrelated to the DMA diagnosis, the Verification module flags a warning. If serious conflicts exist, HITL is triggered.

---

## 3. RAA-TFA Integration

### 3.1 Risk-Aware Retrieval

New method `RetrievalAugmentedAgent.retrieve_risk_aware()`:
- When TFA outputs high/critical risk, automatically retrieves clinical guidelines about risk factors and interventions for the relevant disease
- Compares retrieved knowledge with TFA risk factors:
  - **Consistent** → +0.05 confidence adjustment
  - **Divergent** → −0.05 confidence adjustment, conflict flag set

### 3.2 Tiered Recommendations

`RetrievalAugmentedAgent.generate_risk_recommendations()` produces tiered clinical advice:

| Risk Level | Recommendation Tier |
|---|---|
| `low` | Routine monitoring, maintain current treatment |
| `medium` | Increased observation frequency, periodic review |
| `high` | Timely medical consultation, comprehensive assessment |
| `critical` | Immediate emergency care, ICU consideration |

### 3.3 RAA-TFA Recommendation Fusion

Fusion module merges TFA recommendations with RAA risk-knowledge into a unified output.

---

## 4. Fusion/Verification-TFA Integration

### 4.1 Comprehensive Answer Generation

`KnowledgeFusionAgent.fuse_with_tfa()` produces a structured comprehensive answer:

```json
{
  "diagnosis": {"prediction", "confidence", "severity", "intent", "status"},
  "risk_assessment": {"risk_level", "risk_score", "short_term_risk",
                       "mid_term_risk", "long_term_risk", "primary_risk_factors",
                       "reasoning", "calibration_mode"},
  "recommendations": ["..."],
  "risk_knowledge_cross_ref": {"guidelines_matched", "tfa_raa_consistency"},
  "disclaimer": "本系统输出为AI辅助决策参考..."
}
```

### 4.2 TFA Risk Report Verification

`SimulatedKnowledgeVerificationAgent.verify_tfa_report()` validates:
1. Risk scores within [0, 1] range
2. Risk factors medically relevant to diagnosis
3. Recommendations match risk level severity
4. Degradation notices are properly flagged

Serious conflicts (score out of range) block consensus approval.

### 4.3 Verification Vote Logic

Updated `vote()` method:
- Passes when: verification_ratio ≥ 0.50 AND serious_conflicts = 0 AND tfa_verified = True
- Previously only checked verification_ratio ≥ 0.50

---

## 5. Consensus Module — Layered Voting (Core Redesign)

### 5.1 Voting Weights

| Module | Votes | Rationale |
|---|---|---|
| DMA (diagnosis) | **2** | Core, non-negotiable — diagnosis is foundation of all medical decisions |
| TFA (temporal risk) | **1** | New — auxiliary decision support, evidence-based |
| Verification | **1** | Fact-checking and consistency validation |
| **Total** | **4** | |
| **Pass Threshold** | **3** (75%) | |

### 5.2 Layered Decision Rules (Top-Down)

```
LAYER 1 — FORCE HITL (bypasses voting)
├── DMA HITL=True (confidence < 0.62)
├── Verification serious_conflicts ≥ 1
├── TFA risk_level = critical
└── TFA-diagnosis conflict detected
→ Escalate to human, DO NOT vote

LAYER 2 — DMA VETO
├── DMA votes NO (confidence < threshold)
→ Reject regardless of other votes, escalate to HITL

LAYER 3 — NORMAL VOTE (confidence × risk matrix)
├── DMA ≥ 0.8 + TFA low/medium → AUTO-PASS (4/4)
├── DMA ≥ 0.8 + TFA high → AUTO-PASS with risk note (3/4)
├── DMA 0.6-0.8 + TFA low → AUTO-PASS (4/4)
├── DMA 0.6-0.8 + TFA medium/high → HITL ESCALATION
└── DMA < 0.6 (edge case) → HITL ESCALATION
```

### 5.3 Decision Logging

Every consensus decision is logged in structured format:
```
Consensus -> approved=True | votes={'DMA': True, 'TFA': True, 'Verification': True} | required=3 | reason='DMA confidence 0.88 >= 0.8, TFA risk level low, verification passed'
```

---

## 6. HITL-TFA Integration

### 6.1 TFA-Triggered Intervention

New `should_intervene_tfa()` triggers HITL when:
- **Priority 1**: TFA risk level = `critical` → immediate human review
- **Priority 2**: TFA-diagnosis conflict OR TFA in degraded mode
- **Priority 3**: Verification flagged TFA issues

### 6.2 Human Expert Capabilities

The HITL interface (CLI or future GUI) allows experts to:
- Approve TFA risk prediction
- Correct risk score / risk level
- Add or remove risk factors
- Modify recommendations

### 6.3 Intervention Recording

All TFA interventions are logged with:
- Original vs. corrected risk level/score
- Added/removed risk factors
- Modified recommendations
- Reviewer notes

### 6.4 Periodic TFA Error Reporting

`generate_tfa_intervention_report()` produces:
- Total TFA interventions
- Override rate
- False positive / false negative counts
- Error rate
- Common error type distribution

---

## 7. System Monitoring and Degradation

Implemented in [utils/system_monitor.py](utils/system_monitor.py):

### 7.1 Health Checks

`SystemMonitor.check()` monitors:
- **Ollama**: HTTP health check at `/api/tags`, latency measurement
- **MedTsLLM**: Model directory and weight file existence
- **GPU**: CUDA memory usage via torch
- **System**: CPU, RAM, disk via psutil

### 7.2 Three-Tier Degradation

| Level | Trigger | Action |
|---|---|---|
| **L1** | GPU memory > 90% | Switch to 4-bit quantized model |
| **L2** | MedTsLLM unavailable | Switch to heuristic TFA mode |
| **L3** | Ollama unavailable | Switch to pure rule-based DMA |

### 7.3 Auto-Recovery

Background monitoring thread (configurable interval, default 30s) continuously checks health. When conditions clear, the system automatically returns to NORMAL mode.

### 7.4 User Messaging

Each degradation level produces a user-facing Chinese message explaining the situation and impact.

---

## 8. Files Changed

### New Files
- `utils/__init__.py` — Package init
- `utils/multimodal_utils.py` — MultimodalPatientData + MultimodalPreprocessor (~450 lines)
- `utils/system_monitor.py` — SystemMonitor + DegradationAwareWrapper (~380 lines)
- `ARCHITECTURE_UPGRADE.md` — This document

### Modified Files
- `mapfm_ecosystem_repaired.py` — Core module modifications:
  - `DecisionMakingAgent`: `_assess_severity()`, `diagnosis_confidence`, `disease_severity` fields
  - `TemporalForeseeingAgent`: `_apply_dma_calibration()`, `_enrich_risk_report()`, `_classify_risk_level()`, `_extract_risk_factors()`, `_generate_reasoning()`, `_generate_recommendations()`, `forecast_with_dma_context()`
  - `RetrievalAugmentedAgent`: `retrieve_risk_aware()`, `generate_risk_recommendations()`
  - `KnowledgeFusionAgent`: `fuse_with_tfa()`
  - `SimulatedKnowledgeVerificationAgent`: `verify_tfa_report()`, updated `vote()`
  - `ConsensusModule`: Complete rewrite — layered voting with TFA weight
  - `HumanInTheLoopManager`: `should_intervene_tfa()`, `process_tfa_intervention()`, `generate_tfa_intervention_report()`
  - `HeterogeneousMultiAgentEcosystem.run_collaborative_task()`: Pipeline reordered, TFA with DMA context, risk-aware RAA, comprehensive answer, enhanced HITL
- `api.py` — Multimodal upload endpoints, system health endpoint

### Preserved (Unchanged)
- All original module interfaces remain backward-compatible
- `BaseAgent` and its ABC hierarchy unchanged
- `SystemConfig` / `EcosystemConfig` dataclasses unchanged (new fields optional)
- `OnlineKnowledgeBaseManager`, `TriSourceKnowledgeFusion`, `KnowledgeDistillationEngine`, `DegradationManager`, `ContinuousLearningLoop` unchanged
- All test files unchanged
- `load_medical_dataset()`, `semantic_embedding()`, and all utility functions unchanged
- `PrivacySecurityAgent`, `MaintenanceAgent`, `CommunicationTopologyManager` unchanged

---

## 9. Backward Compatibility Guarantees

1. **Pure text queries**: When no multimodal files are uploaded, the system behaves identically to v1.x
2. **TFA disabled**: Setting `enable_tfa=False` removes TFA from the voting and pipeline
3. **Fallback paths**: All degradation fallbacks preserved — system never crashes
4. **Original DMA contract**: `infer()` output still contains all original fields plus new ones
5. **Existing tests**: All 6 test files continue to pass without modification

---

## 10. Module Interaction Diagram

```
                    ┌──────────────┐
                    │  API Layer   │
                    │  (FastAPI)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         CSV/TXT       Images       Text-only
              │            │            │
              ▼            ▼            │
    ┌──────────────────────────┐        │
    │  MultimodalPreprocessor  │        │
    │  (multimodal_utils.py)   │        │
    └────────────┬─────────────┘        │
                 │                      │
                 ▼                      │
         MultimodalPatientData          │
                 │                      │
                 └──────────┬───────────┘
                            ▼
              ┌─────────────────────────┐
              │      Perception         │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │    RAA (Retrieval)      │
              │  + retrieve_risk_aware  │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Fusion + Verification  │
              │  + verify_tfa_report    │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   DMA (Diagnosis)       │
              │   + severity field      │
              └────────────┬────────────┘
                           │ dma_confidence
                           ▼
              ┌─────────────────────────┐
              │   TFA (Risk Forecast)   │
              │   + DMA calibration     │
              │   + risk classification │
              └────────────┬────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │   HITL   │ │Fusion    │ │Consensus │
        │ + TFA    │ │+ unified │ │+ layered │
        │ triggers │ │ answer   │ │ voting   │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          ▼
                 ┌─────────────────┐
                 │  Final Output   │
                 │  + comprehensive│
                 │  + risk_knowledge│
                 └─────────────────┘
```

---

## 11. System Monitor Integration

```
SystemMonitor (background thread, 30s interval)
    │
    ├── Ollama health ──→ DegradationLevel.LEVEL_3
    ├── MedTsLLM load ──→ DegradationLevel.LEVEL_2
    ├── GPU memory    ──→ DegradationLevel.LEVEL_1
    ├── System RAM    ──→ Warning only
    └── Disk space    ──→ Warning only
            │
            ▼
    DegradationAwareWrapper
            │
            ├── NORMAL:  full MedTsLLM + Ollama
            ├── LEVEL_1: 4-bit quantized MedTsLLM
            ├── LEVEL_2: heuristic TFA fallback
            └── LEVEL_3: pure rule-based DMA
```
