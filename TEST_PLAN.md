# HMAE v2.0 Full Pipeline Test Plan

**Version:** 2.0  
**Date:** 2026-05-27  
**Status:** Active

---

## 1. Overview

This document defines the complete test strategy for the HMAE v2.0 architecture upgrade. The upgrade integrates MedTsLLM (BioBERT backbone) into the TFA module and establishes unified logging/output formats across all modules.

### 1.1 Test Objectives

| Objective | Success Criteria |
|-----------|-----------------|
| TFA module correctness | All 5 TFA test suites pass at 100% |
| System integration | All modules interact correctly with TFA |
| No regressions | All original functionality preserved |
| Format compliance | 100% log lines pass FormatChecker |
| Performance | Single task <= 15s, concurrent 5 <= 30s |
| Security | No PII leaks, no harmful output, local-only processing |

### 1.2 Test Data Policy

- **ALL test data is synthetic or from the held-out test split**
- **ZERO training data is used for testing**
- **All test inputs are reproducible (fixed seeds)**

---

## 2. Test Suites

### 2.1 TFA Connectivity Test (`test_tfa_connectivity.py`)

| # | Test Case | Expected Result |
|---|-----------|-----------------|
| 1 | Ollama service health check | Service reachable or graceful skip |
| 2 | Basic forecast structure | Returns dict with all required keys |
| 3 | Risk probabilities in [0, 1] | All windows validated |
| 4 | Risk level classification | One of: low/medium/high/critical |
| 5 | Empty query raises error | AgentError raised |
| 6 | DMA calibration applied | Different confidence → different mode |
| 7 | Response time < 15s | Single forecast under time limit |
| 8 | Recommendations present | Non-empty list returned |

### 2.2 TFA Multimodal Test (`test_tfa_multimodal.py`)

| # | Test Case | Expected Result |
|---|-----------|-----------------|
| 1 | Vital signs only | Valid forecast |
| 2 | Lab results only | Valid forecast |
| 3 | Clinical text only | Valid forecast |
| 4 | Vitals + Labs | Valid forecast |
| 5 | Vitals + Clinical text | Valid forecast |
| 6 | Labs + Clinical text | Valid forecast |
| 7 | All three data types | Valid forecast with risk factors |
| 8 | None history | Graceful handling |
| 9 | Empty signal dict | Valid forecast |
| 10 | Anomalous vitals | Elevated risk level |

### 2.3 TFA Medical Knowledge Check (`test_tfa_medical_check.py`)

| # | Test Case | Expected Result |
|---|-----------|-----------------|
| 1 | Hypertension stable | Risk < 50% |
| 2 | Acute MI | Risk meaningfully elevated |
| 3 | Symptomatic patients | Risk > 0% |
| 4 | Any scenario | Risk < 100% |
| 5 | Recommendations | No dangerous phrases |
| 6 | Risk factors | At least 1 factor identified |
| 7 | Low DMA confidence | Degradation mode activated |
| 8 | Same input twice | Identical output (reproducibility) |

### 2.4 TFA Fallback Test (`test_tfa_fallback.py`)

| # | Test Case | Expected Result |
|---|-----------|-----------------|
| 1 | Normal mode | TFA works correctly |
| 2 | Safe call with failure | Fallback value returned |
| 3 | Degradation levels | Correct escalation |
| 4 | Recovery | Marking healthy clears degradation |
| 5 | Fallback contract | Output has all required fields |
| 6 | Stats tracking | Degradation counts incremented |
| 7 | MedTsLLM unavailable | Graceful heuristic fallback |

### 2.5 TFA Performance Test (`test_tfa_performance.py`)

| # | Test Case | Threshold |
|---|-----------|-----------|
| 1 | Single task latency | Avg <= 15s |
| 2 | Concurrent 5 tasks | Total <= 30s |
| 3 | Memory usage | RSS <= 32 GB |
| 4 | CPU usage | Not saturated |
| 5 | Warm vs cold start | Ratio <= 3x |
| 6 | Throughput | 10 tasks sequential benchmark |

### 2.6 Integration Test (`test_integration.py`)

| # | Test Case | Expected Result |
|---|-----------|-----------------|
| 1 | Full pipeline | All stages complete without crash |
| 2 | TFA in consensus | TFA vote present in votes dict |
| 3 | TFA → HITL | High risk reflected in HITL |
| 4 | RAA risk-aware | Risk knowledge dict populated |
| 5 | Fusion with TFA | Comprehensive answer produced |
| 6 | Concurrent tasks | All tasks complete successfully |
| 7 | Degradation tracking | Component states populated |
| 8 | Format compliance | All log formats validate |

### 2.7 Regression Test (`test_regression.py`)

| # | Test Case | Expected Result |
|---|-----------|-----------------|
| 1-4 | 4 canonical tasks | All work as before upgrade |
| 5 | DMA classification | Valid prediction + confidence |
| 6 | RAA retrieval | Docs + metadata returned |
| 7 | Fusion/Verification | Accept/reject docs correctly |
| 8 | Consensus voting | Valid vote structure |
| 9 | HITL trigger | Low confidence → HITL triggered |
| 10 | Format backward compat | Original fields preserved |

### 2.8 Ablation Study (`ablation_study.py`)

| Condition | TFA | MedTsLLM | Expected |
|-----------|-----|----------|----------|
| baseline | Yes | No (fallback) | Reference metrics |
| with_medtsllm | Yes | Yes | Best accuracy, lowest HITL |
| without_tfa | No | N/A | No risk prediction |
| tfa_fallback_only | Yes | No | Same as baseline |

**Acceptance criteria:**
- `with_medtsllm` accuracy > baseline accuracy
- `with_medtsllm` HITL rate < baseline HITL rate
- `with_medtsllm` risk AUROC >= 0.85 (if ground truth available)

### 2.9 Security Test (`test_security.py`)

| # | Test Case | Expected Result |
|---|-----------|-----------------|
| 1 | Names removed | No PII in output |
| 2 | Identifiers masked | Phone/SSN/email hidden |
| 3 | Encryption round-trip | Decrypt(Encrypt(x)) == x |
| 4 | Key isolation | Different keys → different ciphertext |
| 5 | Gradient noise | Federated gradient perturbed |
| 6 | Malicious input | No crash, no harmful output |
| 7 | Disclaimer | Present in comprehensive answer |
| 8 | Local-only | No external URLs in code |
| 9 | Audit logging | Events recorded |
| 10 | Internal state | Not exposed in public API |

---

## 3. Acceptance Criteria (Gate Check)

**ALL** of the following must be true for the upgrade to be signed off:

| # | Criterion | Threshold | Measured By |
|---|-----------|-----------|-------------|
| AC-1 | TFA connectivity tests | 100% pass | test_tfa_connectivity.py |
| AC-2 | TFA multimodal tests | 100% pass | test_tfa_multimodal.py |
| AC-3 | TFA medical checks | 100% pass | test_tfa_medical_check.py |
| AC-4 | TFA fallback tests | 100% pass | test_tfa_fallback.py |
| AC-5 | Integration tests | 100% pass | test_integration.py |
| AC-6 | Regression tests | 100% pass | test_regression.py |
| AC-7 | Security tests | 100% pass | test_security.py |
| AC-8 | Single task latency | <= 15s | test_tfa_performance.py |
| AC-9 | Concurrent 5 latency | <= 30s | test_tfa_performance.py |
| AC-10 | GPU VRAM (4-bit) | <= 16 GB | test_tfa_performance.py |
| AC-11 | Risk AUROC | >= 0.85 | ablation_study.py |
| AC-12 | Risk AUPRC | >= 0.75 | ablation_study.py |
| AC-13 | Format compliance | 100% lines pass | utils/format_checker.py |
| AC-14 | Documentation complete | All docs present | File manifest check |

---

## 4. Running Tests

### Quick (unit-level only, no dataset needed):
```bash
python tests/test_tfa_connectivity.py
python tests/test_tfa_multimodal.py
python tests/test_tfa_medical_check.py
python tests/test_tfa_fallback.py
python tests/test_tfa_performance.py
python tests/test_security.py
```

### Full (requires dataset):
```bash
python tests/test_integration.py
python tests/test_regression.py
python tests/ablation_study.py
```

### All-in-one:
```bash
python -m pytest tests/ -v --tb=short
```

---

## 5. Test Environment Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.10+ |
| NumPy | 1.24+ |
| pandas | 2.0+ |
| scikit-learn | 1.3+ |
| Ollama (optional) | Latest |
| RAM | 8 GB |
| Disk | 2 GB free |
