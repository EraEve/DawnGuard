# HMAE Test Report

**Date:** {date}  
**Tester:** {tester_name}  
**System Version:** HMAE v2.0  
**MedTsLLM Version:** {medtsllm_version}  
**Environment:** {environment}

---

## Executive Summary

| Metric | Result | Threshold | Status |
|--------|--------|-----------|--------|
| TFA Connectivity | {tfa_conn_pass}/{tfa_conn_total} | 100% | {tfa_conn_status} |
| TFA Multimodal | {tfa_mm_pass}/{tfa_mm_total} | 100% | {tfa_mm_status} |
| TFA Medical Check | {tfa_med_pass}/{tfa_med_total} | 100% | {tfa_med_status} |
| TFA Fallback | {tfa_fb_pass}/{tfa_fb_total} | 100% | {tfa_fb_status} |
| Integration | {int_pass}/{int_total} | 100% | {int_status} |
| Regression | {reg_pass}/{reg_total} | 100% | {reg_status} |
| Security | {sec_pass}/{sec_total} | 100% | {sec_status} |
| Single Task Latency | {single_latency:.2f}s | <= 15s | {single_latency_status} |
| Concurrent 5 Latency | {concurrent_latency:.2f}s | <= 30s | {concurrent_latency_status} |
| Format Compliance | {format_rate:.1%} | 100% | {format_status} |
| Ablation: MedTsLLM vs Baseline | {ablation_delta:+.4f} accuracy | >0 | {ablation_status} |

**Overall Verdict:** {overall_verdict}

---

## 1. TFA Connectivity Test Results

| # | Test | Status | Duration | Notes |
|---|------|--------|----------|-------|
| 1 | Ollama service check | {tc1_status} | {tc1_duration} | {tc1_notes} |
| 2 | Basic forecast structure | {tc2_status} | {tc2_duration} | {tc2_notes} |
| 3 | Risk probabilities range | {tc3_status} | {tc3_duration} | {tc3_notes} |
| 4 | Risk level classification | {tc4_status} | {tc4_duration} | {tc4_notes} |
| 5 | Empty query error | {tc5_status} | {tc5_duration} | {tc5_notes} |
| 6 | DMA calibration | {tc6_status} | {tc6_duration} | {tc6_notes} |
| 7 | Response time <15s | {tc7_status} | {tc7_duration} | {tc7_notes} |
| 8 | Recommendations present | {tc8_status} | {tc8_duration} | {tc8_notes} |

**Section Result:** {tfa_conn_pass}/{tfa_conn_total} passed

---

## 2. TFA Multimodal Test Results

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | Vital signs only | {tmm1} | |
| 2 | Lab results only | {tmm2} | |
| 3 | Clinical text only | {tmm3} | |
| 4 | Vitals + Labs | {tmm4} | |
| 5 | Vitals + Clinical | {tmm5} | |
| 6 | Labs + Clinical | {tmm6} | |
| 7 | All three types | {tmm7} | |
| 8 | None history | {tmm8} | |
| 9 | Empty data | {tmm9} | |
| 10 | Anomalous values | {tmm10} | |

**Section Result:** {tfa_mm_pass}/{tfa_mm_total} passed

---

## 3. TFA Medical Knowledge Check Results

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | Hypertension risk moderate | {tmk1} | |
| 2 | Acute MI risk elevated | {tmk2} | |
| 3 | No zero risk for sick | {tmk3} | |
| 4 | No 100% risk | {tmk4} | |
| 5 | Safe recommendations | {tmk5} | |
| 6 | Risk factors present | {tmk6} | |
| 7 | Low confidence degrades | {tmk7} | |
| 8 | Reproducibility | {tmk8} | |

**Section Result:** {tfa_med_pass}/{tfa_med_total} passed

---

## 4. TFA Fallback Test Results

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | Normal mode works | {tfb1} | |
| 2 | Safe call returns fallback | {tfb2} | |
| 3 | Degradation levels escalate | {tfb3} | |
| 4 | Recovery after healthy | {tfb4} | |
| 5 | Fallback preserves contract | {tfb5} | |
| 6 | Degradation stats tracked | {tfb6} | |
| 7 | MedTsLLM unavailable graceful | {tfb7} | |

**Section Result:** {tfa_fb_pass}/{tfa_fb_total} passed

---

## 5. Integration Test Results

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | Full pipeline completes | {int1} | |
| 2 | Consensus includes TFA vote | {int2} | |
| 3 | TFA triggers HITL | {int3} | |
| 4 | RAA risk-aware retrieval | {int4} | |
| 5 | Fusion with TFA | {int5} | |
| 6 | Concurrent tasks | {int6} | |
| 7 | Degradation tracking | {int7} | |
| 8 | Format compliance | {int8} | |

**Section Result:** {int_pass}/{int_total} passed

---

## 6. Regression Test Results

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | Task 01: Lung Cancer symptoms | {reg1} | |
| 2 | Task 02: Heart Failure prognosis | {reg2} | |
| 3 | Task 03: Colorectal Cancer treatments | {reg3} | |
| 4 | Task 04: High Blood Pressure treatments | {reg4} | |
| 5 | DMA classification | {reg5} | |
| 6 | RAA retrieval | {reg6} | |
| 7 | Fusion/Verification | {reg7} | |
| 8 | Consensus voting | {reg8} | |
| 9 | HITL trigger | {reg9} | |
| 10 | Original format preserved | {reg10} | |

**Section Result:** {reg_pass}/{reg_total} passed

---

## 7. Security Test Results

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | Names removed | {sec1} | |
| 2 | Identifiers masked | {sec2} | |
| 3 | Encryption round-trip | {sec3} | |
| 4 | Key isolation | {sec4} | |
| 5 | Gradient perturbation | {sec5} | |
| 6 | Malicious input safe | {sec6} | |
| 7 | Disclaimer present | {sec7} | |
| 8 | Local-only processing | {sec8} | |
| 9 | Audit logging | {sec9} | |
| 10 | Internal state protected | {sec10} | |

**Section Result:** {sec_pass}/{sec_total} passed

---

## 8. Performance Report

| Metric | Measured | Threshold | Status |
|--------|----------|-----------|--------|
| Single task avg latency | {single_latency:.2f}s | <= 15.0s | {single_latency_status} |
| Single task max latency | {single_max:.2f}s | N/A | — |
| Concurrent 5 total | {conc_total:.2f}s | <= 30.0s | {conc_status} |
| Throughput (10 tasks) | {throughput:.2f} task/s | N/A | — |
| Memory (RSS) | {memory_gb:.2f} GB | <= 32.0 GB | {memory_status} |

---

## 9. Ablation Study Results

| Condition | Accuracy | Macro-F1 | HITL Rate | Avg Latency | TFA Risk |
|-----------|----------|----------|-----------|-------------|----------|
| baseline | {bl_acc:.4f} | {bl_f1:.4f} | {bl_hitl:.4f} | {bl_lat:.2f}s | {bl_risk:.4f} |
| with_medtsllm | {wm_acc:.4f} | {wm_f1:.4f} | {wm_hitl:.4f} | {wm_lat:.2f}s | {wm_risk:.4f} |
| without_tfa | {wt_acc:.4f} | {wt_f1:.4f} | {wt_hitl:.4f} | {wt_lat:.2f}s | N/A |
| tfa_fallback_only | {tf_acc:.4f} | {tf_f1:.4f} | {tf_hitl:.4f} | {tf_lat:.2f}s | {tf_risk:.4f} |

**MedTsLLM Impact:**
- Accuracy delta: {acc_delta:+.4f}
- F1 delta: {f1_delta:+.4f}
- HITL rate delta: {hitl_delta:+.4f}

---

## 10. Format Compliance Report

| Log File | Lines Checked | Errors | Compliance |
|----------|--------------|--------|------------|
{format_table}

**Overall Format Compliance:** {format_rate:.1%}

---

## 11. Issues Found

{issues}

---

## 12. Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Developer | | | |
| QA Engineer | | | |
| Medical Reviewer | | | |
| Project Lead | | | |

---

*Report generated by HMAE Test Framework v2.0*
