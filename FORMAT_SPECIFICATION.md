# HMAE Unified Log & Output Format Specification

**Version:** 2.0 (HMAE Architecture Upgrade)  
**Last Updated:** 2026-05-27  
**Status:** Enforced via `utils/format_checker.py`

---

## 1. Core Principles

1. **Original format 100% preserved** — All original fields, order, symbols, and numeric precision MUST remain unchanged.
2. **New fields appended only** — All new information MUST use `| key=value` appended at the end of the original log line.
3. **Float precision** — Probabilities/confidences: 4 decimal places. Percentages: 2 decimal places.
4. **Booleans** — Always `True` or `False` (PascalCase).
5. **Log levels** — DEBUG, INFO, WARNING, ERROR, CRITICAL (standard Python levels).

---

## 2. Module Log Formats

### 2.1 DMA (Decision Making Agent)

**Original format (MUST be preserved):**
```
DMA -> prediction={disease} | confidence={confidence:.4f} | HITL={hitl}
```

**Updated format (new fields appended):**
```
DMA -> prediction={disease} | confidence={confidence:.4f} | HITL={hitl} | intent={intent} | status={status} | severity={severity}
```

| Field | Type | Description | Valid Values |
|-------|------|-------------|-------------|
| `prediction` | string | Disease name | Any medical area label |
| `confidence` | float (4dp) | Diagnostic confidence | 0.0000 - 1.0000 |
| `HITL` | bool | Whether HITL was triggered | True, False |
| `intent` | string | User intent classification | symptom_query, treatment_query, prognosis_query, prevention_query, screening_check, symptom_inquiry, treatment_inquiry, prognosis, prevention, general_inquiry |
| `status` | string | Diagnosis result status | success, low_confidence, error |
| `severity` | string | Disease severity | mild, moderate, severe, critical, unknown |

**Example:**
```
DMA -> prediction=Lung Cancer | confidence=0.4810 | HITL=True | intent=symptom_query | status=low_confidence | severity=moderate
```

---

### 2.2 RAA (Retrieval Augmented Agent)

**Original format (MUST be preserved):**
```
RAA -> strategy={strategy} | Nash={nash} | rounds={rounds} | verified_relevance={relevance:.4f}
```

**Updated format (new fields appended):**
```
RAA -> strategy={strategy} | Nash={nash} | rounds={rounds} | verified_relevance={relevance:.4f} | evidence_level={level:.1f} | conflicts={conflicts}
```

| Field | Type | Description | Valid Values |
|-------|------|-------------|-------------|
| `strategy` | string | Retrieval strategy | mixed, rerank, adaptive, Mixed-RAG, Rerank-RAG, Adaptive-RAG:* |
| `Nash` | bool | Nash equilibrium reached | True, False |
| `rounds` | int | Negotiation rounds | 0-6 |
| `verified_relevance` | float (4dp) | Average verified relevance | 0.0000 - 1.0000 |
| `evidence_level` | float (1dp) | Average evidence grade (1=highest) | 1.0 - 4.0 |
| `conflicts` | int | Conflict count in retrieved docs | 0+ |

**Example:**
```
RAA -> strategy=mixed | Nash=True | rounds=2 | verified_relevance=0.5680 | evidence_level=2.5 | conflicts=3
```

---

### 2.3 TFA (Temporal Foreseeing Agent)

**Original format (MUST be preserved):**
```
TFA -> future 24h deterioration risk={risk:.2f}%
```

**Updated format (new fields appended):**
```
TFA -> future 24h deterioration risk={risk:.2f}% | risk_level={level} | confidence={confidence:.4f} | source={source} | model_version={version}
```

| Field | Type | Description | Valid Values |
|-------|------|-------------|-------------|
| `risk` | float (2dp) | 24h deterioration probability | 0.00 - 100.00 |
| `risk_level` | string | Risk category | low, medium, high, critical |
| `confidence` | float (4dp) | Model prediction confidence | 0.0000 - 1.0000 |
| `source` | string | Prediction source | medtsllm, fallback, rules |
| `model_version` | string | Model version used | MedTsLLM-v1.5, heuristic-v2.0, etc. |

**Example:**
```
TFA -> future 24h deterioration risk=72.00% | risk_level=high | confidence=0.8500 | source=medtsllm | model_version=MedTsLLM-v1.5
```

---

### 2.4 Fusion/Verification

**Original format (MUST be preserved):**
```
Fusion/Verification -> input={input} | dedup={dedup} | conflicts={conflicts} | verified={verified}/{total}
```

**Updated format (new fields appended):**
```
Fusion/Verification -> input={input} | dedup={dedup} | conflicts={conflicts} | verified={verified}/{total} | moderate_conflicts={moderate} | minor_conflicts={minor} | status={status}
```

| Field | Type | Description | Valid Values |
|-------|------|-------------|-------------|
| `input` | int | Total input documents | 0+ |
| `dedup` | int | After deduplication | 0+ |
| `conflicts` | int | Total semantic conflicts | 0+ |
| `verified` | int | Docs passing verification | 0+ |
| `total` | int | Docs submitted for verification | 0+ |
| `moderate_conflicts` | int | Moderate conflicts (hamming > 26) | 0+ |
| `minor_conflicts` | int | Minor conflicts (hamming 4-26) | 0+ |
| `status` | string | Overall verification status | passed, passed_with_warnings, failed |

**Example:**
```
Fusion/Verification -> input=26 | dedup=20 | conflicts=15 | verified=20/20 | moderate_conflicts=5 | minor_conflicts=10 | status=failed
```

---

### 2.5 Consensus

**Original format (MUST be preserved):**
```
Consensus -> approved={approved} | votes={votes} | required={required}
```

**Updated format (new fields appended):**
```
Consensus -> approved={approved} | votes={votes} | required={required} | reason="{reason}" | hitl_triggered={hitl} | risk_level={risk_level} | tier={tier}
```

| Field | Type | Description | Valid Values |
|-------|------|-------------|-------------|
| `approved` | bool | Final approval decision | True, False |
| `votes` | dict | Individual agent votes | e.g. {'DMA': True, 'TFA': True, 'Verification': True} |
| `required` | int | Required passing threshold | 1-4 |
| `reason` | string (quoted) | Decision rationale | Free text, double-quoted |
| `hitl_triggered` | bool | Whether HITL was escalated | True, False |
| `risk_level` | string | Consensus risk level | low, medium, high, critical |
| `tier` | string | Decision tier used | force_hitl, dma_veto, auto_pass, auto_pass_risk_noted, escalation_risk_uncertainty, escalation_low_confidence |

**Example:**
```
Consensus -> approved=False | votes={'DMA': False, 'TFA': False, 'Verification': True} | required=3 | reason="Low confidence and high information conflict" | hitl_triggered=True | risk_level=high | tier=force_hitl
```

---

### 2.6 HITL (Human-In-The-Loop)

**Original format (established baseline):**
```
HITL -> triggered={triggered} | reason={reason}
```

**Updated format (new fields appended):**
```
HITL -> triggered={triggered} | reason={reason} | task_id={id} | intervention_time={time} | operator={operator} | priority={priority}
```

| Field | Type | Description | Valid Values |
|-------|------|-------------|-------------|
| `triggered` | bool | Whether HITL was triggered | True, False |
| `reason` | string | Trigger reason(s) | Pipe-separated reasons |
| `task_id` | string | Task identifier | 2-digit zero-padded (01-99) |
| `intervention_time` | datetime | ISO 8601 timestamp | YYYY-MM-DD HH:MM:SS |
| `operator` | string | Operator identifier | system, admin, etc. |
| `priority` | int | Intervention priority | 0=info, 1=critical, 2=conflict, 3=warning |

**Example:**
```
HITL -> triggered=True | reason=High risk | task_id=04 | intervention_time=2024-05-20 14:30:00 | operator=admin | priority=1
```

---

## 3. Unified Error Log Format

All error and critical logs MUST follow this exact format:

```
[LEVEL] [TIMESTAMP] [MODULE] - ERROR_CODE: Error message. Details: {details}
```

### Error Codes

| Code | Description |
|------|-------------|
| E-001 | Model invocation failure |
| E-002 | Data format error |
| E-003 | File not found |
| E-004 | Network error |
| E-005 | Configuration error |
| E-006 | Authentication failure |
| E-007 | Resource exhausted |
| E-008 | Timeout |
| E-009 | Internal error |
| E-010 | Validation failure |

### Examples

```
[ERROR] [2024-05-20 14:30:00] [TFA] - E-001: MedTsLLM invocation failed. Details: Connection refused
[CRITICAL] [2024-05-20 14:30:05] [Consensus] - E-003: Configuration file not found. Details: config.json not found
[ERROR] [2024-05-20 14:31:00] [RAA] - E-004: Network timeout. Details: Request to Ollama timed out after 120s
```

---

## 4. Final User Output Format

The final user-facing output MUST follow this structure:

```
# 诊断结果
{diagnosis_result}

# 病情恶化风险评估
- 风险等级：{risk_level}（{risk_score:.2f}%）
- 置信度：{confidence:.2f}%
- 主要风险因素：
  1. {risk_factor_1}
  2. {risk_factor_2}
  3. {risk_factor_3}

# 预警建议
1. {suggestion_1}
2. {suggestion_2}
3. {suggestion_3}

# 推理过程
{reasoning}

---
⚠️ 免责声明：本结果仅用于研究目的，不构成临床建议，如有不适请及时就医。
```

### Risk Level Color Coding

| Level | Color |
|-------|-------|
| low | Green |
| medium | Yellow |
| high | Orange |
| critical | Red |

### Rules

- All section headers use `#` markdown formatting (bold)
- Lists use numbered or bullet markers
- For text-only tasks (no multimodal data), the risk assessment section is auto-hidden
- Disclaimer MUST appear at the end with visible warning marker

---

## 5. Format Validation

Use the `FormatChecker` class from `utils/format_checker.py`:

```python
from utils.format_checker import FormatChecker, format_error_log

# Validate a log line
checker = FormatChecker()
checker.check_log_line("DMA -> prediction=Flu | confidence=0.9200 | HITL=False | intent=symptom_query | status=success | severity=mild", "DMA")  # True

# Validate an entire log file
errors = checker.check_log_file("logs/app.log")

# Validate final user output
checker.check_output(final_output)  # True/False

# Generate formatted error log
format_error_log("ERROR", "TFA", "E-001", "Model call failed", "Timeout after 120s")
```

### Startup Auto-Check

Format validation runs automatically at system startup. If any log file contains non-conforming lines, a warning is emitted with detailed diagnostics and suggested fixes.

```python
from utils.format_checker import run_startup_check
report = run_startup_check("logs")
```

---

## 6. Compliance Summary

| Module | Original Fields Preserved | New Fields Appended | Float Precision | Validation |
|--------|--------------------------|---------------------|-----------------|------------|
| DMA | 3/3 | +3 (intent, status, severity) | 4dp | regex |
| RAA | 4/4 | +2 (evidence_level, conflicts) | 4dp | regex |
| TFA | 1/1 | +4 (risk_level, confidence, source, model_version) | 2dp (risk), 4dp (conf) | regex |
| Fusion/Verification | 4/4 | +3 (moderate_conflicts, minor_conflicts, status) | N/A | regex |
| Consensus | 3/3 | +4 (reason, hitl_triggered, risk_level, tier) | N/A | regex |
| HITL | 2/2 | +4 (task_id, intervention_time, operator, priority) | N/A | regex |
| Error | N/A (new) | Full unified format | N/A | regex |
