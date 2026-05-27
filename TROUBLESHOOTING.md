# HMAE v2.0 Troubleshooting Guide

Common issues encountered during testing and deployment, with solutions.

---

## TFA / MedTsLLM Issues

### Ollama service not reachable

**Symptom:**
```
[ERROR] [2024-05-20 14:30:00] [TFA] - E-004: Connection refused. Details: http://127.0.0.1:11434
```

**Solution:**
1. Verify Ollama is installed: `ollama --version`
2. Start the service: `ollama serve`
3. Check the port: `curl http://127.0.0.1:11434/api/tags`
4. If port is different, update `ollama_base_url` in EcosystemConfig

### MedTsLLM model not found

**Symptom:** `ERROR: model "medtsllm" not found`

**Solution:**
1. Check available models: `ollama list`
2. Pull the model if missing: `ollama pull medtsllm`
3. Update `ollama_model_name` in EcosystemConfig
4. The system auto-falls back to heuristic mode — check `source=fallback` in TFA logs

### TFA always outputs source=fallback

**Symptom:** All TFA logs show `| source=fallback`

**Solution:**
1. This is expected if MedTsLLM is not deployed — the heuristic fallback is the normal mode without Ollama
2. To use MedTsLLM, ensure `enable_medtsllm=True` in EcosystemConfig
3. Verify `medtsllm_adapter.py` exists and imports correctly
4. Check that Ollama has the MedTsLLM model loaded

### Risk score always 0 or very low

**Symptom:** `TFA -> future 24h deterioration risk=0.00%`

**Solution:**
1. Verify the query contains medical terms — TFA needs clinical context
2. Provide `authoritative_signal` parameters (vital signs, lab results)
3. Check that `enable_tfa=True` in config
4. The heuristic fallback uses query features — very short queries may produce near-zero risk

---

## Format Validation Issues

### Log line rejected by FormatChecker

**Symptom:**
```
Line 42 [DMA]: Line does not match DMA format pattern | raw=DMA -> prediction=Flu | confidence=0.9200 | HITL=False
```

**Common causes and fixes:**

1. **Missing new fields**: If `| intent=... | status=...` is missing, the regex won't match. Update your log output.

2. **Wrong precision**:
   - DMA confidence needs 4 decimal places: `0.9200` not `0.92`
   - TFA risk needs 2 decimal places: `15.50%` not `15.5%`

3. **Boolean format**: Must be `True`/`False`, not `true`/`false` or `1`/`0`

4. **Field name mismatch**:
   - RAA: `verified_relevance` not `verified relevance`
   - TFA: exact phrase `future 24h deterioration risk=` required

### Auto-fix available

```python
from utils.format_checker import FormatChecker
checker = FormatChecker()
fixed = checker.auto_fix_line(bad_line, "DMA")
if fixed:
    print(f"Fixed: {fixed}")
```

---

## Test Failures

### test_tfa_connectivity.py — "Ollama not running"

This test is designed to skip gracefully. The test suite works correctly even without Ollama.

### test_regression.py — "No dataset found"

Place `medNo.22.csv` in the project root, or run the test from the correct directory:
```bash
cd MAPFM_medical_AI_repaired_complete
python tests/test_regression.py
```

### test_integration.py — "Consensus vote missing TFA"

1. Verify `enable_tfa=True` in EcosystemConfig
2. Verify `enable_consensus=True`
3. Check that TFA forecast completed before Consensus voting
4. The integration test creates its own ecosystem instance — check the config

### ablation_study.py — "Low accuracy across all conditions"

1. This uses a small sample (12 by default) — results are approximate
2. Without MedTsLLM, DMA falls back to heuristic/simulated mode
3. Increase `max_samples` for more stable results
4. Verify dataset quality and class balance

---

## Performance Issues

### Single task > 15 seconds

1. **Ollama model not cached**: First inference loads the model into memory. Subsequent calls are faster.
2. **Model too large**: Reduce context length via `ollama_num_ctx` in config
3. **CPU-only mode**: Without GPU, 7B-parameter models are slow. Use heuristic mode for faster CPU-only operation
4. **Disk I/O**: Verify logs directory is on fast storage

### Memory exceeds 32 GB

1. Check for memory leaks in long-running loops
2. Reduce `max_samples` in ablation/robustness experiments
3. The heuristic fallback uses negligible memory — switch to it if RAM is constrained
4. Ollama models: 4-bit quantization should use < 8 GB VRAM

### Concurrent tasks timeout

1. Reduce `max_concurrent_requests` in EcosystemConfig
2. Default is 4 — try 2 for resource-constrained environments
3. Increase `agent_timeout_seconds` from 120 to 180

---

## Security & Privacy

### Desensitizer not masking all PII

The built-in desensitizer uses regex patterns. For production use, integrate with a medical-grade de-identification tool. Current patterns cover:
- Names (common formats)
- Phone numbers
- Email addresses
- SSN-like patterns
- Dates of birth

### Encryption key management

The `PrivacySecurityAgent` uses SHA-256 hashed keys. For production:
1. Use environment variables for keys, not hardcoded strings
2. Rotate keys periodically
3. Use proper key management (AWS KMS, HashiCorp Vault)

---

## General Issues

### Import errors after upgrade

If `from utils import ...` fails after the utils.py → utils/ conversion:
```bash
# Verify the utils package structure
ls utils/
# Should show: __init__.py, format_checker.py

# Clear any cached .pyc files
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```

### Config file not found

Create `config.yaml` in the project root:
```yaml
random_seed: 42
confidence_threshold: 0.62
enable_tfa: true
enable_medtsllm: true
ollama_model_name: "llama3:8b"
log_level: "INFO"
```

Or run without config — defaults are used from `EcosystemConfig` dataclass.

### Log files not being written

1. Verify `logs/` directory exists and is writable
2. Check `configure_logging(log_dir="logs", level="INFO")` at module load
3. On Windows, check that the path doesn't exceed MAX_PATH (260 chars)

---

## Quick Diagnostic Commands

```bash
# Check Python environment
python -c "import numpy; import pandas; import sklearn; print('OK')"

# Check Ollama
curl http://127.0.0.1:11434/api/tags

# Run format check on recent logs
python -c "
from utils.format_checker import run_startup_check
report = run_startup_check('logs')
print(report)
"

# Run all TFA-specific tests
python -m pytest tests/test_tfa_*.py -v --tb=short

# Generate test report
python -c "
from tests.test_tfa_connectivity import TestTFAConnectivity
t = TestTFAConnectivity()
t.setup_class()
t.test_basic_forecast_returns_valid_structure()
print('Basic TFA test: PASS')
"
```

---

*Last updated: 2026-05-27*
