"""
HMAE Unified Format Checker
============================
Validates all module log outputs and user-facing reports against the unified
format specification. Ensures 100% backward compatibility with original formats
while verifying new fields are correctly appended.

Usage:
    checker = FormatChecker()
    checker.check_log_line(line, module="DMA")     # -> bool
    checker.check_log_file("logs/app.log")          # -> List[str] (errors)
    checker.check_output(user_output)               # -> bool
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Regex patterns for each module's log format ──
# Original fields MUST come first, new fields appended after.
_LOG_PATTERNS: Dict[str, re.Pattern] = {
    "DMA": re.compile(
        r"^DMA -> prediction=(?P<prediction>.+?) \| "
        r"confidence=(?P<confidence>\d+\.\d{4}) \| "
        r"HITL=(?P<hitl>True|False)"
        r"(?: \| intent=(?P<intent>symptom_query|treatment_query|prognosis_query|prevention_query|screening_check|symptom_inquiry|treatment_inquiry|prognosis|prevention|general_inquiry))?"
        r"(?: \| status=(?P<status>success|low_confidence|error))?"
        r"(?: \| severity=(?P<severity>\w+))?"
        r"$"
    ),
    "RAA": re.compile(
        r"^RAA -> strategy=(?P<strategy>mixed|rerank|adaptive|Adaptive-RAG|Mixed-RAG|Rerank-RAG|Adaptive-RAG:Deep Retrieval|Adaptive-RAG:Fast Retrieval) \| "
        r"Nash=(?P<nash>True|False) \| "
        r"rounds=(?P<rounds>\d+) \| "
        r"verified.relevance=(?P<relevance>\d+\.\d{4})"
        r"(?: \| evidence_level=(?P<evidence_level>\d+\.\d))?"
        r"(?: \| conflicts=(?P<conflicts>\d+))?"
        r"$"
    ),
    "TFA": re.compile(
        r"^TFA -> future 24h deterioration risk=(?P<risk>\d+\.\d{2})%"
        r"(?: \| risk_level=(?P<risk_level>low|medium|high|critical))?"
        r"(?: \| confidence=(?P<confidence>\d+\.\d{4}))?"
        r"(?: \| source=(?P<source>medtsllm|fallback|rules))?"
        r"(?: \| model_version=(?P<model_version>[\w.\-]+))?"
        r"$"
    ),
    "Fusion/Verification": re.compile(
        r"^Fusion/Verification -> input=(?P<input>\d+) \| "
        r"dedup=(?P<dedup>\d+) \| "
        r"conflicts=(?P<conflicts>\d+) \| "
        r"verified=(?P<verified>\d+)/(?P<total>\d+)"
        r"(?: \| moderate_conflicts=(?P<moderate_conflicts>\d+))?"
        r"(?: \| minor_conflicts=(?P<minor_conflicts>\d+))?"
        r"(?: \| status=(?P<status>passed|passed_with_warnings|failed))?"
        r"$"
    ),
    "Consensus": re.compile(
        r"^Consensus -> approved=(?P<approved>True|False) \| "
        r"votes=(?P<votes>\{.*?\}) \| "
        r"required=(?P<required>\d+)"
        r"(?: \| reason=\"(?P<reason>.*?)\")?"
        r"(?: \| hitl_triggered=(?P<hitl_triggered>True|False))?"
        r"(?: \| risk_level=(?P<risk_level>low|medium|high|critical))?"
        r"(?: \| tier=(?P<tier>\w+))?"
        r"$"
    ),
    "HITL": re.compile(
        r"^HITL -> triggered=(?P<triggered>True|False) \| "
        r"reason=(?P<reason>.+?)"
        r"(?: \| task_id=(?P<task_id>\w+))?"
        r"(?: \| intervention_time=(?P<intervention_time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}))?"
        r"(?: \| operator=(?P<operator>\w+))?"
        r"(?: \| priority=(?P<priority>\d+))?"
        r"$"
    ),
}

# Error log pattern
_ERROR_LOG_PATTERN = re.compile(
    r"^\[(?P<level>ERROR|CRITICAL)\] "
    r"\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] "
    r"\[(?P<module>DMA|RAA|TFA|Fusion|Verification|Consensus|HITL|Perception|Privacy|Maintenance|Topology)\] "
    r"- (?P<error_code>E-\d{3}): "
    r"(?P<message>.+?)"
    r"(?:\. Details: (?P<details>.+))?"
    r"$"
)

# Error code mapping
ERROR_CODES: Dict[str, str] = {
    "E-001": "Model invocation failure",
    "E-002": "Data format error",
    "E-003": "File not found",
    "E-004": "Network error",
    "E-005": "Configuration error",
    "E-006": "Authentication failure",
    "E-007": "Resource exhausted",
    "E-008": "Timeout",
    "E-009": "Internal error",
    "E-010": "Validation failure",
}

_FLOAT_4DP = re.compile(r"^\d+\.\d{4}$")
_PERCENT_2DP = re.compile(r"^\d+\.\d{2}%?$")
_ISO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


class FormatCheckResult:
    """Structured result from a format check."""

    def __init__(self) -> None:
        self.valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.line_number: int = 0
        self.raw_line: str = ""

    def add_error(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __bool__(self) -> bool:
        return self.valid

    def summary(self) -> str:
        parts: List[str] = []
        if self.errors:
            parts.append(f"Errors ({len(self.errors)}): " + "; ".join(self.errors))
        if self.warnings:
            parts.append(f"Warnings ({len(self.warnings)}): " + "; ".join(self.warnings))
        return " | ".join(parts) if parts else "OK"


class FormatChecker:
    """Unified format validator for all HMAE module logs and outputs.

    Ensures:
      - Original fields are preserved in exact order
      - New fields are only appended (| key=value)
      - Float precision: 4 decimal places for probabilities, 2 for percentages
      - Boolean values use True/False
      - Error logs follow the unified [LEVEL][TIMESTAMP][MODULE] format
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict
        self._checked_lines: int = 0
        self._failed_lines: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_log_line(self, line: str, module: str) -> bool:
        """Validate a single log line against the module's expected format.

        Args:
            line: The log line to check.
            module: One of DMA, RAA, TFA, Fusion/Verification, Consensus, HITL.

        Returns:
            True if the line matches the expected format.
        """
        result = self._check_line_detailed(line, module)
        self._checked_lines += 1
        if not result.valid:
            self._failed_lines += 1
        return result.valid

    def check_log_file(self, file_path: str) -> List[str]:
        """Check an entire log file and return all non-conforming lines.

        Returns:
            List of error description strings. Empty list means all lines pass.
        """
        path = Path(file_path)
        if not path.exists():
            return [f"File not found: {file_path}"]

        errors: List[str] = []
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, start=1):
                line = raw_line.rstrip("\n").rstrip("\r")
                if not line.strip():
                    continue
                # Detect module from line prefix
                module = self._detect_module(line)
                if module is None:
                    continue  # Not a recognised structured log line
                result = self._check_line_detailed(line, module)
                if not result.valid:
                    errors.append(
                        f"Line {lineno} [{module}]: {result.summary()}  |  raw={line[:200]}"
                    )
        self._checked_lines += 1
        return errors

    def check_output(self, output: str) -> bool:
        """Validate the final user-facing output structure.

        Checks for required sections: diagnosis, disclaimer.
        """
        if not output or not output.strip():
            return False
        required_markers = ["# 诊断结果", "免责声明"]
        for marker in required_markers:
            if marker not in output:
                return False
        if "不构成临床建议" not in output and "不构成医疗建议" not in output:
            return False
        return True

    # ------------------------------------------------------------------
    # Detailed line-level checking
    # ------------------------------------------------------------------

    def _check_line_detailed(self, line: str, module: str) -> FormatCheckResult:
        result = FormatCheckResult()
        result.raw_line = line
        pattern = _LOG_PATTERNS.get(module)
        if pattern is None:
            result.add_error(f"Unknown module: {module}")
            return result

        match = pattern.match(line.strip())
        if not match:
            result.add_error(f"Line does not match {module} format pattern")
            # Try to auto-diagnose the issue
            diagnosis = self._diagnose_mismatch(line, module)
            if diagnosis:
                result.add_error(f"Diagnosis: {diagnosis}")
            return result

        # Validate field values
        self._validate_fields(match.groupdict(), module, result)
        return result

    def _detect_module(self, line: str) -> Optional[str]:
        """Detect which module a log line belongs to by its prefix."""
        prefixes = {
            "DMA ->": "DMA",
            "RAA ->": "RAA",
            "TFA ->": "TFA",
            "Fusion/Verification ->": "Fusion/Verification",
            "Consensus ->": "Consensus",
            "HITL ->": "HITL",
        }
        for prefix, module in prefixes.items():
            if line.startswith(prefix):
                return module
        # Check for error log format
        if re.match(r"^\[(ERROR|CRITICAL)\]", line):
            return "ERROR"
        return None

    # ------------------------------------------------------------------
    # Field validation
    # ------------------------------------------------------------------

    def _validate_fields(self, fields: Dict[str, Optional[str]], module: str, result: FormatCheckResult) -> None:
        """Validate individual field values for type, range, and precision."""
        validators: Dict[str, Any] = {
            "DMA": self._validate_dma_fields,
            "RAA": self._validate_raa_fields,
            "TFA": self._validate_tfa_fields,
            "Fusion/Verification": self._validate_fusion_fields,
            "Consensus": self._validate_consensus_fields,
            "HITL": self._validate_hitl_fields,
        }
        validator = validators.get(module)
        if validator:
            validator(fields, result)

    def _validate_dma_fields(self, f: Dict[str, Optional[str]], r: FormatCheckResult) -> None:
        if f.get("confidence"):
            if not _FLOAT_4DP.match(f["confidence"]):
                r.add_error(f"DMA confidence must be 4 decimal places, got: {f['confidence']}")
            val = float(f["confidence"])
            if not (0.0 <= val <= 1.0):
                r.add_error(f"DMA confidence out of [0,1]: {val}")
        if f.get("intent"):
            valid_intents = {
                "symptom_query", "treatment_query", "prognosis_query",
                "prevention_query", "screening_check", "symptom_inquiry",
                "treatment_inquiry", "prognosis", "prevention", "general_inquiry",
            }
            if f["intent"] not in valid_intents:
                r.add_warning(f"Unknown DMA intent: {f['intent']}")

    def _validate_raa_fields(self, f: Dict[str, Optional[str]], r: FormatCheckResult) -> None:
        if f.get("relevance"):
            if not _FLOAT_4DP.match(f["relevance"]):
                r.add_error(f"RAA relevance must be 4 decimal places, got: {f['relevance']}")
        if f.get("evidence_level"):
            try:
                val = float(f["evidence_level"])
                if not (1.0 <= val <= 4.0):
                    r.add_warning(f"RAA evidence_level out of [1,4]: {val}")
            except ValueError:
                r.add_error(f"RAA evidence_level not a valid float: {f['evidence_level']}")

    def _validate_tfa_fields(self, f: Dict[str, Optional[str]], r: FormatCheckResult) -> None:
        if f.get("risk"):
            if not _PERCENT_2DP.match(f["risk"]):
                r.add_error(f"TFA risk must be 2 decimal places, got: {f['risk']}")
            val = float(f["risk"])
            if not (0.0 <= val <= 100.0):
                r.add_error(f"TFA risk percentage out of [0,100]: {val}")
        if f.get("confidence"):
            if not _FLOAT_4DP.match(f["confidence"]):
                r.add_error(f"TFA confidence must be 4 decimal places: {f['confidence']}")

    def _validate_fusion_fields(self, f: Dict[str, Optional[str]], r: FormatCheckResult) -> None:
        if f.get("input") and f.get("dedup") and f.get("verified") and f.get("total"):
            try:
                inp = int(f["input"])
                dedup = int(f["dedup"])
                ver = int(f["verified"])
                tot = int(f["total"])
                if ver > tot:
                    r.add_error(f"verified ({ver}) > total ({tot})")
                if dedup > inp:
                    r.add_error(f"dedup ({dedup}) > input ({inp})")
            except ValueError:
                r.add_error("Fusion integer fields parse error")

    def _validate_consensus_fields(self, f: Dict[str, Optional[str]], r: FormatCheckResult) -> None:
        if f.get("required"):
            try:
                req = int(f["required"])
                if req < 1 or req > 4:
                    r.add_warning(f"Consensus required votes unusual: {req}")
            except ValueError:
                r.add_error(f"Consensus required not an integer: {f['required']}")

        # Validate votes dict format
        if f.get("votes"):
            votes_str = f["votes"]
            if not (votes_str.startswith("{") and votes_str.endswith("}")):
                r.add_error(f"Consensus votes not a dict literal: {votes_str[:50]}")

    def _validate_hitl_fields(self, f: Dict[str, Optional[str]], r: FormatCheckResult) -> None:
        if f.get("intervention_time"):
            if not _ISO_TIMESTAMP.match(f["intervention_time"]):
                r.add_error(f"HITL intervention_time not ISO 8601: {f['intervention_time']}")

    # ------------------------------------------------------------------
    # Error log checking
    # ------------------------------------------------------------------

    def check_error_log_line(self, line: str) -> FormatCheckResult:
        """Validate a line against the unified error log format."""
        result = FormatCheckResult()
        result.raw_line = line
        match = _ERROR_LOG_PATTERN.match(line.strip())
        if not match:
            result.add_error("Line does not match unified error format")
            diagnosis = self._diagnose_error_mismatch(line)
            if diagnosis:
                result.add_error(f"Diagnosis: {diagnosis}")
            return result
        # Validate error code
        code = match.group("error_code")
        if code not in ERROR_CODES:
            result.add_warning(f"Unknown error code: {code}")
        # Validate timestamp
        ts = match.group("timestamp")
        try:
            datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            result.add_error(f"Invalid timestamp: {ts}")
        return result

    def _diagnose_mismatch(self, line: str, module: str) -> str:
        """Attempt to diagnose why a line doesn't match its expected format."""
        diagnoses: List[str] = []

        # Check for common precision issues
        float_values = re.findall(r"\d+\.\d+", line)
        for val in float_values:
            decimal_part = val.split(".")[1]
            if module in ("DMA", "RAA", "TFA"):
                if "confidence" in line or "relevance" in line:
                    if len(decimal_part) != 4 and module != "TFA":
                        diagnoses.append(f"Precision: {val} should be 4dp")
            if module == "TFA" and "risk=" in line:
                if len(decimal_part) != 2:
                    diagnoses.append(f"TFA risk percentage should be 2dp: {val}")

        # Check for missing fields
        required_original = {
            "DMA": ["prediction=", "confidence=", "HITL="],
            "RAA": ["strategy=", "Nash=", "rounds=", "verified_relevance="],
            "TFA": ["future 24h deterioration risk="],
            "Fusion/Verification": ["input=", "dedup=", "conflicts=", "verified="],
            "Consensus": ["approved=", "votes=", "required="],
            "HITL": ["triggered=", "reason="],
        }
        for req in required_original.get(module, []):
            if req not in line:
                diagnoses.append(f"Missing required original field: {req}")

        return "; ".join(diagnoses) if diagnoses else ""

    def _diagnose_error_mismatch(self, line: str) -> str:
        """Diagnose why an error log line doesn't match."""
        diagnoses: List[str] = []
        if not line.startswith("["):
            diagnoses.append("Missing opening bracket for log level")
        if not re.search(r"\[(ERROR|CRITICAL)\]", line):
            diagnoses.append("Missing or invalid log level")
        if not re.search(r"E-\d{3}", line):
            diagnoses.append("Missing error code (E-NNN)")
        if not re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line):
            diagnoses.append("Missing ISO 8601 timestamp")
        return "; ".join(diagnoses) if diagnoses else ""

    # ------------------------------------------------------------------
    # Auto-correction helpers
    # ------------------------------------------------------------------

    def auto_fix_float_precision(self, line: str, field: str, decimal_places: int) -> str:
        """Fix float precision on a specific field in a log line."""
        pattern = re.compile(rf"({field}=)(\d+\.\d+)")
        match = pattern.search(line)
        if not match:
            return line
        raw_value = float(match.group(2))
        fixed = f"{match.group(1)}{raw_value:.{decimal_places}f}"
        return line[:match.start()] + fixed + line[match.end():]

    def auto_fix_line(self, line: str, module: str) -> Optional[str]:
        """Attempt to automatically fix a non-conforming log line.

        Returns the fixed line, or None if auto-fix is not possible.
        """
        fixed = line
        # Fix float precision for confidence/relevance fields (4dp)
        if module in ("DMA", "RAA", "TFA"):
            for field_pattern, dp in [("confidence", 4), ("relevance", 4), ("verified_relevance", 4)]:
                fixed = self.auto_fix_float_precision(fixed, field_pattern, dp)
        # Fix TFA risk percentage (2dp)
        if module == "TFA":
            fixed = self.auto_fix_float_precision(fixed, "risk", 2)
        if fixed == line:
            return None  # Nothing was fixed
        return fixed

    # ------------------------------------------------------------------
    # Batch report
    # ------------------------------------------------------------------

    def generate_report(self, file_paths: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive format compliance report across log files."""
        total_lines = 0
        total_errors = 0
        per_file: Dict[str, List[str]] = {}

        for fp in file_paths:
            errs = self.check_log_file(fp)
            per_file[fp] = errs
            total_errors += len(errs)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    total_lines += sum(1 for _ in fh)
            except Exception:
                pass

        return {
            "files_checked": len(file_paths),
            "total_lines": total_lines,
            "total_errors": total_errors,
            "compliance_rate": (
                1.0 - total_errors / max(total_lines, 1)
            ),
            "per_file_errors": per_file,
            "generated_at": datetime.now().isoformat(),
        }


# ── Convenience functions ──

def validate_module_log(line: str, module: str) -> bool:
    """Quick one-shot validation of a log line."""
    return FormatChecker().check_log_line(line, module)


def validate_error_log(line: str) -> bool:
    """Quick one-shot validation of an error log line."""
    result = FormatChecker().check_error_log_line(line)
    return result.valid


def format_error_log(
    level: str,
    module: str,
    error_code: str,
    message: str,
    details: str = "",
    timestamp: Optional[str] = None,
) -> str:
    """Format an error log line in the unified format.

    Args:
        level: ERROR or CRITICAL
        module: DMA, RAA, TFA, Fusion, Verification, Consensus, HITL, etc.
        error_code: E-001 through E-010
        message: Short error description
        details: Optional detailed information
        timestamp: ISO 8601 timestamp (default: now)

    Returns:
        Properly formatted error log line.
    """
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if details:
        return f"[{level}] [{ts}] [{module}] - {error_code}: {message}. Details: {details}"
    return f"[{level}] [{ts}] [{module}] - {error_code}: {message}"


# ── Startup self-check ──

def run_startup_check(log_dir: str = "logs") -> Dict[str, Any]:
    """Run format validation on all log files at system startup.

    Called automatically during system initialization to ensure all
    log outputs conform to the unified format specification.
    """
    checker = FormatChecker(strict=True)
    log_path = Path(log_dir)
    if not log_path.exists():
        return {"status": "no_logs", "message": f"Log directory '{log_dir}' not found"}

    log_files = sorted(log_path.glob("*.log"))[-5:]  # Check last 5 log files
    if not log_files:
        return {"status": "no_logs", "message": "No log files found"}

    report = checker.generate_report([str(f) for f in log_files])
    report["status"] = "pass" if report["total_errors"] == 0 else "fail"
    return report
