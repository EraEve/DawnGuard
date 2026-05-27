"""Audit logging for access and decision trace events."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def append_audit_event(event: str, payload: Dict[str, Any], path: str | Path = "logs/audit.log") -> None:
    """Append an audit event as one JSON line."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), "event": event, "payload": payload}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
