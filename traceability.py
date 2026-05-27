"""Decision-chain persistence helpers."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_decision_trace(trace: Dict[str, Any], trace_dir: str | Path = "logs/traces") -> Path:
    """Persist a complete decision chain for later responsibility tracing."""
    directory = Path(trace_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"decision_trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
