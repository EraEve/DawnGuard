"""
System health monitoring and degradation management for MAPFM HMAE.

Monitors Ollama service, MedTsLLM model, GPU memory, system memory, and disk space.
Implements a three-tier degradation mechanism to keep the system running under
resource pressure.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional GPU monitoring
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Degradation levels
# ---------------------------------------------------------------------------

class DegradationLevel(Enum):
    """System degradation tier."""
    NORMAL = auto()        # Full capability
    LEVEL_1 = auto()       # GPU memory low → 4-bit quantized model
    LEVEL_2 = auto()       # MedTsLLM unavailable → heuristic fallback
    LEVEL_3 = auto()       # Ollama unavailable → pure rule-based mode


@dataclass
class SystemHealth:
    """Snapshot of system health at a point in time."""
    timestamp: datetime = field(default_factory=datetime.now)
    # Ollama
    ollama_healthy: bool = True
    ollama_latency_ms: float = 0.0
    ollama_error: str = ""
    # MedTsLLM
    medtsllm_loaded: bool = False
    medtsllm_model_path: str = ""
    medtsllm_error: str = ""
    # GPU
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_memory_ratio: float = 0.0
    # System
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_ratio: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    # Derived
    degradation_level: DegradationLevel = DegradationLevel.NORMAL
    degradation_reason: str = ""


# ---------------------------------------------------------------------------
# System monitor
# ---------------------------------------------------------------------------

class SystemMonitor:
    """Periodic system health monitoring with configurable thresholds.

    Usage:
        monitor = SystemMonitor()
        health = monitor.check()
        print(health.degradation_level)
    """

    # Thresholds for degradation
    GPU_MEMORY_WARN_RATIO = 0.90       # Level-1 when GPU memory >90% used
    DISK_MIN_FREE_GB = 5.0             # Warn when free disk <5GB
    MEMORY_WARN_RATIO = 0.95           # Warn when system memory >95% used
    OLLAMA_HEALTH_URL = "http://127.0.0.1:11434/api/tags"
    OLLAMA_TIMEOUT_SEC = 5.0

    def __init__(
        self,
        ollama_base_url: str = "http://127.0.0.1:11434",
        medtsllm_model_dir: str = "models/MedTsLLM-v1.5-multimodal",
        check_interval_sec: float = 30.0,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.medtsllm_model_dir = medtsllm_model_dir
        self.check_interval_sec = check_interval_sec
        self._lock = threading.Lock()
        self._last_health: Optional[SystemHealth] = None
        self._health_history: List[SystemHealth] = []
        self._degradation_callbacks: Dict[DegradationLevel, List[Callable[[SystemHealth], None]]] = {
            level: [] for level in DegradationLevel
        }
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ----- Core health check -----

    def check(self) -> SystemHealth:
        """Perform a full system health check and return a snapshot."""
        health = SystemHealth()

        # ── Ollama health ──
        self._check_ollama(health)

        # ── MedTsLLM health ──
        self._check_medtsllm(health)

        # ── GPU health ──
        self._check_gpu(health)

        # ── System resources ──
        self._check_system_resources(health)

        # ── Determine degradation level ──
        self._compute_degradation(health)

        with self._lock:
            self._last_health = health
            self._health_history.append(health)
            if len(self._health_history) > 200:
                self._health_history = self._health_history[-100:]

        # Fire callbacks if degradation changed
        self._fire_degradation_callbacks(health)

        return health

    # ----- Degradation management -----

    @property
    def current_level(self) -> DegradationLevel:
        with self._lock:
            if self._last_health is None:
                return DegradationLevel.NORMAL
            return self._last_health.degradation_level

    @property
    def is_degraded(self) -> bool:
        return self.current_level != DegradationLevel.NORMAL

    def get_degradation_config(self) -> Dict[str, Any]:
        """Return the recommended config adjustments for the current degradation level."""
        level = self.current_level
        if level == DegradationLevel.NORMAL:
            return {"use_4bit": False, "disable_medtsllm": False, "disable_ollama": False}
        elif level == DegradationLevel.LEVEL_1:
            return {"use_4bit": True, "disable_medtsllm": False, "disable_ollama": False}
        elif level == DegradationLevel.LEVEL_2:
            return {"use_4bit": True, "disable_medtsllm": True, "disable_ollama": False}
        else:  # LEVEL_3
            return {"use_4bit": True, "disable_medtsllm": True, "disable_ollama": True}

    def get_user_message(self) -> str:
        """Return a user-facing message explaining the current degradation state."""
        level = self.current_level
        if level == DegradationLevel.NORMAL:
            return ""
        with self._lock:
            reason = self._last_health.degradation_reason if self._last_health else ""
        messages = {
            DegradationLevel.LEVEL_1: (
                "GPU显存不足，系统已自动切换到4-bit量化模型。"
                "诊断精度可能略有下降，但功能不受影响。"
            ),
            DegradationLevel.LEVEL_2: (
                "MedTsLLM多模态模型不可用，系统已切换到启发式时序预测模式。"
                "风险预测精度降低，建议稍后重启MedTsLLM服务。"
            ),
            DegradationLevel.LEVEL_3: (
                "Ollama服务不可用，系统已切换到纯规则诊断模式。"
                "诊断置信度可能降低，建议检查Ollama服务状态。"
            ),
        }
        msg = messages.get(level, "")
        if reason:
            msg = f"{msg} (原因: {reason})"
        return msg

    def on_degradation(
        self, level: DegradationLevel, callback: Callable[[SystemHealth], None]
    ) -> None:
        """Register a callback to be invoked when a specific degradation level is entered."""
        self._degradation_callbacks[level].append(callback)

    # ----- Background monitoring -----

    def start_background_monitoring(self) -> None:
        """Start periodic health checks in a background thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="system-monitor"
        )
        self._monitor_thread.start()

    def stop_background_monitoring(self) -> None:
        """Stop the background health check thread."""
        self._stop_event.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    # ----- Health history -----

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent health snapshots as serializable dicts."""
        with self._lock:
            recent = self._health_history[-limit:]
        return [
            {
                "timestamp": h.timestamp.isoformat(),
                "ollama_healthy": h.ollama_healthy,
                "medtsllm_loaded": h.medtsllm_loaded,
                "gpu_memory_ratio": h.gpu_memory_ratio,
                "memory_ratio": h.memory_ratio,
                "degradation_level": h.degradation_level.name,
                "degradation_reason": h.degradation_reason,
            }
            for h in recent
        ]

    # ----- Internal checks -----

    def _check_ollama(self, health: SystemHealth) -> None:
        if not _HAS_REQUESTS:
            health.ollama_healthy = False
            health.ollama_error = "requests library not installed"
            return
        try:
            started = time.perf_counter()
            resp = requests.get(
                f"{self.ollama_base_url}/api/tags",
                timeout=self.OLLAMA_TIMEOUT_SEC,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            health.ollama_latency_ms = round(elapsed, 1)
            health.ollama_healthy = resp.status_code == 200
            if not health.ollama_healthy:
                health.ollama_error = f"HTTP {resp.status_code}"
        except requests.ConnectionError:
            health.ollama_healthy = False
            health.ollama_error = "Connection refused — Ollama not running"
        except requests.Timeout:
            health.ollama_healthy = False
            health.ollama_error = f"Timeout after {self.OLLAMA_TIMEOUT_SEC}s"
        except Exception as exc:
            health.ollama_healthy = False
            health.ollama_error = str(exc)[:200]

    def _check_medtsllm(self, health: SystemHealth) -> None:
        model_dir = Path(self.medtsllm_model_dir)
        if not model_dir.exists():
            health.medtsllm_loaded = False
            health.medtsllm_error = f"Model directory not found: {self.medtsllm_model_dir}"
            return
        # Check for essential model files (BioBERT config or pytorch_model.bin)
        essential_indicators = [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model.bin.index.json",
        ]
        found = [f for f in essential_indicators if (model_dir / f).exists()]
        if not found:
            health.medtsllm_loaded = False
            health.medtsllm_error = "Model weights not found in directory"
            return
        health.medtsllm_loaded = True
        health.medtsllm_model_path = str(model_dir.resolve())

    def _check_gpu(self, health: SystemHealth) -> None:
        if not _HAS_TORCH:
            return
        try:
            health.gpu_available = torch.cuda.is_available()
            if health.gpu_available:
                health.gpu_count = torch.cuda.device_count()
                total_mem = 0
                used_mem = 0
                for i in range(health.gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    total_mem += props.total_memory
                    used = torch.cuda.memory_allocated(i)
                    used_mem += used
                health.gpu_memory_total_mb = total_mem / (1024 * 1024)
                health.gpu_memory_used_mb = used_mem / (1024 * 1024)
                if health.gpu_memory_total_mb > 0:
                    health.gpu_memory_ratio = health.gpu_memory_used_mb / health.gpu_memory_total_mb
        except Exception:
            pass

    def _check_system_resources(self, health: SystemHealth) -> None:
        if _HAS_PSUTIL:
            try:
                health.cpu_percent = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory()
                health.memory_used_mb = mem.used / (1024 * 1024)
                health.memory_total_mb = mem.total / (1024 * 1024)
                health.memory_ratio = mem.percent / 100.0
                disk = shutil.disk_usage(os.getcwd())
                health.disk_free_gb = disk.free / (1024**3)
                health.disk_total_gb = disk.total / (1024**3)
            except Exception:
                pass
        else:
            # Fallback: estimate from available disk
            try:
                disk = shutil.disk_usage(os.getcwd())
                health.disk_free_gb = disk.free / (1024**3)
                health.disk_total_gb = disk.total / (1024**3)
            except Exception:
                pass

    def _compute_degradation(self, health: SystemHealth) -> None:
        """Apply tiered degradation rules (L1 → L2 → L3 in order of severity)."""
        # Level 3: Ollama unavailable → pure rule-based
        if not health.ollama_healthy:
            health.degradation_level = DegradationLevel.LEVEL_3
            health.degradation_reason = (
                f"Ollama unreachable: {health.ollama_error}"
            )
            return

        # Level 2: MedTsLLM not loaded → heuristic
        if not health.medtsllm_loaded:
            health.degradation_level = DegradationLevel.LEVEL_2
            health.degradation_reason = (
                f"MedTsLLM not available: {health.medtsllm_error}"
            )
            return

        # Level 1: GPU memory pressure → 4-bit quant
        if health.gpu_available and health.gpu_memory_ratio > self.GPU_MEMORY_WARN_RATIO:
            health.degradation_level = DegradationLevel.LEVEL_1
            health.degradation_reason = (
                f"GPU memory {health.gpu_memory_ratio:.1%} used "
                f"({health.gpu_memory_used_mb:.0f}/{health.gpu_memory_total_mb:.0f} MB)"
            )
            return

        # Warn about memory pressure (doesn't trigger degradation yet)
        if health.memory_ratio > self.MEMORY_WARN_RATIO:
            health.degradation_reason = (
                f"System memory high: {health.memory_ratio:.1%} "
                f"({health.memory_used_mb:.0f}/{health.memory_total_mb:.0f} MB)"
            )
            # Stays at NORMAL unless other triggers fire

        if health.disk_free_gb < self.DISK_MIN_FREE_GB:
            health.degradation_reason = (
                f"Low disk space: {health.disk_free_gb:.1f} GB free"
            )

    def _fire_degradation_callbacks(self, health: SystemHealth) -> None:
        """Notify registered callbacks of the current degradation level."""
        level = health.degradation_level
        for callback in self._degradation_callbacks.get(level, []):
            try:
                callback(health)
            except Exception:
                pass

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self.check()
            except Exception:
                pass
            self._stop_event.wait(self.check_interval_sec)


# ---------------------------------------------------------------------------
# Degradation-aware wrapper for integration into the ecosystem
# ---------------------------------------------------------------------------

class DegradationAwareWrapper:
    """Wraps a system monitor and provides degradation-aware execution.

    Integrates with the existing DegradationManager's safe_call pattern while
    adding the three-tier degradation logic from SystemMonitor.

    Usage:
        wrapper = DegradationAwareWrapper(system_monitor)
        result = wrapper.degradation_aware_call(
            normal_fn=medtsllm_forecast,
            fallback_fn=heuristic_forecast,
            level_2_fallback_fn=heuristic_forecast,
            level_3_fallback_fn=rule_based_forecast,
        )
    """

    def __init__(self, monitor: SystemMonitor) -> None:
        self.monitor = monitor

    def degradation_aware_call(
        self,
        normal_fn: Callable[..., Any],
        *args: Any,
        fallback_fn: Optional[Callable[..., Any]] = None,
        level_1_fn: Optional[Callable[..., Any]] = None,
        level_2_fn: Optional[Callable[..., Any]] = None,
        level_3_fn: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the appropriate function based on current degradation level.

        Level NORMAL: normal_fn
        Level 1: level_1_fn or normal_fn (caller decides)
        Level 2: level_2_fn or fallback_fn
        Level 3: level_3_fn or fallback_fn
        """
        level = self.monitor.current_level

        if level == DegradationLevel.NORMAL:
            try:
                return normal_fn(*args, **kwargs)
            except Exception as exc:
                if fallback_fn is not None:
                    return fallback_fn(*args, **kwargs)
                raise

        elif level == DegradationLevel.LEVEL_1:
            fn = level_1_fn or normal_fn
            try:
                return fn(*args, **kwargs)
            except Exception:
                if fallback_fn is not None:
                    return fallback_fn(*args, **kwargs)
                raise

        elif level == DegradationLevel.LEVEL_2:
            fn = level_2_fn or fallback_fn
            if fn is None:
                raise RuntimeError("No fallback available for degradation level 2")
            return fn(*args, **kwargs)

        else:  # LEVEL_3
            fn = level_3_fn or fallback_fn
            if fn is None:
                raise RuntimeError("No fallback available for degradation level 3")
            return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Global convenience instance
# ---------------------------------------------------------------------------

# Module-level singleton (lazily initialized)
_global_monitor: Optional[SystemMonitor] = None
_global_lock = threading.Lock()


def get_system_monitor(
    ollama_base_url: str = "http://127.0.0.1:11434",
    medtsllm_model_dir: str = "models/MedTsLLM-v1.5-multimodal",
) -> SystemMonitor:
    """Get or create the global SystemMonitor singleton."""
    global _global_monitor
    with _global_lock:
        if _global_monitor is None:
            _global_monitor = SystemMonitor(
                ollama_base_url=ollama_base_url,
                medtsllm_model_dir=medtsllm_model_dir,
            )
        return _global_monitor
