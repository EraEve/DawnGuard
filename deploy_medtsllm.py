#!/usr/bin/env python3
"""
MedTsLLM 本地部署与验证脚本
============================
基于 flixpar/med-ts-llm (MLHC 2024) 架构，使用 BioBERT 作为 LLM backbone，
通过 patch reprogramming 层将临床时序信号映射到 LLM 嵌入空间进行联合推理。

用法：
    python deploy_medtsllm.py                  # 完整部署验证
    python deploy_medtsllm.py --check-only     # 仅硬件检测
    python deploy_medtsllm.py --verify-only    # 仅验证 MedTsLLM 适配器
    python deploy_medtsllm.py --test-run       # 运行冒烟测试

与 MAPFM 集成：TemporalForeseeingAgent.forecast() 已自动使用 MedTsLLMAdapter
（通过 EcosystemConfig.enable_medtsllm = True 控制）。

论文: https://arxiv.org/abs/2408.07773
源码: https://github.com/flixpar/med-ts-llm
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix Windows GBK console encoding for emoji-rich output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent
LLM_BACKBONE = "dmis-lab/biobert-v1.1"
ADAPTER_FILE = PROJECT_ROOT / "medtsllm_adapter.py"


# ═══════════════════════════════════════════════════════════════
# Hardware Detection
# ═══════════════════════════════════════════════════════════════

def detect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "ram_gb": 0.0,
        "gpu": {"available": False, "name": "N/A", "vram_gb": 0},
    }
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        info["ram_gb"] = _detect_ram_fallback()
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"]["available"] = True
            info["gpu"]["name"] = torch.cuda.get_device_name(0)
            info["gpu"]["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1
            )
    except ImportError:
        pass
    return info


def _detect_ram_fallback() -> float:
    try:
        if platform.system() == "Windows":
            import ctypes
            kernel32 = ctypes.windll.kernel32

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                ]

            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return round(m.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def print_hardware_report(hw: Dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("  MedTsLLM Hardware Detection Report")
    print("=" * 64)
    print(f"  OS:              {hw['os']} {hw['os_version']}")
    print(f"  Python:          {hw['python_version'].split()[0]}")
    print(f"  CPU cores:       {hw['cpu_count']}")
    print(f"  RAM:             {hw['ram_gb']:.1f} GB")
    print(f"  GPU available:   {'Yes' if hw['gpu']['available'] else 'No'}")
    if hw["gpu"]["available"]:
        print(f"  GPU:             {hw['gpu']['name']} ({hw['gpu']['vram_gb']:.1f} GB VRAM)")
    print(f"  LLM backbone:    {LLM_BACKBONE} (CPU-compatible, 108M params)")
    print("=" * 64)


# ═══════════════════════════════════════════════════════════════
# Dependency Checks
# ═══════════════════════════════════════════════════════════════

def check_dependencies() -> Dict[str, bool]:
    results = {}
    deps = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "numpy": "NumPy",
    }
    for module, label in deps.items():
        try:
            __import__(module)
            results[label] = True
        except ImportError:
            results[label] = False
    return results


def check_biobert_cached() -> bool:
    """Check if BioBERT is available in HuggingFace cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    biobert_dir = cache_dir / "models--dmis-lab--biobert-v1.1"
    if biobert_dir.exists():
        snapshots = biobert_dir / "snapshots"
        if snapshots.exists():
            for snap in snapshots.iterdir():
                if (snap / "config.json").exists():
                    return True
    return False


def check_adapter_file() -> bool:
    return ADAPTER_FILE.exists()


# ═══════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════

def run_verification() -> Dict[str, Any]:
    print("\n" + "=" * 64)
    print("  MedTsLLM Adapter Verification")
    print("=" * 64)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": {},
    }

    # Check 1: Dependencies
    print("\n  [1/5] Checking dependencies...")
    deps = check_dependencies()
    for label, ok in deps.items():
        status = "PASS" if ok else "FAIL"
        print(f"    {status}: {label}")
    results["checks"]["dependencies"] = all(deps.values())

    # Check 2: BioBERT cache
    print("\n  [2/5] Checking BioBERT cache...")
    biobert_ok = check_biobert_cached()
    print(f"    {'PASS' if biobert_ok else 'FAIL'}: {LLM_BACKBONE}")
    results["checks"]["biobert_cached"] = biobert_ok

    # Check 3: Adapter file
    print("\n  [3/5] Checking adapter file...")
    adapter_ok = check_adapter_file()
    print(f"    {'PASS' if adapter_ok else 'FAIL'}: medtsllm_adapter.py")
    results["checks"]["adapter_file"] = adapter_ok

    # Check 4: Load adapter and run inference
    print("\n  [4/5] Loading MedTsLLMAdapter and running inference...")
    try:
        from medtsllm_adapter import MedTsLLMAdapter
        adapter = MedTsLLMAdapter()
        result = adapter.forecast(
            query="Patient with hypertension BP 155/95 mmHg, bedridden 3 days",
            history=[0.01 * i for i in range(72)],
        )
        short_risk = result["short_term"]["risk_probability"]
        mid_risk = result["mid_term"]["risk_probability"]
        long_risk = result["long_term"]["risk_probability"]
        engine = result.get("engine", "unknown")
        print(f"    PASS: short={short_risk:.4f}, mid={mid_risk:.4f}, long={long_risk:.4f}")
        print(f"    Engine: {engine}")
        results["checks"]["inference"] = True
        results["sample_output"] = {
            "short_term_risk": short_risk,
            "mid_term_risk": mid_risk,
            "long_term_risk": long_risk,
            "engine": engine,
        }
    except Exception as e:
        print(f"    FAIL: {e}")
        results["checks"]["inference"] = False

    # Check 5: MAPFM integration
    print("\n  [5/5] Checking MAPFM integration...")
    try:
        import mapfm_ecosystem_repaired as m
        config = m.EcosystemConfig()
        assert config.enable_medtsllm, "enable_medtsllm should be True"
        tfa = m.TemporalForeseeingAgent(config)
        tfa_result = tfa.forecast("test query")
        engine = tfa_result.get("engine", "heuristic")
        assert "short_term" in tfa_result, "Missing short_term"
        assert "mid_term" in tfa_result, "Missing mid_term"
        assert "long_term" in tfa_result, "Missing long_term"
        print(f"    PASS: TFA forecast OK, engine={engine}")
        results["checks"]["mapfm_integration"] = True
    except Exception as e:
        print(f"    FAIL: {e}")
        results["checks"]["mapfm_integration"] = False

    # Summary
    all_pass = all(results["checks"].values())
    print("\n" + "-" * 64)
    print(f"  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    for check, ok in results["checks"].items():
        status = "PASS" if ok else "FAIL"
        print(f"    {status}: {check}")
    print("-" * 64)

    results["all_pass"] = all_pass
    return results


def run_smoke_tests() -> Dict[str, Any]:
    """Run a set of smoke tests with varied clinical scenarios."""
    print("\n" + "=" * 64)
    print("  MedTsLLM Smoke Tests")
    print("=" * 64)

    from medtsllm_adapter import MedTsLLMAdapter
    adapter = MedTsLLMAdapter()

    test_cases = [
        {
            "name": "Hypertension + Bed Rest",
            "query": "Patient with hypertension BP 160/95, bedridden for 5 days, pressure ulcer stage 2",
            "history": [0.02 * (i % 24) for i in range(72)],
            "expect": "elevated long-term risk due to pressure ulcer",
        },
        {
            "name": "Hypoglycemia Emergency",
            "query": "Patient with diabetes type 2, hypoglycemia event, blood sugar 45 mg/dL, syncope",
            "history": [-0.05 + 0.01 * i for i in range(72)],
            "expect": "elevated short-term risk due to acute hypoglycemia",
        },
        {
            "name": "Stable Post-Op",
            "query": "Patient post-appendectomy day 3, vitals stable, ambulating well",
            "history": [0.0] * 72,
            "expect": "low risk across all windows",
        },
        {
            "name": "Heart Failure + Infection",
            "query": "Patient with CHF, possible pneumonia, fever 38.5C, crackles bilateral",
            "history": [0.03 * (i // 12) for i in range(72)],
            "expect": "elevated mid-term risk with infection",
        },
    ]

    results = []
    for tc in test_cases:
        print(f"\n  [{tc['name']}]")
        print(f"    Query: {tc['query'][:80]}...")
        try:
            result = adapter.forecast(query=tc["query"], history=tc["history"])
            s = result["short_term"]["risk_probability"]
            m = result["mid_term"]["risk_probability"]
            l = result["long_term"]["risk_probability"]
            print(f"    Short(24h)={s:.4f}  Mid(30d)={m:.4f}  Long(12m)={l:.4f}")
            print(f"    Expected: {tc['expect']}")
            results.append({"name": tc["name"], "short": s, "mid": m, "long": l, "ok": True})
        except Exception as e:
            print(f"    FAIL: {e}")
            results.append({"name": tc["name"], "ok": False, "error": str(e)})

    n_ok = sum(1 for r in results if r["ok"])
    print(f"\n  Smoke tests: {n_ok}/{len(results)} passed")
    return {"smoke_tests": results, "pass_rate": f"{n_ok}/{len(results)}"}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MedTsLLM Local Deployment & Verification (BioBERT backbone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deploy_medtsllm.py                  # Full deployment verification
  python deploy_medtsllm.py --check-only     # Hardware check only
  python deploy_medtsllm.py --verify-only    # Verification only
  python deploy_medtsllm.py --test-run       # Smoke tests only
""",
    )
    parser.add_argument("--check-only", action="store_true", help="Only run hardware detection")
    parser.add_argument("--verify-only", action="store_true", help="Only run adapter verification")
    parser.add_argument("--test-run", action="store_true", help="Run smoke tests")
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("  MedTsLLM Local Deployment Tool")
    print("  Architecture: flixpar/med-ts-llm (MLHC 2024)")
    print(f"  LLM Backbone: {LLM_BACKBONE} (108M params, CPU-compatible)")
    print("  MAPFM Integration: TemporalForeseeingAgent.forecast()")
    print("=" * 64)

    # ── Hardware detection ──
    hw = detect_hardware()
    print_hardware_report(hw)

    if args.check_only:
        return

    # ── Verification ──
    results = run_verification()

    # ── Smoke tests ──
    if args.test_run or (not args.verify_only):
        if results["checks"].get("inference", False):
            smoke_results = run_smoke_tests()
            results.update(smoke_results)

    # ── Save report ──
    if not args.check_only:
        report_path = PROJECT_ROOT / "medtsllm_deployment_report.json"
        report_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  Deployment report saved: {report_path}")

    print("\n  MedTsLLM deployment check complete.\n")


if __name__ == "__main__":
    main()
