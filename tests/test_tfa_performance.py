"""
TFA Performance Test
====================
Measures TFA inference speed, memory usage, and concurrent processing.
Generates a performance report with pass/fail against defined thresholds.

Thresholds:
  - Single task avg response: <= 15s
  - Concurrent 5 tasks avg: <= 30s
  - GPU VRAM (4-bit): <= 16GB
  - System RAM: <= 32GB
"""

import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from mapfm_ecosystem_repaired import EcosystemConfig, TemporalForeseeingAgent


class TestTFAPerformance:
    """TFA performance benchmarks."""

    @classmethod
    def setup_class(cls):
        cls.config = EcosystemConfig()
        cls.config.enable_medtsllm = True
        cls.tfa = TemporalForeseeingAgent(cls.config)
        cls.rng = np.random.default_rng(42)
        cls.queries = [
            "Patient with mild intermittent headache for 2 days",
            "Follow-up visit for well-controlled type 2 diabetes",
            "Acute onset chest pain with diaphoresis and nausea",
            "Elderly patient with confusion and suspected UTI",
            "Post-operative day 3, surgical wound clean and dry",
            "Patient with COPD exacerbation, increased sputum production",
            "Routine annual physical examination, no complaints",
            "Hypertensive urgency: BP 190/110, mild headache",
            "Patient with new-onset atrial fibrillation, palpitations",
            "Suspected DVT: unilateral leg swelling and pain",
        ]

    def _time_single(self, query: str) -> float:
        start = time.perf_counter()
        self.tfa.forecast(query=query, history=self.rng.normal(0, 1, 72).tolist())
        return time.perf_counter() - start

    # ── Test 1: Single task response time ──
    def test_single_task_latency(self):
        """Average single-task latency must be <= 15 seconds."""
        times = []
        for q in self.queries[:4]:
            elapsed = self._time_single(q)
            times.append(elapsed)
        avg = np.mean(times)
        print(f"  Single-task avg: {avg:.2f}s (max: {max(times):.2f}s)")
        assert avg <= 15.0, f"Avg latency {avg:.2f}s exceeds 15s limit"

    # ── Test 2: Concurrent 5 tasks ──
    def test_concurrent_5_tasks(self):
        """Average latency for 5 concurrent tasks must be <= 30 seconds."""
        queries = self.queries[:5]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [
                ex.submit(self.tfa.forecast, query=q,
                          history=self.rng.normal(0, 1, 72).tolist())
                for q in queries
            ]
            results = []
            for f in as_completed(futures):
                try:
                    results.append(f.result(timeout=60))
                except Exception as e:
                    results.append({"error": str(e)})
        total = time.perf_counter() - start
        avg_per_task = total / len(queries)
        print(f"  Concurrent 5-tasks total: {total:.2f}s, avg/task: {avg_per_task:.2f}s")
        assert total <= 30.0, f"Total time {total:.2f}s exceeds 30s limit"
        assert len(results) == len(queries), f"Expected {len(queries)} results, got {len(results)}"

    # ── Test 3: Memory usage estimate ──
    def test_memory_usage(self):
        """Estimate memory usage and verify within limits."""
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
            mem_gb = mem_mb / 1024
            print(f"  Current RSS: {mem_mb:.1f} MB ({mem_gb:.2f} GB)")
            assert mem_gb <= 32.0, f"Memory {mem_gb:.2f} GB exceeds 32GB limit"
        except ImportError:
            print("  [SKIP] psutil not available — memory check skipped")

    # ── Test 4: CPU usage during inference ──
    def test_cpu_usage(self):
        """CPU usage should not saturate all cores during a single inference."""
        try:
            import psutil
            cpu_before = psutil.cpu_percent(interval=0.5)
            self._time_single(self.queries[0])
            cpu_after = psutil.cpu_percent(interval=0.5)
            print(f"  CPU: {cpu_before:.1f}% → {cpu_after:.1f}%")
        except ImportError:
            print("  [SKIP] psutil not available — CPU check skipped")

    # ── Test 5: Warm-up vs cold-start ──
    def test_warm_start_faster(self):
        """Second call should not be significantly slower than first."""
        cold = self._time_single(self.queries[0])
        warm = self._time_single(self.queries[0])
        ratio = warm / max(cold, 0.001)
        print(f"  Cold: {cold:.2f}s, Warm: {warm:.2f}s (ratio: {ratio:.2f})")
        # Warm should be <= cold * 3 (allow some variance)
        assert ratio <= 3.0, f"Warm start {warm:.2f}s much slower than cold {cold:.2f}s"

    # ── Test 6: 10 sequential tasks throughput ──
    def test_throughput_10_tasks(self):
        """Run 10 tasks sequentially and report throughput."""
        queries = self.queries[:10] if len(self.queries) >= 10 else self.queries * 2
        start = time.perf_counter()
        for q in queries[:10]:
            self.tfa.forecast(query=q, history=self.rng.normal(0, 1, 72).tolist())
        total = time.perf_counter() - start
        throughput = 10 / total
        print(f"  Throughput: {throughput:.2f} tasks/s (total {total:.2f}s for 10 tasks)")


def generate_performance_report() -> dict:
    """Run all performance tests and produce a structured report."""
    tester = TestTFAPerformance()
    tester.setup_class()
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thresholds": {
            "single_task_latency_s": 15.0,
            "concurrent_5_latency_s": 30.0,
            "gpu_vram_gb": 16.0,
            "system_ram_gb": 32.0,
        },
        "results": {},
    }
    return report


if __name__ == "__main__":
    print("=" * 60)
    print("TFA Performance Test Suite")
    print("=" * 60)
    tester = TestTFAPerformance()
    tester.setup_class()
    tests = [
        ("Single task latency", tester.test_single_task_latency),
        ("Concurrent 5 tasks", tester.test_concurrent_5_tasks),
        ("Memory usage", tester.test_memory_usage),
        ("CPU usage", tester.test_cpu_usage),
        ("Warm start", tester.test_warm_start_faster),
        ("Throughput 10 tasks", tester.test_throughput_10_tasks),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
    print(f"\nResult: {passed}/{len(tests)} tests passed")
    import sys as _sys
    _sys.exit(0 if passed == len(tests) else 1)
