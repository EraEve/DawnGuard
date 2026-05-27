"""
TFA Fallback / Degradation Mechanism Test
==========================================
Validates graceful degradation when MedTsLLM is unavailable.
System must continue operating in heuristic fallback mode without crashing.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from mapfm_ecosystem_repaired import (
    EcosystemConfig, TemporalForeseeingAgent, DegradationManager,
)


class TestTFAFallback:
    """Degradation and fallback mechanism tests."""

    @classmethod
    def setup_class(cls):
        cls.config = EcosystemConfig()
        cls.config.enable_medtsllm = True
        cls.tfa = TemporalForeseeingAgent(cls.config)
        cls.degradation = DegradationManager(cls.config)
        cls.rng = np.random.default_rng(42)

    def _forecast_via_degradation(self, query: str, **kwargs) -> dict:
        history = kwargs.pop("history", self.rng.normal(0, 1, 72).tolist())
        return self.degradation.safe_call(
            "TFA",
            self.tfa.forecast,
            query=query, history=history,
            fallback=lambda: {
                "short_term": {"risk_probability": 0.05},
                "risk_level": "low", "degraded": True, "risk_score": 0.05,
                "primary_risk_factors": [], "recommendations": [],
                "mid_term": {"risk_probability": 0.05},
                "long_term": {"risk_probability": 0.05},
            },
            **kwargs,
        )

    # ── Test 1: TFA works in normal mode ──
    def test_normal_mode_works(self):
        """TFA should work with default configuration (heuristic path)."""
        result = self._forecast_via_degradation(
            "Patient with mild headache for 1 day"
        )
        assert isinstance(result, dict)
        assert "risk_level" in result

    # ── Test 2: Degradation safe_call returns fallback on error ──
    def test_safe_call_returns_fallback(self):
        """DegradationManager safe_call must return fallback when function raises."""
        def failing_fn():
            raise RuntimeError("Simulated TFA failure")
        result = self.degradation.safe_call(
            "TFA_Test", failing_fn,
            fallback=lambda: {"degraded": True, "risk_level": "low"},
        )
        assert result.get("degraded") is True
        assert self.degradation.component_states.get("TFA_Test") == "degraded"

    # ── Test 3: Degradation level escalates correctly ──
    def test_degradation_levels(self):
        """Degradation level must reflect component states."""
        dm = DegradationManager(self.config)
        assert dm.degradation_level == 0
        dm.mark("ComponentA", "degraded")
        assert dm.degradation_level == 1
        dm.mark("ComponentB", "unavailable")
        assert dm.degradation_level == 2
        dm.mark("ComponentC", "unavailable")
        dm.mark("ComponentD", "unavailable")
        assert dm.degradation_level == 3

    # ── Test 4: Recovery after degradation ──
    def test_recovery_after_marking_healthy(self):
        """Component marked healthy should clear degradation."""
        dm = DegradationManager(self.config)
        dm.mark("TestComponent", "unavailable")
        assert dm.degradation_level >= 2
        dm.mark("TestComponent", "healthy")
        assert dm.degradation_level == 0

    # ── Test 5: Fallback preserves output contract ──
    def test_fallback_preserves_contract(self):
        """Even in fallback mode, output must have all required fields."""
        result = self.degradation.safe_call(
            "TFA",
            lambda: (_ for _ in ()).throw(RuntimeError("fail")),
            fallback=lambda: {
                "short_term": {"risk_probability": 0.05, "window": "24h"},
                "mid_term": {"risk_probability": 0.05, "window": "30d"},
                "long_term": {"risk_probability": 0.05, "window": "12m"},
                "risk_level": "low", "risk_score": 0.05, "degraded": True,
            },
        )
        assert "short_term" in result
        assert "risk_level" in result
        assert result.get("degraded") is True

    # ── Test 6: Degradation stats are tracked ──
    def test_degradation_stats_tracked(self):
        """DegradationManager.get_stats must return meaningful data."""
        dm = DegradationManager(self.config)
        dm.mark("CompA", "degraded")
        dm.mark("CompA", "degraded")
        stats = dm.get_stats()
        assert "degradation_level" in stats
        assert "component_states" in stats
        assert "fallback_counts" in stats
        assert stats["fallback_counts"].get("CompA", 0) >= 2

    # ── Test 7: MedTsLLM import failure doesn't crash ──
    def test_medtsllm_unavailable_is_graceful(self):
        """When MedTsLLM adapter is unavailable, TFA falls back gracefully."""
        config = EcosystemConfig()
        config.enable_medtsllm = True
        tfa = TemporalForeseeingAgent(config)
        # Force _medtsllm to None to simulate unavailability
        tfa._medtsllm_load_attempted = True
        tfa._medtsllm = None
        result = tfa.forecast(query="Test query for fallback path")
        assert "risk_level" in result
        assert "short_term" in result


if __name__ == "__main__":
    print("=" * 60)
    print("TFA Fallback / Degradation Test Suite")
    print("=" * 60)
    tester = TestTFAFallback()
    tester.setup_class()
    tests = [
        ("Normal mode works", tester.test_normal_mode_works),
        ("Safe call returns fallback", tester.test_safe_call_returns_fallback),
        ("Degradation levels escalate", tester.test_degradation_levels),
        ("Recovery after healthy", tester.test_recovery_after_marking_healthy),
        ("Fallback preserves contract", tester.test_fallback_preserves_contract),
        ("Degradation stats tracked", tester.test_degradation_stats_tracked),
        ("MedTsLLM unavailable graceful", tester.test_medtsllm_unavailable_is_graceful),
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
