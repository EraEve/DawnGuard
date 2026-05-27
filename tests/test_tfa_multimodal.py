"""
TFA Multimodal Input Test
=========================
Validates TFA module handling of various multimodal input combinations.
All test data is synthetic — zero training data contamination.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from mapfm_ecosystem_repaired import EcosystemConfig, TemporalForeseeingAgent


class TestTFAMultimodal:
    """Multimodal input handling tests for TFA."""

    @classmethod
    def setup_class(cls):
        cls.config = EcosystemConfig()
        cls.tfa = TemporalForeseeingAgent(cls.config)
        cls.rng = np.random.default_rng(42)
        cls.base_query = "Patient reports worsening condition over the past week"

    def _make_history(self, n: int = 72) -> list:
        return self.rng.normal(0, 1, n).tolist()

    def _make_vitals(self) -> dict:
        return {
            "heart_rate": float(self.rng.integers(60, 110)),
            "systolic_bp": float(self.rng.integers(90, 180)),
            "diastolic_bp": float(self.rng.integers(60, 110)),
            "spo2": float(self.rng.uniform(88, 100)),
            "temperature": float(self.rng.uniform(36.0, 39.5)),
            "respiratory_rate": float(self.rng.integers(12, 28)),
        }

    def _make_lab_results(self) -> dict:
        return {
            "wbc": float(self.rng.uniform(3.5, 15.0)),
            "crp": float(self.rng.uniform(0.5, 150.0)),
            "creatinine": float(self.rng.uniform(0.5, 3.5)),
            "glucose": float(self.rng.uniform(3.5, 20.0)),
            "hemoglobin": float(self.rng.uniform(7.0, 18.0)),
            "platelets": float(self.rng.uniform(100, 500)),
        }

    def _make_clinical_text(self) -> str:
        return (
            "65-year-old male with history of hypertension and type 2 diabetes. "
            "Presents with progressive dyspnea on exertion and bilateral lower extremity edema. "
            "Echocardiogram shows reduced ejection fraction of 35%."
        )

    # ── Test 1: Vital signs only ──
    def test_vitals_only(self):
        result = self.tfa.forecast(
            query=self.base_query,
            history=self._make_history(),
            authoritative_signal=self._make_vitals(),
        )
        assert "risk_level" in result
        assert 0 <= result.get("risk_score", 0) <= 1

    # ── Test 2: Lab results only ──
    def test_lab_results_only(self):
        result = self.tfa.forecast(
            query=self.base_query,
            history=self._make_history(),
            authoritative_signal=self._make_lab_results(),
        )
        assert "risk_level" in result

    # ── Test 3: Clinical text only ──
    def test_clinical_text_only(self):
        result = self.tfa.forecast(
            query=self._make_clinical_text(),
            history=self._make_history(),
        )
        assert "risk_level" in result
        assert "primary_risk_factors" in result

    # ── Test 4: Vitals + Lab results ──
    def test_vitals_plus_labs(self):
        signal = {**self._make_vitals(), **self._make_lab_results()}
        result = self.tfa.forecast(
            query=self.base_query, history=self._make_history(),
            authoritative_signal=signal,
        )
        assert "risk_level" in result

    # ── Test 5: Vitals + Clinical text ──
    def test_vitals_plus_clinical(self):
        result = self.tfa.forecast(
            query=self._make_clinical_text(), history=self._make_history(),
            authoritative_signal=self._make_vitals(),
        )
        assert "risk_level" in result

    # ── Test 6: Lab + Clinical text ──
    def test_labs_plus_clinical(self):
        result = self.tfa.forecast(
            query=self._make_clinical_text(), history=self._make_history(),
            authoritative_signal=self._make_lab_results(),
        )
        assert "risk_level" in result

    # ── Test 7: All three data types ──
    def test_all_three_types(self):
        signal = {**self._make_vitals(), **self._make_lab_results(),
                  "posture_instability": 0.3, "activity_reduction": 0.4, "bedrest_hours": 8.0}
        result = self.tfa.forecast(
            query=self._make_clinical_text(), history=self._make_history(),
            authoritative_signal=signal,
        )
        assert "risk_level" in result
        assert len(result.get("primary_risk_factors", [])) >= 1

    # ── Test 8: Missing values (None history) ──
    def test_none_history(self):
        result = self.tfa.forecast(query=self.base_query, history=None)
        assert "risk_level" in result
        short = float(result["short_term"]["risk_probability"])
        assert 0 <= short <= 1

    # ── Test 9: Empty data ──
    def test_empty_signal(self):
        result = self.tfa.forecast(
            query=self.base_query,
            history=[],
            authoritative_signal={},
        )
        assert "risk_level" in result

    # ── Test 10: Anomalous values ──
    def test_anomalous_values(self):
        extreme_signal = {
            "heart_rate": 200.0, "systolic_bp": 250.0,
            "spo2": 60.0, "temperature": 42.0,
        }
        result = self.tfa.forecast(
            query=self.base_query, history=self._make_history(),
            authoritative_signal=extreme_signal,
        )
        risk = result.get("risk_level", "low")
        assert risk in ("high", "critical"), \
            f"Anomalous vitals should produce elevated risk, got: {risk}"


if __name__ == "__main__":
    print("=" * 60)
    print("TFA Multimodal Input Test Suite")
    print("=" * 60)
    tester = TestTFAMultimodal()
    tester.setup_class()
    tests = [
        ("Vitals only", tester.test_vitals_only),
        ("Lab results only", tester.test_lab_results_only),
        ("Clinical text only", tester.test_clinical_text_only),
        ("Vitals + Labs", tester.test_vitals_plus_labs),
        ("Vitals + Clinical", tester.test_vitals_plus_clinical),
        ("Labs + Clinical", tester.test_labs_plus_clinical),
        ("All three types", tester.test_all_three_types),
        ("None history", tester.test_none_history),
        ("Empty data", tester.test_empty_signal),
        ("Anomalous values", tester.test_anomalous_values),
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
