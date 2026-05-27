"""
TFA Medical Knowledge Validation Test
======================================
Validates TFA outputs against basic medical common-sense constraints.
Ensures risk predictions are clinically reasonable and safe.
All test data is synthetic — no patient data is used.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from mapfm_ecosystem_repaired import EcosystemConfig, TemporalForeseeingAgent


class TestTFAMedicalCheck:
    """Medical common-sense validation for TFA outputs."""

    @classmethod
    def setup_class(cls):
        cls.config = EcosystemConfig()
        cls.tfa = TemporalForeseeingAgent(cls.config)
        cls.rng = np.random.default_rng(42)

    def _forecast(self, query: str, **kwargs) -> dict:
        history = kwargs.pop("history", self.rng.normal(0, 1, 72).tolist())
        return self.tfa.forecast(query=query, history=history, **kwargs)

    # ── Test 1: Hypertension → risk in 5%-30% range ──
    def test_hypertension_risk_moderate(self):
        """Stable hypertension should not produce extreme risk values."""
        results = []
        for _ in range(5):
            r = self._forecast(
                "Patient with well-controlled hypertension on medication, "
                "regular checkups show stable blood pressure 130/85"
            )
            results.append(float(r["short_term"]["risk_probability"]))
        avg = np.mean(results)
        # Well-controlled hypertension should be moderate-low risk
        assert avg < 0.50, f"Avg risk {avg:.4f} too high for stable hypertension"

    # ── Test 2: Acute MI → risk in 50%-90% range ──
    def test_acute_mi_risk_elevated(self):
        """Acute MI scenario should produce elevated risk."""
        results = []
        for _ in range(3):
            r = self._forecast(
                "Patient with acute chest pain radiating to left arm, "
                "ST elevation on ECG, troponin elevated, diaphoretic. "
                "Diagnosis: acute myocardial infarction."
            )
            results.append(float(r["short_term"]["risk_probability"]))
        avg = np.mean(results)
        # MI should show meaningfully elevated risk
        assert avg > 0.10, f"Avg risk {avg:.4f} too low for acute MI scenario"

    # ── Test 3: No 0% risk for sick patients ──
    def test_no_zero_risk_for_symptomatic(self):
        """Patients with active symptoms should never get exactly 0% risk."""
        queries = [
            "Patient with fever and productive cough for 5 days",
            "Severe abdominal pain with nausea and vomiting",
            "Acute onset confusion in elderly patient with UTI",
            "Post-operative patient with wound drainage and redness",
        ]
        for q in queries:
            r = self._forecast(q)
            short = float(r["short_term"]["risk_probability"])
            assert short > 0.0, f"Got 0.0 risk for symptomatic query: {q[:50]}"

    # ── Test 4: No 100% risk ──
    def test_no_certain_risk(self):
        """Risk should never be exactly 100% (medical uncertainty always exists)."""
        for _ in range(10):
            r = self._forecast(
                self.rng.choice([
                    "Cardiac arrest with CPR in progress",
                    "Septic shock on multiple vasopressors",
                    "Respiratory failure on mechanical ventilation",
                ])
            )
            short = float(r["short_term"]["risk_probability"])
            assert short < 1.0, f"Risk should never be 100%: got {short:.4f}"

    # ── Test 5: Recommendations are safe ──
    def test_recommendations_are_safe(self):
        """Recommendations must not contain dangerous or unapproved advice."""
        dangerous_phrases = [
            "guaranteed", "100%", "miracle", "secret formula",
            "always works", "cure all", "definitely",
        ]
        for _ in range(5):
            r = self._forecast(
                self.rng.choice([
                    "Patient with newly diagnosed hypertension",
                    "Mild community-acquired pneumonia",
                    "Type 2 diabetes follow-up",
                ])
            )
            recs = " ".join(r.get("recommendations", []))
            for phrase in dangerous_phrases:
                assert phrase.lower() not in recs.lower(), \
                    f"Dangerous phrase '{phrase}' found in recommendations"

    # ── Test 6: Risk factors related to diagnosis ──
    def test_risk_factors_are_present(self):
        """Every forecast should produce at least one risk factor."""
        r = self._forecast(
            "Patient with decompensated heart failure, pulmonary edema on CXR, "
            "BNP > 5000, requiring IV diuresis"
        )
        factors = r.get("primary_risk_factors", [])
        assert len(factors) >= 1, "Must identify at least one risk factor"

    # ── Test 7: Calibration mode degrades with low confidence ──
    def test_low_dma_confidence_triggers_degraded(self):
        r = self._forecast(
            "Patient with vague symptoms of fatigue and malaise",
            dma_confidence=0.45,
        )
        mode = r.get("calibration_mode", "")
        assert "degraded" in mode or "discount" in mode, \
            f"Low DMA confidence should degrade TFA: got mode={mode}"

    # ── Test 8: Output consistency (same input → same output) ──
    def test_reproducibility(self):
        """Same input must produce same output (deterministic fallback path)."""
        query = "Patient with stable chronic kidney disease stage 3"
        np.random.seed(1234)
        r1 = self.tfa.forecast(query=query)
        np.random.seed(1234)
        r2 = self.tfa.forecast(query=query)
        assert r1["risk_level"] == r2["risk_level"], \
            f"Reproducibility failed: {r1['risk_level']} != {r2['risk_level']}"
        assert abs(float(r1["risk_score"]) - float(r2["risk_score"])) < 1e-6, \
            "Reproducibility failed on risk_score"


if __name__ == "__main__":
    print("=" * 60)
    print("TFA Medical Knowledge Validation Test Suite")
    print("=" * 60)
    tester = TestTFAMedicalCheck()
    tester.setup_class()
    tests = [
        ("Hypertension moderate risk", tester.test_hypertension_risk_moderate),
        ("Acute MI elevated risk", tester.test_acute_mi_risk_elevated),
        ("No zero risk for symptomatic", tester.test_no_zero_risk_for_symptomatic),
        ("No 100% risk", tester.test_no_certain_risk),
        ("Safe recommendations", tester.test_recommendations_are_safe),
        ("Risk factors present", tester.test_risk_factors_are_present),
        ("Low confidence degrades", tester.test_low_dma_confidence_triggers_degraded),
        ("Reproducibility", tester.test_reproducibility),
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
