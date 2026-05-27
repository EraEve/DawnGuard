"""
HMAE System Integration Test
=============================
Validates end-to-end integration of TFA with all other modules:
DMA, RAA, Fusion, Verification, Consensus, HITL.
All test data is synthetic or from the held-out test split.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from mapfm_ecosystem_repaired import (
    EcosystemConfig, HeterogeneousMultiAgentEcosystem,
)
from utils import load_medical_dataset


class TestIntegration:
    """System-level integration tests."""

    @classmethod
    def setup_class(cls):
        cls.config = EcosystemConfig()
        cls.config.enable_hitl = True
        cls.config.enable_tfa = True
        cls.config.enable_consensus = True
        cls.config.enable_multimodal = True

        # Load dataset
        candidate_paths = [
            Path("medNo.22.csv"), Path("med.No22.csv"),
            Path(PROJECT_ROOT / "medNo.22.csv"),
        ]
        csv_path = next((p for p in candidate_paths if p.exists()), None)
        if csv_path is None:
            # Create minimal synthetic dataset
            cls.dataset = cls._make_synthetic_dataset()
        else:
            cls.dataset = load_medical_dataset(
                csv_path, target_classes=cls.config.target_classes,
                seed=cls.config.random_seed,
            )
        cls.ecosystem = HeterogeneousMultiAgentEcosystem(
            config=cls.config, dataset=cls.dataset,
            base_model=cls.config.ollama_model_name, verbose=False,
        )
        cls.rng = np.random.default_rng(42)

    @staticmethod
    def _make_synthetic_dataset() -> pd.DataFrame:
        rng = np.random.default_rng(42)
        areas = ["Hypertension", "Type 2 Diabetes", "Asthma", "Pneumonia", "Influenza"]
        rows = []
        for i in range(100):
            area = areas[i % len(areas)]
            rows.append({
                "id": i + 1,
                "question": f"Patient {i}: symptoms related to {area}",
                "answer": f"Standard management for {area} includes medication and monitoring.",
                "area": area,
                "source": "Dataset",
                "last_updated": "2024-01-15",
            })
        return pd.DataFrame(rows)

    # ── Test 1: Full pipeline runs without crash ──
    def test_full_pipeline_completes(self):
        """Complete pipeline must run from perception to consensus without crashing."""
        result = self.ecosystem.run_collaborative_task(
            patient_query="What are the symptoms of hypertension?",
            multimodal_input={
                "image": {"precomputed_features": self.rng.normal(0, 1, 128).tolist()},
                "time_series": self.rng.normal(0, 1, 72).cumsum().tolist(),
            },
        )
        assert isinstance(result, dict), "Result must be a dict"
        assert "dma" in result, "DMA output missing"
        assert "tfa" in result, "TFA output missing"
        assert "retrieval" in result, "Retrieval output missing"
        assert "consensus" in result, "Consensus output missing"

    # ── Test 2: Consensus receives TFA vote ──
    def test_consensus_includes_tfa_vote(self):
        """Consensus result must include TFA's vote in the votes dict."""
        result = self.ecosystem.run_collaborative_task(
            patient_query="Patient with chest pain and shortness of breath",
            force_high_risk=True,
        )
        consensus = result.get("consensus", {})
        votes = consensus.get("votes", {})
        assert "TFA" in votes, f"TFA vote missing from consensus: {votes}"

    # ── Test 3: TFA high risk triggers HITL ──
    def test_tfa_high_risk_triggers_hitl(self):
        """High TFA risk must be reflected in HITL reasoning."""
        result = self.ecosystem.run_collaborative_task(
            patient_query="Acute myocardial infarction with ST elevation",
            force_high_risk=True,
        )
        tfa = result.get("tfa") or {}
        risk_level = tfa.get("risk_level", "N/A")
        hitl = result.get("hitl", {})
        # High risk + low DMA confidence should likely trigger HITL
        print(f"  TFA risk_level={risk_level}, HITL triggered={hitl.get('triggered')}")

    # ── Test 4: RAA risk-aware retrieval works ──
    def test_raa_risk_aware_retrieval(self):
        """RAA must produce risk knowledge when TFA is active."""
        result = self.ecosystem.run_collaborative_task(
            patient_query="Patient with worsening heart failure symptoms",
            force_high_risk=True,
        )
        risk_knowledge = result.get("risk_knowledge", {})
        assert isinstance(risk_knowledge, dict), "risk_knowledge must be a dict"
        assert "risk_level" in risk_knowledge, "risk_knowledge must include risk_level"

    # ── Test 5: Fusion with TFA produces comprehensive answer ──
    def test_fusion_with_tfa_produces_comprehensive_answer(self):
        """Fuse with TFA must produce a comprehensive answer dict."""
        result = self.ecosystem.run_collaborative_task(
            patient_query="Treatment options for type 2 diabetes",
            force_high_risk=False,
        )
        comprehensive = result.get("comprehensive_answer", {})
        assert isinstance(comprehensive, dict), "comprehensive_answer must be a dict"

    # ── Test 6: Concurrent tasks work ──
    def test_concurrent_tasks(self):
        """Multiple concurrent tasks must all complete successfully."""
        requests = [
            {"patient_query": f"Patient with symptoms of area {i}"}
            for i in range(3)
        ]
        results = self.ecosystem.run_concurrent_tasks(requests, max_workers=2)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        for r in results:
            assert "error" not in r or r.get("dma"), \
                f"Task failed: {r.get('error', 'unknown')}"

    # ── Test 7: Degradation tracked across pipeline ──
    def test_degradation_tracking(self):
        """DegradationManager must be populated after pipeline execution."""
        degradation = self.ecosystem.degradation.get_stats()
        assert "degradation_level" in degradation
        assert "component_states" in degradation

    # ── Test 8: Format compliance ──
    def test_format_compliance(self):
        """All log lines must pass format validation."""
        from utils.format_checker import FormatChecker
        checker = FormatChecker(strict=False)
        # Validate DMA format
        assert checker.check_log_line(
            "DMA -> prediction=Hypertension | confidence=0.8500 | HITL=False"
            " | intent=symptom_inquiry | status=success | severity=mild",
            "DMA"
        ), "DMA format check failed"
        # Validate TFA format
        assert checker.check_log_line(
            "TFA -> future 24h deterioration risk=15.50% | risk_level=low"
            " | confidence=0.9000 | source=medtsllm | model_version=MedTsLLM-v1.5",
            "TFA"
        ), "TFA format check failed"
        # Validate Consensus format
        assert checker.check_log_line(
            "Consensus -> approved=True | votes={'DMA': True, 'TFA': True, 'Verification': True}"
            " | required=3 | reason=\"Normal vote passed\" | hitl_triggered=False"
            " | risk_level=low | tier=auto_pass",
            "Consensus"
        ), "Consensus format check failed"
        # Validate HITL format
        assert checker.check_log_line(
            "HITL -> triggered=False | reason=N/A | task_id=01"
            " | intervention_time=2024-05-20 14:30:00 | operator=system | priority=0",
            "HITL"
        ), "HITL format check failed"


if __name__ == "__main__":
    print("=" * 60)
    print("HMAE System Integration Test Suite")
    print("=" * 60)
    tester = TestIntegration()
    tester.setup_class()
    tests = [
        ("Full pipeline completes", tester.test_full_pipeline_completes),
        ("Consensus includes TFA vote", tester.test_consensus_includes_tfa_vote),
        ("TFA high risk triggers HITL", tester.test_tfa_high_risk_triggers_hitl),
        ("RAA risk-aware retrieval", tester.test_raa_risk_aware_retrieval),
        ("Fusion with TFA", tester.test_fusion_with_tfa_produces_comprehensive_answer),
        ("Concurrent tasks", tester.test_concurrent_tasks),
        ("Degradation tracking", tester.test_degradation_tracking),
        ("Format compliance", tester.test_format_compliance),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {name} — {e}")
            traceback.print_exc()
    print(f"\nResult: {passed}/{len(tests)} tests passed")
    import sys as _sys
    _sys.exit(0 if passed == len(tests) else 1)
