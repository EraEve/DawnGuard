"""
HMAE Regression Test Suite
===========================
Ensures all original functionality is preserved after the HMAE v2.0 upgrade.
Tests the 4 canonical tasks and verifies no regressions in core modules.

Test data: held-out test split, never used for training.
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


CANONICAL_TASKS = [
    ("Task 01", "What are common symptoms of Lung Cancer?"),
    ("Task 02", "What is the prognosis for Heart Failure?"),
    ("Task 03", "What treatments are used for Colorectal Cancer?"),
    ("Task 04", "What treatments are used for High Blood Pressure?"),
]


class TestRegression:
    """Regression tests verifying no functionality loss after upgrade."""

    @classmethod
    def setup_class(cls):
        cls.config = EcosystemConfig()
        candidate_paths = [
            Path("medNo.22.csv"), Path("med.No22.csv"),
            Path(PROJECT_ROOT / "medNo.22.csv"),
        ]
        csv_path = next((p for p in candidate_paths if p.exists()), None)
        if csv_path is None:
            rng = np.random.default_rng(42)
            areas = ["Hypertension", "Type 2 Diabetes", "Asthma", "Pneumonia",
                     "Influenza", "Lung Cancer", "Heart Failure", "Colorectal Cancer"]
            rows = []
            for i in range(80):
                area = areas[i % len(areas)]
                rows.append({
                    "id": i + 1, "question": f"Symptoms of {area} case {i}",
                    "answer": f"Standard information about {area}.",
                    "area": area, "source": "Dataset", "last_updated": "2024-01-15",
                })
            cls.dataset = pd.DataFrame(rows)
        else:
            cls.dataset = load_medical_dataset(
                csv_path, target_classes=cls.config.target_classes,
                seed=cls.config.random_seed,
            )
        cls.ecosystem = HeterogeneousMultiAgentEcosystem(
            config=cls.config, dataset=cls.dataset,
            base_model=cls.config.ollama_model_name, verbose=False,
        )

    # ── Test 1-4: Canonical tasks still work ──
    def test_task_01_lung_cancer_symptoms(self):
        result = self.ecosystem.run_collaborative_task(
            patient_query=CANONICAL_TASKS[0][1],
        )
        assert result.get("dma"), "DMA output missing"
        assert result["dma"].get("prediction"), "DMA prediction missing"

    def test_task_02_heart_failure_prognosis(self):
        result = self.ecosystem.run_collaborative_task(
            patient_query=CANONICAL_TASKS[1][1],
        )
        assert result.get("dma"), "DMA output missing"

    def test_task_03_colorectal_cancer_treatments(self):
        result = self.ecosystem.run_collaborative_task(
            patient_query=CANONICAL_TASKS[2][1],
        )
        assert result.get("dma"), "DMA output missing"

    def test_task_04_high_blood_pressure_treatments(self):
        result = self.ecosystem.run_collaborative_task(
            patient_query=CANONICAL_TASKS[3][1],
        )
        assert result.get("dma"), "DMA output missing"

    # ── Test 5: DMA classification still works ──
    def test_dma_classification_basic(self):
        """DMA must still produce a valid prediction with confidence."""
        result = self.ecosystem.run_collaborative_task(
            patient_query="Patient with high blood pressure, headaches, and vision changes",
        )
        dma = result["dma"]
        assert "prediction" in dma
        assert "confidence" in dma
        assert 0.0 <= float(dma["confidence"]) <= 1.0

    # ── Test 6: RAA retrieval still works ──
    def test_raa_retrieval_basic(self):
        """RAA must retrieve documents and return metadata."""
        docs, meta = self.ecosystem.raa.retrieve(
            query="diabetes treatment options",
            strategy="mixed",
            top_k=5,
        )
        assert isinstance(docs, list)
        assert isinstance(meta, dict)
        assert "strategy" in meta

    # ── Test 7: Fusion/Verification still works ──
    def test_fusion_verification_basic(self):
        """Fusion must deduplicate and Verification must score documents."""
        docs, _ = self.ecosystem.raa.retrieve(
            query="hypertension management", strategy="mixed", top_k=8,
        )
        fused, fusion_summary = self.ecosystem.fusion_agent.fuse([docs])
        verified, ver_summary = self.ecosystem.verification_agent.verify(
            fused, fusion_summary=fusion_summary,
        )
        assert isinstance(fused, list)
        assert isinstance(verified, list)
        assert "accepted_docs" in ver_summary

    # ── Test 8: Consensus voting still works ──
    def test_consensus_voting_basic(self):
        """Consensus module must still produce valid votes."""
        dma_result = {"prediction": "Hypertension", "confidence": 0.85,
                      "hitl_status": "auto_verified", "review_required": False}
        ver_summary = {"accepted_docs": 8, "rejected_docs": 2,
                       "verification_ratio": 0.8, "verification_status": "passed",
                       "serious_conflicts": 0, "tfa_verified": True}
        consensus = self.ecosystem.consensus_module.hierarchical_vote(
            dma_result=dma_result, tfa_prediction=None,
            verification_summary=ver_summary,
            verification_agent=self.ecosystem.verification_agent,
        )
        assert "approved" in consensus
        assert "votes" in consensus
        assert "tier" in consensus

    # ── Test 9: HITL still triggers on low confidence ──
    def test_hitl_triggers_on_low_confidence(self):
        """HITL must trigger when DMA confidence is below threshold."""
        triggered, reason = self.ecosystem.hitl_manager.should_intervene(
            confidence=0.40,
            retrieval_relevance=0.50,
            risk_score=0.10,
            high_risk=False,
        )
        assert triggered, f"HITL should trigger on confidence 0.40, got: {triggered}"
        assert "low confidence" in reason.lower()

    # ── Test 10: Output format is 100% backward compatible ──
    def test_original_format_preserved(self):
        """Verify original log format fields are exactly preserved."""
        from utils.format_checker import FormatChecker
        checker = FormatChecker()
        # Original DMA format (without new fields) should still match
        assert checker.check_log_line(
            "DMA -> prediction=Lung Cancer | confidence=0.4810 | HITL=True"
            " | intent=symptom_query | status=low_confidence | severity=moderate",
            "DMA",
        ), "DMA format regression"
        # Original RAA format
        assert checker.check_log_line(
            "RAA -> strategy=mixed | Nash=True | rounds=2 | verified_relevance=0.5680"
            " | evidence_level=2.5 | conflicts=3",
            "RAA",
        ), "RAA format regression"
        # Original TFA format
        assert checker.check_log_line(
            "TFA -> future 24h deterioration risk=34.81% | risk_level=medium"
            " | confidence=0.8500 | source=fallback | model_version=heuristic-v2.0",
            "TFA",
        ), "TFA format regression"


if __name__ == "__main__":
    print("=" * 60)
    print("HMAE Regression Test Suite")
    print("=" * 60)
    tester = TestRegression()
    tester.setup_class()
    tests = [
        ("Task 01: Lung Cancer symptoms", tester.test_task_01_lung_cancer_symptoms),
        ("Task 02: Heart Failure prognosis", tester.test_task_02_heart_failure_prognosis),
        ("Task 03: Colorectal Cancer treatments", tester.test_task_03_colorectal_cancer_treatments),
        ("Task 04: High Blood Pressure treatments", tester.test_task_04_high_blood_pressure_treatments),
        ("DMA classification basic", tester.test_dma_classification_basic),
        ("RAA retrieval basic", tester.test_raa_retrieval_basic),
        ("Fusion/Verification basic", tester.test_fusion_verification_basic),
        ("Consensus voting basic", tester.test_consensus_voting_basic),
        ("HITL low confidence trigger", tester.test_hitl_triggers_on_low_confidence),
        ("Original format preserved", tester.test_original_format_preserved),
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
