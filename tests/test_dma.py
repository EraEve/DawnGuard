from mapfm_ecosystem_repaired import DecisionMakingAgent, EcosystemConfig, load_medical_dataset


def test_dma_legacy_fallback_contract(monkeypatch):
    config = EcosystemConfig(allow_dma_legacy_fallback=True)
    dma = DecisionMakingAgent(config)
    df = load_medical_dataset(None, target_classes=3, seed=42)
    dma.fit(df)
    def fail(*args, **kwargs):
        from exceptions import InferenceError
        raise InferenceError("offline", code="TEST_OFFLINE")
    monkeypatch.setattr(dma, "_ollama_json_infer", fail)
    result = dma.infer("What is Breast Cancer?", context_vector=None, verified_docs=[])
    assert {"prediction", "confidence", "probabilities"}.issubset(result)
