from mapfm_ecosystem_repaired import EcosystemConfig, OnlineKnowledgeBaseManager, RetrievalAugmentedAgent, ContrastiveRetrievalUpdater, load_medical_dataset


def test_retrieval_returns_metadata():
    config = EcosystemConfig()
    df = load_medical_dataset(None, target_classes=3, seed=42)
    kb = OnlineKnowledgeBaseManager(config)
    kb.build_from_dataframe(df)
    raa = RetrievalAugmentedAgent(config, kb, ContrastiveRetrievalUpdater(config))
    docs, meta = raa.retrieve("What is Breast Cancer?", strategy="mixed", top_k=3)
    assert "result_count" in meta
    assert isinstance(docs, list)
