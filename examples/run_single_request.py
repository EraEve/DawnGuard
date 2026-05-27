"""Minimal single-request smoke example."""
from mapfm_ecosystem_repaired import EcosystemConfig, HeterogeneousMultiAgentEcosystem, load_medical_dataset

config = EcosystemConfig()
dataset = load_medical_dataset(None, target_classes=22, seed=config.random_seed)
ecosystem = HeterogeneousMultiAgentEcosystem(config=config, dataset=dataset, base_model=config.ollama_model_name, verbose=False)
result = ecosystem.run_collaborative_task(
    patient_query="Emergency fall risk with reduced activity and unstable posture.",
    multimodal_input={"posture_instability": 0.9, "activity_reduction": 0.8, "bedrest_hours": 36},
)
print(result["dma"]["prediction"], result["tfa"]["short_term"]["risk_probability"])
