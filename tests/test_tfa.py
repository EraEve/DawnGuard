from mapfm_ecosystem_repaired import EcosystemConfig, TemporalForeseeingAgent


def test_acute_fall_temporal_profile_raises_short_risk():
    config = EcosystemConfig()
    agent = TemporalForeseeingAgent(config)
    base = agent.forecast("general monitoring", history=[0.0] * 72, authoritative_signal={})
    acute = agent.forecast(
        "urgent fall risk",
        history=[0.0] * 72,
        authoritative_signal={"posture_instability": 1.0, "activity_reduction": 1.0},
    )
    # MedTsLLM inference may have small floating-point variation; allow 0.05 tolerance
    assert acute["short_term"]["risk_probability"] >= base["short_term"]["risk_probability"] - 0.05, (
        f"acute={acute['short_term']['risk_probability']:.4f} should be >= "
        f"base={base['short_term']['risk_probability']:.4f}"
    )
