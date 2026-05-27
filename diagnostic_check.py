"""Diagnostic script to check all core issues from the prompt."""
import sys
import json
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from mapfm_ecosystem_repaired import (
    EcosystemConfig, HeterogeneousMultiAgentEcosystem,
    load_medical_dataset, DecisionMakingAgent,
    TemporalForeseeingAgent, ConsensusModule,
    KnowledgeFusionAgent, KnowledgeVerificationAgent,
    HumanInTheLoopManager,
)

print("=" * 80)
print("MAPFM DIAGNOSTIC CHECK")
print("=" * 80)

config = EcosystemConfig(allow_dma_legacy_fallback=True, enable_hitl=True, enable_tfa=True)
df = load_medical_dataset(None, target_classes=6, seed=42)

print(f"\n[1] Dataset loaded: {len(df)} rows, classes: {df['area'].nunique()}")

# Build ecosystem
eco = HeterogeneousMultiAgentEcosystem(config, df)

# Run our 4 key tasks
tasks = [
    ("Task 01", "What are common symptoms of Lung Cancer?", "Lung Cancer"),
    ("Task 02", "What is the prognosis for Heart Failure?", "Heart Failure"),
    ("Task 03", "What treatments are used for Colorectal Cancer?", "Colorectal Cancer"),
    ("Task 04", "What treatments are used for High Blood Pressure?", "High Blood Pressure"),
]

print("\n" + "=" * 80)
print("RUNNING 4 KEY TASKS")
print("=" * 80)

all_results = []
for task_name, query, expected_label in tasks:
    print(f"\n{'─' * 60}")
    print(f"[{task_name}]: '{query}'")
    print(f"   Expected: {expected_label}")

    result = eco.run_collaborative_task(query, true_label=expected_label)
    all_results.append(result)

    dma_pred = result["dma"]["prediction"]
    dma_conf = result["dma"]["confidence"]
    tfa_short = result["tfa"]["short_term"]["risk_probability"] if result["tfa"] else "N/A"
    consensus = result["consensus"]
    hitl = result["hitl"]
    fusion = result["fusion"]
    verification = result["verification"]

    print(f"   DMA Prediction: {dma_pred} (conf={dma_conf:.4f})")
    correct = dma_pred == expected_label
    print(f"   DMA Correct? {'YES' if correct else 'NO <<<'}")
    print(f"   TFA 24h Risk: {tfa_short}")
    print(f"   Fusion conflicts: {fusion.get('conflict_count', 'N/A')}")
    verif_accepted = verification.get('accepted_docs', 'N/A')
    verif_ratio = verification.get('verification_ratio', 'N/A')
    print(f"   Verification: accepted={verif_accepted}, ratio={verif_ratio}")
    print(f"   HITL triggered: {hitl.get('triggered', 'N/A')}")
    if consensus:
        print(f"   Consensus: approved={consensus.get('approved', 'N/A')}")
        print(f"   Votes: {consensus.get('votes', 'N/A')}")
        print(f"   Weights: {consensus.get('vote_weights', 'N/A')}")
    else:
        print(f"   Consensus: NOT RUN (high_risk={result.get('high_risk', 'N/A')})")

# Summary check
print("\n" + "=" * 80)
print("ISSUE-BY-ISSUE DIAGNOSTIC")
print("=" * 80)

print("\n[ISSUE 1] TFA voting rights check:")
tfa = TemporalForeseeingAgent(config)
sample_forecast = tfa.forecast("test query")
has_voting = sample_forecast.get("has_voting_right", "MISSING")
print(f"  TFA has_voting_right field: {has_voting}")
if has_voting == "MISSING":
    print(f"  STATUS: ** PROBLEM - TFA forecast output does not have has_voting_right field")
elif has_voting == False:
    print(f"  STATUS: OK - TFA has_voting_right=False")
else:
    print(f"  STATUS: ** PROBLEM - TFA has_voting_right=True (should be False)")

print("\n[ISSUE 2] Consensus voting weights check:")
dummy_dma = {"prediction": "Hypertension", "confidence": 0.85}
dummy_tfa = {"short_term": {"risk_probability": 0.99}}
dummy_verification = {"accepted_docs": 5, "rejected_docs": 0, "verification_ratio": 1.0}
verification_agent = KnowledgeVerificationAgent(config)
vote_result = eco.consensus_module.vote(dummy_dma, dummy_tfa, dummy_verification, verification_agent)
print(f"  Vote result: approved={vote_result['approved']}")
print(f"  Vote weights: DMA={vote_result['vote_weights'].get('DMA', '?')}, TFA={vote_result['vote_weights'].get('TFA', '?')}, Verification={vote_result['vote_weights'].get('Verification', '?')}")
print(f"  Votes: {vote_result['votes']}")
tfa_weight = vote_result['vote_weights'].get('TFA', -1)
print(f"  STATUS: {'** PROBLEM - TFA weight should be 0' if tfa_weight != 0 else 'OK - TFA weight is 0'}")

print("\n[ISSUE 3] TFA output medical sanity:")
for i, (task_name, _, expected_label) in enumerate(tasks):
    if i < len(all_results):
        tfa_out = all_results[i]["tfa"]
        if tfa_out:
            risk = tfa_out["short_term"]["risk_probability"]
            is_extreme = risk < 0.05 or risk > 0.95
            flag = " ** EXTREME VALUE!" if is_extreme else " OK"
            print(f"  {task_name} ({expected_label}): 24h risk={risk:.4f}{flag}")
        else:
            print(f"  {task_name}: TFA not available")

print("\n[ISSUE 4] Fusion conflict detection:")
for i, (task_name, _, _) in enumerate(tasks):
    if i < len(all_results):
        fusion = all_results[i]["fusion"]
        print(f"  {task_name}: conflicts={fusion.get('conflict_count', '?')}, input={fusion.get('input_docs', '?')}, dedup={fusion.get('deduplicated_docs', '?')}")

print("\n[ISSUE 5] Verification pass-rate:")
for i, (task_name, _, _) in enumerate(tasks):
    if i < len(all_results):
        verif = all_results[i]["verification"]
        accepted = verif.get('accepted_docs', 0)
        rejected = verif.get('rejected_docs', 0)
        ratio = verif.get('verification_ratio', 0)
        total = accepted + rejected
        print(f"  {task_name}: {accepted}/{total} verified (ratio={ratio:.2f})")

print("\n[ISSUE 6] HITL flow check:")
for i, (task_name, _, _) in enumerate(tasks):
    if i < len(all_results):
        hitl = all_results[i]["hitl"]
        print(f"  {task_name}: triggered={hitl.get('triggered', '?')}, layer={hitl.get('intervention_layer', '?')}")

print("\n[ISSUE 7] Consensus runs only for high_risk:")
for i, (task_name, _, _) in enumerate(tasks):
    if i < len(all_results):
        result = all_results[i]
        cons = result.get('consensus')
        print(f"  {task_name}: high_risk={result.get('high_risk', '?')}, consensus_ran={'yes' if cons else 'NO'}")

print("\n[ISSUE 8] HITL actual blocking vs log-only:")
print("  Code check: HITL process_decision() sets intervention_layer='needs_human_review'")
print("  but never pauses the pipeline or waits for actual human input.")
print("  STATUS: ** PROBLEM - HITL is log-only, no actual interruption")

print("\n[ISSUE 9] DMA 'intent' recognition field:")
for i, (task_name, _, _) in enumerate(tasks):
    if i < len(all_results):
        dma = all_results[i]["dma"]
        has_intent = "intent" in dma
        print(f"  {task_name}: intent field present={has_intent}")

print("\n[ISSUE 10] TFA has_voting_right in actual pipeline results:")
for i, (task_name, _, _) in enumerate(tasks):
    if i < len(all_results):
        tfa_out = all_results[i]["tfa"]
        if tfa_out:
            print(f"  {task_name}: has_voting_right={tfa_out.get('has_voting_right', 'MISSING')}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
