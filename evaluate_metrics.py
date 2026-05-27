"""Comprehensive evaluation: accuracy, precision, recall, F1-score."""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mapfm_ecosystem_repaired import (
    EcosystemConfig,
    HeterogeneousMultiAgentEcosystem,
    load_medical_dataset,
    canonical_area,
)
from config_loader import load_yaml_config
from exceptions import ConfigError


def main():
    # --- Load config ---
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            config = load_yaml_config(EcosystemConfig, config_path)
        except ConfigError:
            config = EcosystemConfig()
    else:
        config = EcosystemConfig()

    # --- Load dataset ---
    csv_path = Path("medNo.22.csv")
    dataset = load_medical_dataset(
        csv_path if csv_path.exists() else None,
        target_classes=config.target_classes,
        seed=config.random_seed,
    )
    print(f"Dataset: {len(dataset)} samples, {dataset['area'].nunique()} classes")

    # --- Build ecosystem ---
    print("Building ecosystem + training DMA ...")
    ecosystem = HeterogeneousMultiAgentEcosystem(
        config=config,
        dataset=dataset,
        base_model=config.ollama_model_name,
        verbose=False,
    )

    # --- Run evaluation over entire test set ---
    test_df = ecosystem.test_df
    print(f"Evaluating on {len(test_df)} test samples ...\n")

    y_true: list[str] = []
    y_pred: list[str] = []
    confidences: list[float] = []
    latencies: list[float] = []

    for idx, (_, row) in enumerate(test_df.iterrows()):
        query = str(row["question"])
        true_label = canonical_area(str(row["area"]))
        try:
            start = time.time()
            result = ecosystem.run_collaborative_task(
                patient_query=query,
                multimodal_input=None,
                true_label=true_label,
                force_high_risk=False,
            )
            elapsed = time.time() - start
            latencies.append(elapsed)
            predicted = str(result["dma"]["prediction"])
            confidence = float(result["dma"]["confidence"])
        except Exception:
            predicted = "Unknown"
            confidence = 0.0
            elapsed = 0.0
            latencies.append(elapsed)

        y_true.append(true_label)
        y_pred.append(predicted)
        confidences.append(confidence)

        if (idx + 1) % 20 == 0:
            print(f"  ... {idx + 1}/{len(test_df)} done")

    # --- Compute metrics ---
    labels = sorted(set(y_true + y_pred) - {"Unknown"})

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # --- Print results ---
    print("\n" + "=" * 72)
    print("COMPREHENSIVE EVALUATION — HMAE Ecosystem")
    print("=" * 72)
    print(f"Test samples evaluated : {len(y_true)}")
    print(f"Classes               : {len(labels)}")
    print(f"Avg confidence        : {np.mean(confidences):.4f}")
    print(f"Avg latency           : {np.mean(latencies):.2f}s")
    print("-" * 72)
    print(f"{'Metric':<18} {'Macro':>10} {'Micro':>10} {'Weighted':>10}")
    print("-" * 72)
    print(f"{'Accuracy':<18} {'—':>10} {'—':>10} {accuracy:>10.4f}")
    print(f"{'Precision':<18} {precision_macro:>10.4f} {precision_micro:>10.4f} {precision_weighted:>10.4f}")
    print(f"{'Recall':<18} {recall_macro:>10.4f} {recall_micro:>10.4f} {recall_weighted:>10.4f}")
    print(f"{'F1-score':<18} {f1_macro:>10.4f} {f1_micro:>10.4f} {f1_weighted:>10.4f}")
    print("-" * 72)

    # --- Per-class breakdown ---
    print("\n" + "=" * 72)
    print("PER-CLASS METRICS")
    print("=" * 72)
    # Use sklearn's classification_report with zero_division
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    # Header
    print(f"{'Class':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("-" * 72)

    for label in labels:
        entry = report[label]
        name = label[:27]
        print(
            f"{name:<28} {entry['precision']:>10.4f} {entry['recall']:>10.4f} "
            f"{entry['f1-score']:>10.4f} {int(entry['support']):>8}"
        )

    # Print averages
    for avg in ("macro avg", "weighted avg"):
        entry = report[avg]
        label = avg.title()
        print(f"{'─' * 72}")
        print(
            f"{label:<28} {entry['precision']:>10.4f} {entry['recall']:>10.4f} "
            f"{entry['f1-score']:>10.4f} {int(entry['support']):>8}"
        )

    # --- Accuracy (also from report) ---
    print(f"{'─' * 72}")
    print(f"{'Accuracy':<28} {'—':>10} {'—':>10} {accuracy:>10.4f} {int(report.get('accuracy', 0) * len(y_true)):>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
