"""
Evaluate deduplication pipeline using ground-truth evaluation pairs.
Computes pairwise precision, recall, F1, and cheapest-price correctness.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional


def evaluate_pairwise(
    predictions: list[dict],
    ground_truth_path: str = "data/synthetic/evaluation_pairs.csv",
) -> dict:
    """
    Evaluate predicted pairs against ground truth.
    Returns metrics dict.
    """
    gt_df = pd.read_csv(ground_truth_path)
    print(f"Loaded {len(gt_df)} ground truth pairs")

    # Build ground truth lookup: (sorted title pair) -> is_duplicate
    gt_lookup = {}
    for _, row in gt_df.iterrows():
        key = tuple(sorted([str(row["title_a"]).strip(), str(row["title_b"]).strip()]))
        gt_lookup[key] = bool(row["is_duplicate"])

    # Build prediction lookup
    pred_lookup = {}
    for p in predictions:
        key = tuple(sorted([str(p.get("title_a", "")).strip(), str(p.get("title_b", "")).strip()]))
        pred_lookup[key] = p.get("is_duplicate_pred", False)

    # Compute metrics on overlapping pairs
    tp = fp = fn = tn = 0
    matched_pairs = set(pred_lookup.keys()) & set(gt_lookup.keys())

    for key in gt_lookup:
        actual = gt_lookup[key]
        predicted = pred_lookup.get(key, False)

        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_gt_pairs": len(gt_lookup),
        "total_pred_pairs": len(pred_lookup),
        "overlapping_pairs": len(matched_pairs),
    }

    print(f"\nPairwise Evaluation:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    return metrics


def evaluate_price_correctness(
    clusters: pd.DataFrame,
    listings: pd.DataFrame,
) -> dict:
    """Check if cluster min prices are correct based on model_id grouping."""
    if "model_id" not in listings.columns:
        return {"price_accuracy": None, "note": "no model_id in listings"}

    correct = 0
    total = 0
    errors = []

    for _, cluster in clusters[clusters["num_listings"] > 1].iterrows():
        member_indices = cluster.get("member_indices", [])
        if not member_indices:
            continue

        member_rows = listings.loc[member_indices]
        model_ids = member_rows["model_id"].dropna().unique()

        if len(model_ids) == 1:
            true_prices = member_rows["price"].dropna()
            if len(true_prices) == 0:
                continue
            true_min = true_prices.min()
            pred_min = cluster["min_price"]

            total += 1
            if pred_min is not None and abs(pred_min - true_min) < 0.01:
                correct += 1
            else:
                errors.append({
                    "cluster_id": cluster["cluster_id"],
                    "canonical_title": cluster["canonical_title"],
                    "predicted_min": pred_min,
                    "actual_min": true_min,
                })

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nPrice Correctness:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total} clusters)")

    return {
        "price_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "errors": errors[:10],
    }


def save_metrics(
    metrics: dict,
    output_dir: str = "results/metrics",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"Metrics saved to {output_dir}/evaluation_metrics.json")
