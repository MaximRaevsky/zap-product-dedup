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
    """
    Evaluate whether the customer sees the true cheapest price after dedup.

    Two dimensions:
    1. Cluster purity -- does each cluster contain only one real product?
       Impure clusters mix different products, making the shown price meaningless.
    2. Price accuracy on pure clusters -- among correctly-grouped products,
       did we surface the cheapest available listing?
    """
    if "model_id" not in listings.columns:
        return {"note": "no model_id in listings"}

    idx_to_cluster = {}
    for _, cluster in clusters.iterrows():
        for idx in cluster.get("member_indices", []):
            idx_to_cluster[idx] = cluster["cluster_id"]

    cluster_min_price = clusters.set_index("cluster_id")["min_price"].to_dict()

    # For each cluster, find which model_ids it contains
    cluster_model_ids: dict[int, set] = {}
    for idx, row in listings.iterrows():
        cid = idx_to_cluster.get(idx)
        if cid is not None:
            cluster_model_ids.setdefault(cid, set()).add(row["model_id"])

    pure_clusters = {cid for cid, mids in cluster_model_ids.items() if len(mids) == 1}
    multi_listing_cids = set(clusters[clusters["num_listings"] > 1]["cluster_id"])
    pure_multi = pure_clusters & multi_listing_cids
    impure_multi = multi_listing_cids - pure_clusters

    # Price accuracy: for each model_id, check if its primary cluster shows
    # the true cheapest. Only meaningful for products in pure clusters.
    correct = 0
    total_pure = 0
    total_impure = 0
    price_errors = []

    for mid, group in listings.groupby("model_id"):
        prices = group["price"].dropna()
        if len(prices) < 2:
            continue
        true_min = prices.min()

        cluster_counts: dict[int, int] = {}
        for idx in group.index:
            cid = idx_to_cluster.get(idx)
            if cid is not None:
                cluster_counts[cid] = cluster_counts.get(cid, 0) + 1

        if not cluster_counts:
            continue

        primary_cid = max(cluster_counts, key=cluster_counts.get)

        if primary_cid not in pure_clusters:
            total_impure += 1
            continue

        total_pure += 1
        shown_min = cluster_min_price.get(primary_cid)
        if shown_min is not None and abs(shown_min - true_min) < 0.01:
            correct += 1
        else:
            price_errors.append({
                "model_id": str(mid),
                "true_cheapest": true_min,
                "shown_cheapest": shown_min,
                "direction": "too_high" if (shown_min or 0) > true_min else "too_low",
            })

    price_acc = correct / total_pure if total_pure > 0 else 0.0
    purity = len(pure_multi) / len(multi_listing_cids) if multi_listing_cids else 0.0

    savings = _compute_savings(clusters, pure_clusters)

    print(f"\nCluster Quality:")
    print(f"  Pure clusters (single product): {len(pure_multi)}/{len(multi_listing_cids)} ({purity:.1%})")
    print(f"  Over-merged clusters: {len(impure_multi)}")
    print(f"\nPrice Correctness (pure clusters only):")
    print(f"  Cheapest price found: {price_acc:.1%} ({correct}/{total_pure} products)")
    print(f"  Products in over-merged clusters (excluded): {total_impure}")
    print(f"\nBusiness Impact (from correctly-grouped products):")
    print(f"  Products with price variation: {savings['products_with_variation']}/{savings['products_evaluated']}")
    if savings["products_with_variation"] > 0:
        print(f"  Avg savings from grouping: {savings['avg_saving_pct']:.1f}%")
        print(f"  Max savings from grouping: {savings['max_saving_pct']:.1f}%")
        print(f"  Median saving per product: {savings['median_saving_abs']:.0f} NIS")

    return {
        "cluster_purity": round(purity, 4),
        "pure_clusters": len(pure_multi),
        "impure_clusters": len(impure_multi),
        "price_accuracy": round(price_acc, 4),
        "price_correct": correct,
        "price_evaluated": total_pure,
        "products_in_impure_clusters": total_impure,
        "price_errors": price_errors[:10],
        "savings": savings,
    }


def _compute_savings(clusters: pd.DataFrame, pure_clusters: set) -> dict:
    """
    For pure multi-listing clusters, measure how much grouping saves
    customers vs seeing a single store's price.
    """
    saving_pcts = []
    saving_abs = []
    products_evaluated = 0

    for _, cluster in clusters[clusters["num_listings"] > 1].iterrows():
        if cluster["cluster_id"] not in pure_clusters:
            continue
        price_str = cluster.get("all_prices", "")
        if not price_str:
            continue
        prices = []
        for p in str(price_str).split(","):
            p = p.strip()
            try:
                prices.append(float(p))
            except ValueError:
                continue
        if len(prices) < 2:
            continue

        products_evaluated += 1
        lo, hi = min(prices), max(prices)
        if hi > 0 and hi != lo:
            saving_pcts.append((hi - lo) / hi * 100)
            saving_abs.append(hi - lo)

    n_var = len(saving_pcts)
    return {
        "products_evaluated": products_evaluated,
        "products_with_variation": n_var,
        "avg_saving_pct": round(sum(saving_pcts) / n_var, 1) if n_var else 0,
        "max_saving_pct": round(max(saving_pcts), 1) if n_var else 0,
        "median_saving_abs": round(sorted(saving_abs)[n_var // 2], 1) if n_var else 0,
        "avg_saving_abs": round(sum(saving_abs) / n_var, 1) if n_var else 0,
    }


def save_metrics(
    metrics: dict,
    output_dir: str = "results/metrics",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"Metrics saved to {output_dir}/evaluation_metrics.json")
