"""
Analyze errors from the deduplication pipeline.
Identifies false positives, false negatives, low-confidence cases,
and LLM disagreements for debugging.
"""

import json
import pandas as pd
from pathlib import Path


def analyze_errors(
    predictions: list[dict],
    ground_truth_path: str = "data/synthetic/evaluation_pairs.csv",
    output_dir: str = "results/debug",
):
    """Comprehensive error analysis with debug output."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    gt_df = pd.read_csv(ground_truth_path)
    gt_lookup = {}
    gt_details = {}
    for _, row in gt_df.iterrows():
        key = tuple(sorted([str(row["title_a"]).strip(), str(row["title_b"]).strip()]))
        gt_lookup[key] = bool(row["is_duplicate"])
        gt_details[key] = row.to_dict()

    # Classify predictions
    false_positives = []
    false_negatives = []
    low_confidence = []
    llm_disagreements = []

    pred_lookup = {}
    for p in predictions:
        key = tuple(sorted([str(p.get("title_a", "")).strip(), str(p.get("title_b", "")).strip()]))
        pred_lookup[key] = p

        is_dup_pred = p.get("is_duplicate_pred", False)
        is_dup_gt = gt_lookup.get(key)

        if is_dup_gt is not None:
            if is_dup_pred and not is_dup_gt:
                false_positives.append(_format_error(p, "false_positive", gt_details.get(key, {})))
            elif not is_dup_pred and is_dup_gt:
                false_negatives.append(_format_error(p, "false_negative", gt_details.get(key, {})))

        if p.get("is_duplicate_pred") and p.get("final_confidence", 1.0) < 0.6:
            low_confidence.append(_format_error(p, "low_confidence"))

        if p.get("rule_llm_disagreement"):
            llm_disagreements.append(_format_error(p, "llm_disagreement"))

    # Find missed ground truth pairs (not in predictions at all)
    for key, is_dup in gt_lookup.items():
        if is_dup and key not in pred_lookup:
            false_negatives.append({
                "type": "missed_pair",
                "title_a": key[0],
                "title_b": key[1],
                "note": "Pair not generated as candidate",
            })

    _save_json(false_positives, f"{output_dir}/false_positives.json")
    _save_json(false_negatives, f"{output_dir}/false_negatives.json")
    _save_json(low_confidence, f"{output_dir}/low_confidence_cases.json")
    _save_json(llm_disagreements, f"{output_dir}/llm_disagreements.json")

    print(f"\nError Analysis:")
    print(f"  False positives: {len(false_positives)}")
    print(f"  False negatives: {len(false_negatives)}")
    print(f"  Low confidence duplicates: {len(low_confidence)}")
    print(f"  LLM disagreements: {len(llm_disagreements)}")

    return {
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "low_confidence": len(low_confidence),
        "llm_disagreements": len(llm_disagreements),
    }


def _format_error(pair: dict, error_type: str, gt_info: dict = None) -> dict:
    result = {
        "type": error_type,
        "title_a": pair.get("title_a", ""),
        "title_b": pair.get("title_b", ""),
        "rule_confidence": pair.get("rule_confidence"),
        "rule_fired": pair.get("rule_fired"),
        "llm_confidence": pair.get("llm_confidence"),
        "llm_rationale": pair.get("llm_rationale"),
        "final_confidence": pair.get("final_confidence"),
        "is_duplicate_pred": pair.get("is_duplicate_pred"),
        "decision_source": pair.get("decision_source"),
        "token_sort_ratio": pair.get("token_sort_ratio"),
        "brand_a": pair.get("brand_a"),
        "brand_b": pair.get("brand_b"),
        "model_a": pair.get("model_a"),
        "model_b": pair.get("model_b"),
        "storage_a": pair.get("storage_a"),
        "storage_b": pair.get("storage_b"),
    }
    if gt_info:
        result["gt_difficulty"] = gt_info.get("difficulty", "")
        result["gt_is_duplicate"] = gt_info.get("is_duplicate")
    return result


def _save_json(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
