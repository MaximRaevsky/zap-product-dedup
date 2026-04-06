"""
Deterministic rule-based matching for candidate pairs.
Assigns confidence scores based on attribute overlap and fuzzy similarity.
"""

from rapidfuzz import fuzz
import pandas as pd


def apply_rules(candidate: dict, rec_df: pd.DataFrame) -> dict:
    """
    Apply deterministic matching rules to a candidate pair.
    Returns enriched candidate dict with rule_confidence and rule_decision.
    """
    a = rec_df[rec_df["idx"] == candidate["idx_a"]].iloc[0]
    b = rec_df[rec_df["idx"] == candidate["idx_b"]].iloc[0]

    norm_a = a["normalized"]
    norm_b = b["normalized"]
    cmp_a = a["norm_compare"]
    cmp_b = b["norm_compare"]

    token_sort = fuzz.token_sort_ratio(cmp_a, cmp_b)
    token_set = fuzz.token_set_ratio(cmp_a, cmp_b)
    partial = fuzz.partial_ratio(cmp_a, cmp_b)

    same_brand = (
        a["brand"] is not None
        and b["brand"] is not None
        and a["brand"] == b["brand"]
    )
    same_model = (
        a["model_number"] is not None
        and b["model_number"] is not None
        and a["model_number"] == b["model_number"]
    )
    same_series = (
        a["series"] is not None
        and b["series"] is not None
        and _series_match(a["series"], b["series"])
    )
    same_storage = (
        a["storage"] is not None
        and b["storage"] is not None
        and a["storage"] == b["storage"]
    )
    conflicting_storage = (
        a["storage"] is not None
        and b["storage"] is not None
        and a["storage"] != b["storage"]
    )
    same_screen = (
        a["screen_size"] is not None
        and b["screen_size"] is not None
        and a["screen_size"] == b["screen_size"]
    )
    conflicting_screen = (
        a["screen_size"] is not None
        and b["screen_size"] is not None
        and a["screen_size"] != b["screen_size"]
    )

    # Rule cascade (highest confidence first)
    rule_fired = None
    confidence = 0.0

    if norm_a == norm_b:
        confidence = 0.99
        rule_fired = "exact_normalized_match"

    elif same_brand and same_model:
        if conflicting_storage:
            confidence = 0.10
            rule_fired = "same_model_diff_storage"
        else:
            confidence = 0.95
            rule_fired = "brand_and_model_match"

    elif same_brand and same_series and same_storage:
        confidence = 0.90
        rule_fired = "brand_series_storage_match"

    elif same_brand and same_series and not conflicting_storage:
        if token_sort > 80:
            confidence = 0.85
            rule_fired = "brand_series_high_fuzzy"
        else:
            confidence = 0.60
            rule_fired = "brand_series_low_fuzzy"

    elif conflicting_storage or conflicting_screen:
        confidence = 0.05
        rule_fired = "conflicting_attributes"

    elif same_brand and token_sort > 85:
        # Check for near-identical model numbers that differ (different SKUs)
        if _has_differing_part_numbers(a["raw_title"], b["raw_title"]):
            confidence = 0.45
            rule_fired = "brand_high_fuzzy_diff_part_number"
        else:
            confidence = 0.80
            rule_fired = "brand_high_token_sort"

    elif token_sort > 85:
        confidence = 0.70
        rule_fired = "high_token_sort_no_brand"

    elif same_brand and token_set > 85:
        confidence = 0.65
        rule_fired = "brand_high_token_set"

    elif token_sort > 70:
        confidence = 0.50
        rule_fired = "medium_token_sort"

    elif token_set > 80 and partial > 80:
        confidence = 0.45
        rule_fired = "token_set_partial_combo"

    else:
        confidence = 0.20
        rule_fired = "low_similarity"

    return {
        **candidate,
        "rule_confidence": confidence,
        "rule_fired": rule_fired,
        "token_sort_ratio": token_sort,
        "token_set_ratio": token_set,
        "partial_ratio": partial,
        "same_brand": same_brand,
        "same_model": same_model,
        "same_series": same_series,
        "same_storage": same_storage,
        "conflicting_storage": conflicting_storage,
        "title_a": a["raw_title"],
        "title_b": b["raw_title"],
        "brand_a": a["brand"],
        "brand_b": b["brand"],
        "model_a": a.get("model_number"),
        "model_b": b.get("model_number"),
        "series_a": a.get("series"),
        "series_b": b.get("series"),
        "storage_a": a.get("storage"),
        "storage_b": b.get("storage"),
    }


def _has_differing_part_numbers(title_a: str, title_b: str) -> bool:
    """Detect when titles are near-identical but have different part/SKU numbers."""
    import re
    part_pattern = re.compile(r"\b([A-Z]{1,5}\d{2,}[A-Z0-9]{0,6}(?:[./][A-Z0-9]+)?)\b")
    parts_a = set(m.group(0) for m in part_pattern.finditer(title_a))
    parts_b = set(m.group(0) for m in part_pattern.finditer(title_b))
    if parts_a and parts_b:
        diff_a = parts_a - parts_b
        diff_b = parts_b - parts_a
        if diff_a and diff_b:
            for da in diff_a:
                for db in diff_b:
                    ratio = fuzz.ratio(da, db)
                    if 65 < ratio < 100:
                        return True
    return False


def _series_match(s1: str, s2: str) -> bool:
    """Check if two series strings match (case-insensitive, normalized whitespace)."""
    n1 = " ".join(s1.lower().split())
    n2 = " ".join(s2.lower().split())
    return n1 == n2 or fuzz.ratio(n1, n2) > 90


def apply_all_rules(candidates: list[dict], rec_df: pd.DataFrame) -> list[dict]:
    """Apply rules to all candidate pairs."""
    results = []
    for c in candidates:
        results.append(apply_rules(c, rec_df))

    rule_counts = {}
    for r in results:
        rule_counts[r["rule_fired"]] = rule_counts.get(r["rule_fired"], 0) + 1

    print("Rule-based matching results:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count}")

    high = sum(1 for r in results if r["rule_confidence"] >= 0.8)
    ambiguous = sum(1 for r in results if 0.3 <= r["rule_confidence"] < 0.8)
    low = sum(1 for r in results if r["rule_confidence"] < 0.3)
    print(f"  High confidence (>=0.8): {high}")
    print(f"  Ambiguous (0.3-0.8): {ambiguous}")
    print(f"  Low confidence (<0.3): {low}")

    return results
