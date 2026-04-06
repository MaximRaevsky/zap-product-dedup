"""
Deterministic rule-based matching for candidate pairs.
Deliberately simple: only handle the obvious cases.
Everything ambiguous goes to the LLM judge.
"""

from rapidfuzz import fuzz
import pandas as pd


def apply_rules(candidate: dict, rec_df: pd.DataFrame) -> dict:
    a = rec_df[rec_df["idx"] == candidate["idx_a"]].iloc[0]
    b = rec_df[rec_df["idx"] == candidate["idx_b"]].iloc[0]

    cmp_a, cmp_b = a["norm_compare"], b["norm_compare"]
    token_sort = fuzz.token_sort_ratio(cmp_a, cmp_b)
    token_set = fuzz.token_set_ratio(cmp_a, cmp_b)

    same_brand = a["brand"] and b["brand"] and a["brand"] == b["brand"]
    same_model = a["model_number"] and b["model_number"] and a["model_number"] == b["model_number"]
    same_model_id = a["model_id"] and b["model_id"] and str(a["model_id"]) == str(b["model_id"]) and str(a["model_id"]) != ""
    conflict_storage = a["storage"] and b["storage"] and a["storage"] != b["storage"]
    conflict_screen = a["screen_size"] and b["screen_size"] and a["screen_size"] != b["screen_size"]

    # Rule 1: same Zap model_id (strongest signal -- same product page)
    if same_model_id and not (conflict_storage or conflict_screen):
        conf, rule = 0.98, "same_model_id"
    # Rule 2: exact normalized match
    elif a["normalized"] == b["normalized"]:
        conf, rule = 0.99, "exact_match"
    # Rule 3: conflicting specs -> not duplicate
    elif conflict_storage or conflict_screen:
        conf, rule = 0.05, "conflicting_specs"
    # Rule 4: same brand + same model number
    elif same_brand and same_model:
        conf, rule = 0.95, "brand_model_match"
    # Rule 5: same brand + high fuzzy -> probably, let LLM confirm
    elif same_brand and token_sort > 85:
        conf, rule = 0.70, "brand_high_fuzzy"
    # Rule 6: decent fuzzy -> maybe, let LLM decide
    elif token_sort > 70 or (token_set > 85 and same_brand):
        conf, rule = 0.50, "moderate_fuzzy"
    # Rule 7: low similarity
    else:
        conf, rule = 0.15, "low_similarity"

    return {
        **candidate,
        "rule_confidence": conf,
        "rule_fired": rule,
        "token_sort_ratio": token_sort,
        "token_set_ratio": token_set,
        "same_brand": same_brand,
        "same_model": same_model,
        "title_a": a["raw_title"],
        "title_b": b["raw_title"],
        "brand_a": a["brand"],
        "brand_b": b["brand"],
        "model_a": a.get("model_number"),
        "model_b": b.get("model_number"),
        "storage_a": a.get("storage"),
        "storage_b": b.get("storage"),
    }


def apply_all_rules(candidates: list[dict], rec_df: pd.DataFrame) -> list[dict]:
    results = [apply_rules(c, rec_df) for c in candidates]

    rule_counts = {}
    for r in results:
        rule_counts[r["rule_fired"]] = rule_counts.get(r["rule_fired"], 0) + 1
    print("Rule-based matching:")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"  {rule}: {count}")

    high = sum(1 for r in results if r["rule_confidence"] >= 0.85)
    ambiguous = sum(1 for r in results if 0.3 <= r["rule_confidence"] < 0.85)
    low = sum(1 for r in results if r["rule_confidence"] < 0.3)
    print(f"  High: {high} | Ambiguous (->LLM): {ambiguous} | Low: {low}")

    return results
