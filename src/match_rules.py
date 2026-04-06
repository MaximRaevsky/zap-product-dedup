"""
Deterministic rule-based matching for candidate pairs.
No brand dictionaries -- only structural signals (model_id, model number, specs, fuzzy).
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

    same_model = a["model_number"] and b["model_number"] and a["model_number"] == b["model_number"]
    same_model_id = a["model_id"] and b["model_id"] and str(a["model_id"]) == str(b["model_id"]) and str(a["model_id"]) != ""
    conflict_storage = a["storage"] and b["storage"] and a["storage"] != b["storage"]
    conflict_screen = a["screen_size"] and b["screen_size"] and a["screen_size"] != b["screen_size"]

    exact_norm = a["normalized"] == b["normalized"]

    if same_model_id and not (conflict_storage or conflict_screen):
        conf, rule = 0.98, "same_model_id"
    elif exact_norm:
        conf, rule = 0.10, "exact_cross_product"
    elif conflict_storage or conflict_screen:
        conf, rule = 0.05, "conflicting_specs"
    elif same_model:
        conf, rule = 0.90, "same_model_number"
    elif token_sort > 85:
        conf, rule = 0.65, "high_fuzzy"
    elif token_sort > 65 or token_set > 85:
        conf, rule = 0.45, "moderate_fuzzy"
    else:
        conf, rule = 0.15, "low_similarity"

    return {
        **candidate,
        "rule_confidence": conf,
        "rule_fired": rule,
        "token_sort_ratio": token_sort,
        "token_set_ratio": token_set,
        "same_model": same_model,
        "title_a": a["raw_title"],
        "title_b": b["raw_title"],
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
