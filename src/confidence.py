"""
Aggregate confidence scores across pipeline stages into a final duplicate decision.
"""


def compute_final_confidence(pair: dict) -> dict:
    """
    Compute final confidence and duplicate decision for a pair.
    LLM overrides rules when available; otherwise rules decide.
    """
    rule_conf = pair.get("rule_confidence", 0.0)
    llm_conf = pair.get("llm_confidence", 0.0)
    llm_decided = pair.get("llm_is_duplicate") is not None
    candidate_conf = pair.get("candidate_confidence", 0.5)

    if llm_decided:
        # LLM overrides for ambiguous cases
        is_dup = pair["llm_is_duplicate"]
        final_conf = llm_conf * 0.7 + rule_conf * 0.2 + candidate_conf * 0.1
        decision_source = "llm"
    elif rule_conf >= 0.8:
        is_dup = True
        final_conf = rule_conf * 0.8 + candidate_conf * 0.2
        decision_source = "rules_high"
    elif rule_conf < 0.3:
        is_dup = False
        final_conf = 1.0 - rule_conf
        decision_source = "rules_low"
    else:
        # Ambiguous but LLM wasn't called -- be conservative
        is_dup = rule_conf >= 0.55
        final_conf = rule_conf
        decision_source = "rules_fallback"

    # Check for rule-LLM disagreement
    disagreement = False
    if llm_decided:
        rule_says_dup = rule_conf >= 0.5
        if rule_says_dup != pair["llm_is_duplicate"]:
            disagreement = True

    return {
        **pair,
        "is_duplicate_pred": is_dup,
        "final_confidence": round(final_conf, 4),
        "decision_source": decision_source,
        "rule_llm_disagreement": disagreement,
    }


def compute_all_confidences(pairs: list[dict]) -> list[dict]:
    """Compute final confidence for all pairs."""
    results = [compute_final_confidence(p) for p in pairs]

    dup_count = sum(1 for r in results if r["is_duplicate_pred"])
    non_dup = len(results) - dup_count
    disagree = sum(1 for r in results if r["rule_llm_disagreement"])
    low_conf = sum(1 for r in results if r["final_confidence"] < 0.6 and r["is_duplicate_pred"])

    print(f"Final decisions: {dup_count} duplicates, {non_dup} non-duplicates")
    print(f"  Rule-LLM disagreements: {disagree}")
    print(f"  Low-confidence duplicates: {low_conf}")

    source_counts = {}
    for r in results:
        source_counts[r["decision_source"]] = source_counts.get(r["decision_source"], 0) + 1
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")

    return results
