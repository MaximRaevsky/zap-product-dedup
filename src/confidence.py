"""
Aggregate confidence into a final duplicate decision.
Simple policy: LLM overrides rules when present; rules handle the clear cases.
"""


def compute_final_confidence(pair: dict) -> dict:
    rule_conf = pair.get("rule_confidence", 0.0)
    llm_decided = pair.get("llm_is_duplicate") is not None

    if llm_decided:
        is_dup = pair["llm_is_duplicate"]
        final_conf = pair["llm_confidence"]
        source = "llm"
    elif rule_conf >= 0.85:
        is_dup = True
        final_conf = rule_conf
        source = "rules_high"
    elif rule_conf < 0.3:
        is_dup = False
        final_conf = 1.0 - rule_conf
        source = "rules_low"
    else:
        is_dup = rule_conf >= 0.6
        final_conf = rule_conf
        source = "rules_fallback"

    disagreement = False
    if llm_decided:
        if (rule_conf >= 0.5) != pair["llm_is_duplicate"]:
            disagreement = True

    return {
        **pair,
        "is_duplicate_pred": is_dup,
        "final_confidence": round(final_conf, 4),
        "decision_source": source,
        "rule_llm_disagreement": disagreement,
    }


def compute_all_confidences(pairs: list[dict]) -> list[dict]:
    results = [compute_final_confidence(p) for p in pairs]

    dup_count = sum(1 for r in results if r["is_duplicate_pred"])
    print(f"Final: {dup_count} duplicates, {len(results) - dup_count} non-duplicates")

    source_counts = {}
    for r in results:
        source_counts[r["decision_source"]] = source_counts.get(r["decision_source"], 0) + 1
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")

    disagree = sum(1 for r in results if r["rule_llm_disagreement"])
    if disagree:
        print(f"  Rule-LLM disagreements: {disagree}")
    return results
