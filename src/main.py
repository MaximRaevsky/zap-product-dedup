"""
Main orchestrator for the Zap product deduplication pipeline.
Runs end-to-end: collect -> seed -> augment -> normalize -> extract ->
candidates -> rules -> LLM -> cluster -> price -> evaluate -> analyze.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collect_zap_data import scrape_all
from collect_seed_data import build_seed_dataset
from generate_variants import generate_all
from candidate_generation import generate_candidates
from match_rules import apply_all_rules
from llm_judge import judge_ambiguous_pairs
from confidence import compute_all_confidences
from cluster_products import build_clusters
from select_best_price import select_and_export
from evaluate import evaluate_pairwise, evaluate_price_correctness, save_metrics
from analyze_errors import analyze_errors

import pandas as pd


def save_debug_artifacts(rec_df, candidates, rule_results, final_pairs, clusters):
    """Save comprehensive debug artifacts."""
    debug_dir = Path("results/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Normalization examples
    norm_examples = []
    for _, row in rec_df.head(20).iterrows():
        norm_examples.append({
            "raw_title": row["raw_title"],
            "normalized": row["normalized"],
            "norm_compare": row["norm_compare"],
            "brand": row.get("brand"),
            "series": row.get("series"),
            "storage": row.get("storage"),
        })
    _save_json(norm_examples, debug_dir / "normalization_examples.json")

    # Attribute extraction examples
    attr_examples = []
    for _, row in rec_df.head(30).iterrows():
        attr_examples.append({
            "raw_title": row["raw_title"],
            "brand": row.get("brand"),
            "brand_confidence": row.get("brand_confidence"),
            "model_number": row.get("model_number"),
            "series": row.get("series"),
            "storage": row.get("storage"),
            "screen_size": row.get("screen_size"),
        })
    _save_json(attr_examples, debug_dir / "attribute_extraction_examples.json")

    # Candidate pairs log
    cand_log = []
    for c in candidates[:200]:
        cand_log.append({
            "idx_a": c["idx_a"],
            "idx_b": c["idx_b"],
            "reason": c["reason"],
            "candidate_confidence": c.get("candidate_confidence"),
        })
    _save_json(cand_log, debug_dir / "candidate_pairs_log.json")

    # Rule decisions log
    rule_log = []
    for r in rule_results[:200]:
        rule_log.append({
            "title_a": r.get("title_a", "")[:80],
            "title_b": r.get("title_b", "")[:80],
            "rule_fired": r.get("rule_fired"),
            "rule_confidence": r.get("rule_confidence"),
            "token_sort_ratio": r.get("token_sort_ratio"),
            "same_brand": r.get("same_brand"),
            "same_model": r.get("same_model"),
            "same_series": r.get("same_series"),
        })
    _save_json(rule_log, debug_dir / "rule_decisions_log.json")


def _save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def run_pipeline(
    skip_scrape: bool = False,
    skip_synthetic: bool = False,
    debug: bool = True,
    confidence_threshold: float = 0.5,
):
    """Run the full deduplication pipeline."""
    start_time = time.time()
    print("=" * 60)
    print("ZAP PRODUCT DEDUPLICATION PIPELINE")
    print("=" * 60)

    # Step 1: Data collection
    if not skip_scrape:
        print("\n[1/9] Scraping Zap data...")
        scrape_all()
    else:
        print("\n[1/9] Skipping scrape (using existing data)")

    # Step 2: Seed dataset
    print("\n[2/9] Building seed dataset...")
    seed_df = build_seed_dataset()

    # Step 3: Synthetic augmentation
    if not skip_synthetic:
        print("\n[3/9] Generating synthetic variants...")
        all_listings, eval_pairs = generate_all()
    else:
        print("\n[3/9] Skipping synthetic generation (using existing data)")
        all_listings = pd.read_csv("data/synthetic/all_listings.csv")
        eval_pairs = pd.read_csv("data/synthetic/evaluation_pairs.csv")

    # Step 4: Candidate generation
    print("\n[4/9] Generating candidate pairs...")
    candidates, rec_df = generate_candidates(all_listings)

    # Step 5: Rule-based matching
    print("\n[5/9] Applying rule-based matching...")
    rule_results = apply_all_rules(candidates, rec_df)

    # Step 6: LLM judge for ambiguous cases
    print("\n[6/9] LLM judging ambiguous pairs...")
    llm_results = judge_ambiguous_pairs(rule_results)

    # Step 7: Final confidence
    print("\n[7/9] Computing final confidences...")
    final_pairs = compute_all_confidences(llm_results)

    # Step 8: Clustering and price selection
    print("\n[8/9] Clustering and selecting best prices...")
    clusters = build_clusters(final_pairs, all_listings, confidence_threshold)
    export_df = select_and_export(clusters)

    # Step 9: Evaluation
    print("\n[9/9] Evaluating...")
    pairwise_metrics = evaluate_pairwise(final_pairs)
    price_metrics = evaluate_price_correctness(clusters, all_listings)
    error_analysis = analyze_errors(final_pairs)

    all_metrics = {
        "pairwise": pairwise_metrics,
        "price": price_metrics,
        "errors": error_analysis,
        "pipeline_stats": {
            "total_listings": len(all_listings),
            "candidate_pairs": len(candidates),
            "rule_results": len(rule_results),
            "final_pairs_dup": sum(1 for p in final_pairs if p.get("is_duplicate_pred")),
            "clusters_total": len(clusters),
            "clusters_multi": int((clusters["num_listings"] > 1).sum()),
            "elapsed_seconds": round(time.time() - start_time, 1),
        },
    }
    save_metrics(all_metrics)

    if debug:
        print("\nSaving debug artifacts...")
        save_debug_artifacts(rec_df, candidates, rule_results, final_pairs, clusters)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  F1: {pairwise_metrics['f1']:.4f} (P={pairwise_metrics['precision']:.4f}, R={pairwise_metrics['recall']:.4f})")
    print(f"  Clusters: {len(clusters)} ({(clusters['num_listings'] > 1).sum()} grouped)")
    print(f"{'=' * 60}")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zap Product Deduplication Pipeline")
    parser.add_argument("--skip-scrape", action="store_true", help="Skip web scraping")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--debug", action="store_true", default=True, help="Save debug artifacts")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for clustering")
    args = parser.parse_args()

    run_pipeline(
        skip_scrape=args.skip_scrape,
        skip_synthetic=args.skip_synthetic,
        debug=args.debug,
        confidence_threshold=args.threshold,
    )
