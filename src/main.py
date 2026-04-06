"""
Main orchestrator for the Zap product deduplication pipeline.
Supports multiple runs with different random category samples.
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


def _save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def save_debug_artifacts(rec_df, candidates, rule_results, final_pairs, clusters):
    debug_dir = Path("results/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)

    norm_examples = [
        {"raw_title": r["raw_title"], "normalized": r["normalized"],
         "brand": r.get("brand"), "storage": r.get("storage")}
        for _, r in rec_df.head(20).iterrows()
    ]
    _save_json(norm_examples, debug_dir / "normalization_examples.json")

    cand_log = [{"idx_a": c["idx_a"], "idx_b": c["idx_b"], "reason": c["reason"],
                 "confidence": c.get("candidate_confidence")} for c in candidates[:200]]
    _save_json(cand_log, debug_dir / "candidate_pairs_log.json")

    rule_log = [{"title_a": r.get("title_a", "")[:80], "title_b": r.get("title_b", "")[:80],
                 "rule": r.get("rule_fired"), "conf": r.get("rule_confidence"),
                 "brand": r.get("same_brand"), "model": r.get("same_model")}
                for r in rule_results[:200]]
    _save_json(rule_log, debug_dir / "rule_decisions_log.json")


def run_pipeline(
    skip_scrape: bool = False,
    skip_synthetic: bool = False,
    debug: bool = True,
    confidence_threshold: float = 0.5,
    seed: int = None,
    n_categories: int = 10,
):
    start = time.time()
    print("=" * 60)
    print("ZAP PRODUCT DEDUPLICATION PIPELINE")
    if seed is not None:
        print(f"  Random seed: {seed}")
    print("=" * 60)

    if not skip_scrape:
        print("\n[1/9] Scraping Zap data...")
        scrape_all(n_categories=n_categories, seed=seed)
    else:
        print("\n[1/9] Skipping scrape (using existing data)")

    print("\n[2/9] Building seed dataset...")
    seed_df = build_seed_dataset()

    if not skip_synthetic:
        print("\n[3/9] Generating synthetic variants...")
        all_listings, eval_pairs = generate_all()
    else:
        print("\n[3/9] Using existing synthetic data")
        all_listings = pd.read_csv("data/synthetic/all_listings.csv")

    print("\n[4/9] Generating candidate pairs...")
    candidates, rec_df = generate_candidates(all_listings)

    print("\n[5/9] Rule-based matching...")
    rule_results = apply_all_rules(candidates, rec_df)

    print("\n[6/9] LLM judging ambiguous pairs...")
    llm_results = judge_ambiguous_pairs(rule_results)

    print("\n[7/9] Final confidence...")
    final_pairs = compute_all_confidences(llm_results)

    print("\n[8/9] Clustering and best price...")
    clusters = build_clusters(final_pairs, all_listings, confidence_threshold)
    select_and_export(clusters)

    print("\n[9/9] Evaluation...")
    pairwise = evaluate_pairwise(final_pairs)
    price = evaluate_price_correctness(clusters, all_listings)
    errors = analyze_errors(final_pairs)

    metrics = {
        "pairwise": pairwise,
        "price": price,
        "errors": errors,
        "stats": {
            "total_listings": len(all_listings),
            "candidates": len(candidates),
            "dup_predictions": sum(1 for p in final_pairs if p.get("is_duplicate_pred")),
            "clusters": len(clusters),
            "grouped_clusters": int((clusters["num_listings"] > 1).sum()),
            "elapsed_s": round(time.time() - start, 1),
            "seed": seed,
        },
    }
    save_metrics(metrics)

    if debug:
        save_debug_artifacts(rec_df, candidates, rule_results, final_pairs, clusters)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s | F1={pairwise['f1']:.4f} P={pairwise['precision']:.4f} R={pairwise['recall']:.4f}")
    print(f"{'=' * 60}")

    return metrics


def run_multiple(n_runs: int = 3, n_categories: int = 10):
    """Run pipeline multiple times with different random category samples."""
    log_path = Path("results/metrics/iteration_log.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = []
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("# Pipeline Runs (Diverse Category Samples)\n\n")
        log.write("| Run | Seed | F1 | Precision | Recall | Listings | Clusters | Elapsed |\n")
        log.write("|-----|------|----|-----------|--------|----------|----------|---------|\n")

    for i in range(n_runs):
        seed = 100 + i * 37
        print(f"\n{'#' * 60}")
        print(f"# RUN {i+1}/{n_runs} (seed={seed})")
        print(f"{'#' * 60}")

        metrics = run_pipeline(seed=seed, n_categories=n_categories)
        all_results.append(metrics)

        with open(log_path, "a", encoding="utf-8") as log:
            p = metrics["pairwise"]
            s = metrics["stats"]
            log.write(f"| {i+1} | {seed} | {p['f1']:.4f} | {p['precision']:.4f} | {p['recall']:.4f} | {s['total_listings']} | {s['clusters']} | {s['elapsed_s']}s |\n")

    # Summary
    f1s = [m["pairwise"]["f1"] for m in all_results]
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n**Average F1: {sum(f1s)/len(f1s):.4f}** (min={min(f1s):.4f}, max={max(f1s):.4f})\n")

    print(f"\n{'=' * 60}")
    print(f"Multi-run complete: avg F1={sum(f1s)/len(f1s):.4f}")
    print(f"{'=' * 60}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zap Product Deduplication")
    parser.add_argument("--skip-scrape", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--multi", type=int, default=0, help="Run N times with different seeds")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for category sampling")
    parser.add_argument("--n-categories", type=int, default=10)
    args = parser.parse_args()

    if args.multi > 0:
        run_multiple(n_runs=args.multi, n_categories=args.n_categories)
    else:
        run_pipeline(
            skip_scrape=args.skip_scrape,
            skip_synthetic=args.skip_synthetic,
            debug=args.debug,
            confidence_threshold=args.threshold,
            seed=args.seed,
            n_categories=args.n_categories,
        )
