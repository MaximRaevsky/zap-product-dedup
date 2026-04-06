"""
Generate candidate pairs for deduplication comparison.
Uses blocking strategies: brand match, TF-IDF similarity, shared model number.
"""

import re
from itertools import combinations
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from normalize import normalize_title, normalize_for_comparison
from extract_attributes import extract_all_attributes


def generate_candidates(
    listings: pd.DataFrame,
    tfidf_threshold: float = 0.30,
    fuzzy_threshold: float = 60,
    max_candidates: int = 3000,
) -> list[dict]:
    """
    Generate candidate pairs using multiple blocking strategies.
    Returns list of candidate pair dicts with generation reason and confidence.
    """
    print("Extracting attributes and normalizing titles...")
    records = []
    for idx, row in listings.iterrows():
        attrs = extract_all_attributes(row["raw_title"])
        norm = normalize_title(row["raw_title"])
        norm_cmp = normalize_for_comparison(row["raw_title"])
        records.append({
            "idx": idx,
            "raw_title": row["raw_title"],
            "normalized": norm,
            "norm_compare": norm_cmp,
            "category": row.get("category", ""),
            "price": row.get("price"),
            "model_id": row.get("model_id", ""),
            **attrs,
        })

    rec_df = pd.DataFrame(records)
    candidates = {}

    # Strategy 1: Same brand blocking
    print("  Strategy 1: Brand blocking...")
    for brand, group in rec_df[rec_df["brand"].notna()].groupby("brand"):
        if len(group) < 2:
            continue
        idxs = group["idx"].tolist()
        for i, j in combinations(range(len(idxs)), 2):
            a_idx, b_idx = idxs[i], idxs[j]
            key = (min(a_idx, b_idx), max(a_idx, b_idx))
            if key not in candidates:
                ratio = fuzz.token_sort_ratio(
                    group.iloc[i]["norm_compare"],
                    group.iloc[j]["norm_compare"],
                )
                if ratio > fuzzy_threshold:
                    candidates[key] = {
                        "idx_a": key[0], "idx_b": key[1],
                        "reason": "brand_block",
                        "candidate_confidence": min(0.3 + ratio / 200, 0.8),
                        "fuzzy_ratio": ratio,
                    }

    # Strategy 2: TF-IDF cosine similarity
    print("  Strategy 2: TF-IDF similarity...")
    norm_titles = rec_df["norm_compare"].tolist()
    if len(norm_titles) > 1:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        tfidf_matrix = vectorizer.fit_transform(norm_titles)
        sim_matrix = cosine_similarity(tfidf_matrix)

        for i in range(len(norm_titles)):
            for j in range(i + 1, len(norm_titles)):
                if sim_matrix[i, j] >= tfidf_threshold:
                    a_idx = rec_df.iloc[i]["idx"]
                    b_idx = rec_df.iloc[j]["idx"]
                    key = (min(a_idx, b_idx), max(a_idx, b_idx))
                    if key not in candidates:
                        candidates[key] = {
                            "idx_a": key[0], "idx_b": key[1],
                            "reason": "tfidf_similarity",
                            "candidate_confidence": min(sim_matrix[i, j], 0.9),
                            "tfidf_score": float(sim_matrix[i, j]),
                        }

    # Strategy 3: Shared model number
    print("  Strategy 3: Model number matching...")
    model_groups = {}
    for _, row in rec_df[rec_df["model_number"].notna()].iterrows():
        mn = row["model_number"]
        model_groups.setdefault(mn, []).append(row["idx"])

    for mn, idxs in model_groups.items():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            key = (min(i, j), max(i, j))
            if key not in candidates:
                candidates[key] = {
                    "idx_a": key[0], "idx_b": key[1],
                    "reason": "model_number_match",
                    "candidate_confidence": 0.85,
                    "shared_model": mn,
                }

    result = list(candidates.values())
    if len(result) > max_candidates:
        result.sort(key=lambda x: x.get("candidate_confidence", 0), reverse=True)
        result = result[:max_candidates]

    print(f"  Generated {len(result)} candidate pairs")
    reason_counts = {}
    for c in result:
        reason_counts[c["reason"]] = reason_counts.get(c["reason"], 0) + 1
    for reason, count in reason_counts.items():
        print(f"    {reason}: {count}")

    return result, rec_df


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic/all_listings.csv")
    candidates, rec_df = generate_candidates(df)
    print(f"\nSample candidates:")
    for c in candidates[:5]:
        a = rec_df[rec_df["idx"] == c["idx_a"]].iloc[0]
        b = rec_df[rec_df["idx"] == c["idx_b"]].iloc[0]
        print(f"  [{c['reason']}] {a['raw_title'][:50]} <-> {b['raw_title'][:50]}")
