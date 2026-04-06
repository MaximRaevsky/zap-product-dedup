"""
Generate candidate pairs for deduplication using blocking strategies:
brand match, TF-IDF character similarity, shared model number.
"""

from itertools import combinations

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
    max_candidates: int = 10000,
) -> tuple[list[dict], pd.DataFrame]:
    print("Extracting attributes and normalizing...")
    records = []
    for idx, row in listings.iterrows():
        attrs = extract_all_attributes(row["raw_title"])
        records.append({
            "idx": idx,
            "raw_title": row["raw_title"],
            "normalized": normalize_title(row["raw_title"]),
            "norm_compare": normalize_for_comparison(row["raw_title"]),
            "category": row.get("category", ""),
            "price": row.get("price"),
            "model_id": row.get("model_id", ""),
            **attrs,
        })

    rec_df = pd.DataFrame(records)
    candidates = {}

    # Strategy 0: model_id blocking (strongest signal -- same product on Zap)
    if "model_id" in rec_df.columns:
        print("  Model ID blocking...")
        for mid, group in rec_df[rec_df["model_id"].notna() & (rec_df["model_id"] != "")].groupby("model_id"):
            if len(group) < 2:
                continue
            idxs = group["idx"].tolist()
            for i, j in combinations(idxs, 2):
                key = (min(i, j), max(i, j))
                if key not in candidates:
                    candidates[key] = {"idx_a": key[0], "idx_b": key[1],
                                       "reason": "model_id_block", "candidate_confidence": 0.90}

    # Brand blocking + fuzzy filter
    print("  Brand blocking...")
    for _, group in rec_df[rec_df["brand"].notna()].groupby("brand"):
        if len(group) < 2:
            continue
        idxs = group["idx"].tolist()
        for i, j in combinations(range(len(idxs)), 2):
            ratio = fuzz.token_sort_ratio(group.iloc[i]["norm_compare"], group.iloc[j]["norm_compare"])
            if ratio > fuzzy_threshold:
                key = (min(idxs[i], idxs[j]), max(idxs[i], idxs[j]))
                if key not in candidates:
                    candidates[key] = {"idx_a": key[0], "idx_b": key[1],
                                       "reason": "brand_block",
                                       "candidate_confidence": min(0.3 + ratio / 200, 0.8)}

    # TF-IDF similarity
    print("  TF-IDF similarity...")
    titles = rec_df["norm_compare"].tolist()
    if len(titles) > 1:
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        sim = cosine_similarity(tfidf.fit_transform(titles))
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                if sim[i, j] >= tfidf_threshold:
                    a, b = int(rec_df.iloc[i]["idx"]), int(rec_df.iloc[j]["idx"])
                    key = (min(a, b), max(a, b))
                    if key not in candidates:
                        candidates[key] = {"idx_a": key[0], "idx_b": key[1],
                                           "reason": "tfidf", "candidate_confidence": min(sim[i, j], 0.9)}

    # Shared model number
    print("  Model number matching...")
    model_groups: dict[str, list] = {}
    for _, row in rec_df[rec_df["model_number"].notna()].iterrows():
        model_groups.setdefault(row["model_number"], []).append(row["idx"])
    for idxs in model_groups.values():
        if len(idxs) < 2:
            continue
        for i, j in combinations(idxs, 2):
            key = (min(i, j), max(i, j))
            if key not in candidates:
                candidates[key] = {"idx_a": key[0], "idx_b": key[1],
                                   "reason": "model_match", "candidate_confidence": 0.85}

    # Model_id pairs are known-good; only cap discovery-based pairs
    model_id_pairs = [c for c in candidates.values() if c["reason"] == "model_id_block"]
    discovery_pairs = [c for c in candidates.values() if c["reason"] != "model_id_block"]
    if len(discovery_pairs) > max_candidates:
        discovery_pairs.sort(key=lambda x: x.get("candidate_confidence", 0), reverse=True)
        discovery_pairs = discovery_pairs[:max_candidates]
    result = model_id_pairs + discovery_pairs

    print(f"  {len(result)} candidate pairs ({len(model_id_pairs)} from model_id)")
    counts = {}
    for c in result:
        counts[c["reason"]] = counts.get(c["reason"], 0) + 1
    for r, n in counts.items():
        print(f"    {r}: {n}")

    return result, rec_df
