"""
Cluster duplicate products using Union-Find and select canonical representatives.
"""

import pandas as pd
from typing import Optional


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def build_clusters(
    pairs: list[dict],
    listings: pd.DataFrame,
    confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build product clusters from duplicate pair predictions.
    Returns a DataFrame of clusters with canonical names and min prices.
    """
    uf = UnionFind()

    dup_pairs = [
        p for p in pairs
        if p.get("is_duplicate_pred") and p.get("final_confidence", 0) >= confidence_threshold
    ]
    print(f"Building clusters from {len(dup_pairs)} confirmed duplicate pairs")

    for p in dup_pairs:
        uf.union(p["idx_a"], p["idx_b"])

    # Also ensure all listing indices are in the UF (singletons)
    for idx in listings.index:
        uf.find(idx)

    # Group by cluster root
    clusters = {}
    for idx in listings.index:
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

    # Build cluster records
    cluster_records = []
    for cluster_id, (root, members) in enumerate(sorted(clusters.items(), key=lambda x: -len(x[1]))):
        member_rows = listings.loc[members]
        titles = member_rows["raw_title"].tolist()
        prices = member_rows["price"].dropna().tolist()
        categories = member_rows["category"].unique().tolist()

        # Pick canonical title: prefer comparison source, longest title
        comparison_rows = member_rows[member_rows.get("source_type", pd.Series(dtype=str)) == "comparison"] if "source_type" in member_rows.columns else member_rows
        if len(comparison_rows) > 0:
            canonical = comparison_rows.loc[comparison_rows["raw_title"].str.len().idxmax(), "raw_title"]
        else:
            canonical = member_rows.loc[member_rows["raw_title"].str.len().idxmax(), "raw_title"]

        min_price = min(prices) if prices else None

        # Cluster confidence: minimum pairwise confidence among members
        member_set = set(members)
        relevant_confs = [
            p["final_confidence"]
            for p in dup_pairs
            if p["idx_a"] in member_set and p["idx_b"] in member_set
        ]
        cluster_conf = min(relevant_confs) if relevant_confs else 1.0

        cluster_records.append({
            "cluster_id": cluster_id,
            "canonical_title": canonical,
            "min_price": min_price,
            "all_titles": " ||| ".join(titles),
            "all_prices": ", ".join(str(p) for p in sorted(set(prices))) if prices else "",
            "num_listings": len(members),
            "cluster_confidence": round(cluster_conf, 4),
            "category": categories[0] if categories else "",
            "member_indices": members,
        })

    result_df = pd.DataFrame(cluster_records)
    multi = result_df[result_df["num_listings"] > 1]
    print(f"Formed {len(result_df)} clusters ({len(multi)} with >1 listing)")

    return result_df
