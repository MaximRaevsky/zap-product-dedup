"""
Consolidate raw scraped Zap data into a clean seed dataset.
Groups listings by model_id and picks a canonical title and best (lowest) price.
"""

import pandas as pd
from pathlib import Path


def build_seed_dataset(
    raw_path: str = "data/raw/zap_listings.csv",
    output_path: str = "data/processed/seed_products.csv",
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    print(f"Loaded {len(df)} raw listings")

    seeds = []
    for (model_id, category), group in df.groupby(["model_id", "category"]):
        comparison = group[group["source_type"] == "comparison"]
        canonical_title = comparison.iloc[0]["raw_title"] if len(comparison) > 0 else group.iloc[0]["raw_title"]

        all_titles = group["raw_title"].unique().tolist()
        prices = group["price"].dropna().tolist()
        min_price = min(prices) if prices else None

        seeds.append({
            "model_id": model_id,
            "category": category,
            "canonical_title": canonical_title,
            "all_titles": " ||| ".join(all_titles),
            "num_variants": len(all_titles),
            "min_price": min_price,
            "all_prices": ", ".join(str(p) for p in sorted(set(prices))),
            "num_sources": len(group),
        })

    seed_df = pd.DataFrame(seeds)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved {len(seed_df)} seed products to {output_path}")
    print(f"  Categories: {seed_df['category'].value_counts().to_dict()}")
    print(f"  Products with >1 title variant: {(seed_df['num_variants'] > 1).sum()}")

    return seed_df


if __name__ == "__main__":
    build_seed_dataset()
