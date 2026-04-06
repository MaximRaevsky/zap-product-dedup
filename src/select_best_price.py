"""
Select the best (lowest) price for each product cluster and export results.
"""

import pandas as pd
from pathlib import Path


def select_and_export(
    clusters: pd.DataFrame,
    output_path: str = "results/grouped_products.csv",
) -> pd.DataFrame:
    """Export grouped products with best prices."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    export_df = clusters[[
        "cluster_id", "canonical_title", "min_price", "all_titles",
        "all_prices", "num_listings", "cluster_confidence", "category",
    ]].copy()

    export_df = export_df.sort_values(["category", "min_price"], na_position="last")
    export_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nGrouped products exported to {output_path}")
    print(f"  Total clusters: {len(export_df)}")
    print(f"  Multi-listing clusters: {(export_df['num_listings'] > 1).sum()}")
    print(f"  Avg listings per multi-cluster: {export_df[export_df['num_listings'] > 1]['num_listings'].mean():.1f}")

    # Summary by category
    for cat in export_df["category"].unique():
        cat_df = export_df[export_df["category"] == cat]
        multi = cat_df[cat_df["num_listings"] > 1]
        print(f"  {cat}: {len(cat_df)} clusters ({len(multi)} grouped)")

    return export_df
