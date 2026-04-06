"""
Generate synthetic noisy duplicates and hard negatives from seed products.
All transforms are category-agnostic.
"""

import random
import re
import pandas as pd
from pathlib import Path
from typing import Optional

random.seed(42)

# Hebrew-English brand swaps (the same table used in extract_attributes)
BRAND_HEB = {
    "Apple": "אפל", "Samsung": "סמסונג", "Xiaomi": "שיאומי",
    "Sony": "סוני", "LG": "אל ג'י", "Google": "גוגל",
    "Nespresso": "נספרסו", "DeLonghi": "דלונגי", "Philips": "פיליפס",
    "Lenovo": "לנובו", "HP": "אייץ' פי", "Dell": "דל",
    "ASUS": "אסוס", "Hisense": "היסנס", "TCL": "טי סי אל",
    "Bosch": "בוש", "Dyson": "דייסון", "Tefal": "טפאל",
}
BRAND_ENG = {v: k for k, v in BRAND_HEB.items()}

SELLER_NOISE = ["יבואן רשמי", "אחריות יבואן רשמי", "משלוח חינם", "יבואן מורשה"]
COLOR_NOISE = ["בצבע שחור", "בצבע לבן", "בצבע כסף", "Black", "White", "Silver"]


def _swap_brand(title: str) -> Optional[str]:
    for eng, heb in BRAND_HEB.items():
        if eng in title:
            return title.replace(eng, heb, 1)
        if heb in title:
            return title.replace(heb, eng, 1)
    return None


def _reorder_tokens(title: str) -> str:
    tokens = title.split()
    if len(tokens) < 3:
        return title
    mid = len(tokens) // 2
    return " ".join(tokens[mid:] + tokens[:mid])


def _drop_sku(title: str) -> str:
    """Drop any alphanumeric SKU-like token (generic, not Samsung-specific)."""
    return re.sub(r"\b[A-Z]{2,5}-?[A-Z0-9]{3,}(?:/\w+)?\b", "", title).strip()


def _add_seller_noise(title: str) -> str:
    return title + " " + random.choice(SELLER_NOISE)


def _add_color(title: str) -> str:
    return title + " " + random.choice(COLOR_NOISE)


def _abbreviate(title: str) -> str:
    for old, new in [("True Wireless", "TW"), ("Bluetooth", "BT"), ("אינטש", '"')]:
        if old in title:
            return title.replace(old, new, 1)
    return title


def _jitter_price(price: float) -> float:
    return round(price * random.uniform(0.85, 1.25))


TRANSFORMS = [
    ("brand_swap", _swap_brand),
    ("reorder", _reorder_tokens),
    ("drop_sku", _drop_sku),
    ("seller_noise", _add_seller_noise),
    ("color_noise", _add_color),
    ("abbreviate", _abbreviate),
]


def generate_noisy_duplicate(row: pd.Series) -> list[dict]:
    title = row["canonical_title"]
    price = row["min_price"]

    shuffled = list(TRANSFORMS)
    random.shuffle(shuffled)
    variants = []
    for name, fn in shuffled[:random.randint(2, 4)]:
        new_title = fn(title)
        if new_title and new_title != title and len(new_title) > 5:
            variants.append({
                "raw_title": new_title,
                "price": _jitter_price(price) if price else None,
                "model_id": row["model_id"],
                "category": row["category"],
                "source_type": "synthetic_duplicate",
                "transform": name,
                "original_title": title,
            })
    return variants


def generate_hard_negatives(seed_df: pd.DataFrame) -> list[dict]:
    negatives = []
    for _, group in seed_df.groupby("category"):
        titles = group["canonical_title"].tolist()
        mids = group["model_id"].tolist()
        prices = group["min_price"].tolist()
        for i in range(len(titles)):
            for j in range(i + 1, min(i + 3, len(titles))):
                negatives.append({
                    "title_a": titles[i], "price_a": prices[i], "model_id_a": mids[i],
                    "title_b": titles[j], "price_b": prices[j], "model_id_b": mids[j],
                    "category": group.iloc[0]["category"],
                    "is_duplicate": False, "difficulty": "hard_negative",
                })
    return negatives


def generate_all(
    seed_path: str = "data/processed/seed_products.csv",
    output_dir: str = "data/synthetic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_df = pd.read_csv(seed_path)
    print(f"Loaded {len(seed_df)} seed products")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_variants = []
    for _, row in seed_df.iterrows():
        if pd.isna(row["min_price"]):
            continue
        all_variants.extend(generate_noisy_duplicate(row))
    variant_df = pd.DataFrame(all_variants)
    print(f"Generated {len(variant_df)} synthetic variants")

    original_listings = []
    for _, row in seed_df.iterrows():
        original_listings.append({
            "raw_title": row["canonical_title"],
            "price": row["min_price"],
            "model_id": row["model_id"],
            "category": row["category"],
            "source_type": "seed",
            "transform": "original",
            "original_title": row["canonical_title"],
        })
        if " ||| " in str(row.get("all_titles", "")):
            for alt in str(row["all_titles"]).split(" ||| ")[1:]:
                original_listings.append({
                    "raw_title": alt,
                    "price": row["min_price"],
                    "model_id": row["model_id"],
                    "category": row["category"],
                    "source_type": "zap_variant",
                    "transform": "real_variant",
                    "original_title": row["canonical_title"],
                })

    all_listings = pd.concat([pd.DataFrame(original_listings), variant_df], ignore_index=True)
    all_listings.to_csv(f"{output_dir}/all_listings.csv", index=False, encoding="utf-8-sig")
    print(f"Total listings: {len(all_listings)}")

    # Build evaluation pairs
    pairs = []
    for _, v in variant_df.iterrows():
        pairs.append({
            "title_a": v["original_title"], "price_a": v["price"],
            "title_b": v["raw_title"], "price_b": v["price"],
            "model_id_a": v["model_id"], "model_id_b": v["model_id"],
            "category": v["category"], "is_duplicate": True, "difficulty": v["transform"],
        })
    for _, row in seed_df.iterrows():
        if " ||| " not in str(row.get("all_titles", "")):
            continue
        titles = str(row["all_titles"]).split(" ||| ")
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                pairs.append({
                    "title_a": titles[i], "price_a": row["min_price"],
                    "title_b": titles[j], "price_b": row["min_price"],
                    "model_id_a": row["model_id"], "model_id_b": row["model_id"],
                    "category": row["category"], "is_duplicate": True, "difficulty": "real_variant",
                })
    pairs.extend(generate_hard_negatives(seed_df))

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(f"{output_dir}/evaluation_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"Eval pairs: {len(pairs_df)} ({pairs_df['is_duplicate'].sum()} pos, {(~pairs_df['is_duplicate']).sum()} neg)")
    return all_listings, pairs_df
