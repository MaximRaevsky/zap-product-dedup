"""
Generate synthetic noisy duplicates and hard negatives from seed products
for robust evaluation of the deduplication pipeline.
"""

import random
import re
import pandas as pd
from pathlib import Path
from typing import Optional

random.seed(42)

BRAND_HEB_MAP = {
    "Apple": "אפל",
    "Samsung": "סמסונג",
    "Xiaomi": "שיאומי",
    "Sony": "סוני",
    "LG": "אל ג'י",
    "Bose": "בוז",
    "JBL": "ג'יי בי אל",
    "Google": "גוגל",
    "Nespresso": "נספרסו",
    "DeLonghi": "דלונגי",
    "Philips": "פיליפס",
    "Breville": "ברוויל",
    "Lenovo": "לנובו",
    "HP": "אייץ' פי",
    "Dell": "דל",
    "Asus": "אסוס",
    "Hisense": "היסנס",
    "TCL": "טי סי אל",
}

BRAND_ENG_MAP = {v: k for k, v in BRAND_HEB_MAP.items()}

CATEGORY_PREFIXES_HEB = {
    "smartphones": ["טלפון סלולרי", "סלולרי", "טלפון חכם", "מכשיר סלולרי"],
    "headphones": ["אוזניות", "אוזניות אלחוטיות", "אוזניות בלוטות'"],
    "tvs": ["טלוויזיה", "טלויזיה", "מסך", "TV"],
    "laptops": ["מחשב נייד", "לפטופ", "נייד"],
    "coffee_machines": ["מכונת קפה", "מכונת אספרסו", "מכונת קפה אוטומטית"],
}

SELLER_NOISE = [
    "יבואן רשמי",
    "אחריות יבואן רשמי",
    "משלוח חינם",
    "יבואן מורשה",
    "DCS",
    "אחריות מעבדות",
]

COLOR_NOISE = [
    "בצבע שחור", "בצבע לבן", "בצבע כסף", "בצבע כחול",
    "בצבע אפור", "Black", "White", "Silver", "Blue",
]


def _swap_brand_language(title: str) -> Optional[str]:
    for eng, heb in BRAND_HEB_MAP.items():
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


def _drop_part_number(title: str) -> str:
    return re.sub(r"\bSM-\w+\b", "", title).strip()


def _drop_category_prefix(title: str, category: str) -> str:
    prefixes = CATEGORY_PREFIXES_HEB.get(category, [])
    for prefix in prefixes:
        if title.startswith(prefix):
            return title[len(prefix):].strip()
    return title


def _add_category_prefix(title: str, category: str) -> str:
    prefixes = CATEGORY_PREFIXES_HEB.get(category, [])
    if prefixes:
        return random.choice(prefixes) + " " + title
    return title


def _add_seller_noise(title: str) -> str:
    return title + " " + random.choice(SELLER_NOISE)


def _add_color_noise(title: str) -> str:
    return title + " " + random.choice(COLOR_NOISE)


def _abbreviate(title: str) -> str:
    replacements = [
        ("True Wireless", "TW"),
        ("אלחוטיות", "אלחוטי"),
        ("Bluetooth", "BT"),
        ("אינטש", "\""),
        ("Ultra", "U"),
    ]
    result = title
    for old, new in replacements:
        if old in result:
            result = result.replace(old, new, 1)
            break
    return result


def _jitter_price(price: float) -> float:
    offset = random.uniform(-0.15, 0.25) * price
    return round(price + offset, 0)


def generate_noisy_duplicate(row: pd.Series) -> list[dict]:
    """Generate 2-4 noisy duplicate variants of a seed product."""
    title = row["canonical_title"]
    category = row["category"]
    price = row["min_price"]

    transforms = [
        ("brand_swap", _swap_brand_language),
        ("reorder", lambda t: _reorder_tokens(t)),
        ("drop_part", lambda t: _drop_part_number(t)),
        ("drop_prefix", lambda t: _drop_category_prefix(t, category)),
        ("add_prefix", lambda t: _add_category_prefix(
            _drop_category_prefix(t, category), category)),
        ("seller_noise", lambda t: _add_seller_noise(t)),
        ("color_noise", lambda t: _add_color_noise(t)),
        ("abbreviate", lambda t: _abbreviate(t)),
    ]

    random.shuffle(transforms)
    variants = []
    num_variants = random.randint(2, 4)

    for name, fn in transforms[:num_variants]:
        new_title = fn(title)
        if new_title and new_title != title and len(new_title) > 5:
            new_price = _jitter_price(price) if price else None
            variants.append({
                "raw_title": new_title,
                "price": new_price,
                "model_id": row["model_id"],
                "category": category,
                "source_type": "synthetic_duplicate",
                "transform": name,
                "original_title": title,
            })

    return variants


def generate_hard_negatives(seed_df: pd.DataFrame) -> list[dict]:
    """Generate hard negatives: similar but non-identical products."""
    negatives = []

    for category, group in seed_df.groupby("category"):
        titles = group["canonical_title"].tolist()
        model_ids = group["model_id"].tolist()
        prices = group["min_price"].tolist()

        for i in range(len(titles)):
            for j in range(i + 1, min(i + 3, len(titles))):
                negatives.append({
                    "title_a": titles[i],
                    "price_a": prices[i],
                    "model_id_a": model_ids[i],
                    "title_b": titles[j],
                    "price_b": prices[j],
                    "model_id_b": model_ids[j],
                    "category": category,
                    "is_duplicate": False,
                    "difficulty": "hard_negative",
                })

    return negatives


def generate_all(
    seed_path: str = "data/processed/seed_products.csv",
    output_dir: str = "data/synthetic",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_df = pd.read_csv(seed_path)
    print(f"Loaded {len(seed_df)} seed products")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate noisy duplicates
    all_variants = []
    for _, row in seed_df.iterrows():
        if pd.isna(row["min_price"]):
            continue
        all_variants.extend(generate_noisy_duplicate(row))

    variant_df = pd.DataFrame(all_variants)
    print(f"Generated {len(variant_df)} synthetic duplicate variants")

    # Build full listing set: original seed titles + synthetic variants
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
            for alt_title in str(row["all_titles"]).split(" ||| ")[1:]:
                original_listings.append({
                    "raw_title": alt_title,
                    "price": row["min_price"],
                    "model_id": row["model_id"],
                    "category": row["category"],
                    "source_type": "zap_variant",
                    "transform": "real_variant",
                    "original_title": row["canonical_title"],
                })

    original_df = pd.DataFrame(original_listings)
    all_listings = pd.concat([original_df, variant_df], ignore_index=True)
    all_listings.to_csv(f"{output_dir}/all_listings.csv", index=False, encoding="utf-8-sig")
    print(f"Total listings (seed + real variants + synthetic): {len(all_listings)}")

    # Build evaluation pairs
    pairs = []

    # Positive pairs: each synthetic variant paired with its original
    for _, v in variant_df.iterrows():
        pairs.append({
            "title_a": v["original_title"],
            "price_a": v["price"],
            "title_b": v["raw_title"],
            "price_b": v["price"],
            "model_id_a": v["model_id"],
            "model_id_b": v["model_id"],
            "category": v["category"],
            "is_duplicate": True,
            "difficulty": v["transform"],
        })

    # Real variant pairs (same model_id, different title from Zap)
    for _, row in seed_df.iterrows():
        if " ||| " not in str(row.get("all_titles", "")):
            continue
        titles = str(row["all_titles"]).split(" ||| ")
        for i in range(len(titles)):
            for j in range(i + 1, len(titles)):
                pairs.append({
                    "title_a": titles[i],
                    "price_a": row["min_price"],
                    "title_b": titles[j],
                    "price_b": row["min_price"],
                    "model_id_a": row["model_id"],
                    "model_id_b": row["model_id"],
                    "category": row["category"],
                    "is_duplicate": True,
                    "difficulty": "real_variant",
                })

    # Hard negatives
    hard_negs = generate_hard_negatives(seed_df)
    pairs.extend(hard_negs)

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_csv(f"{output_dir}/evaluation_pairs.csv", index=False, encoding="utf-8-sig")
    print(f"Evaluation pairs: {len(pairs_df)} ({pairs_df['is_duplicate'].sum()} positive, {(~pairs_df['is_duplicate']).sum()} negative)")

    return all_listings, pairs_df


if __name__ == "__main__":
    generate_all()
