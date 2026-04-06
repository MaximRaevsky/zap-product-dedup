"""
Generate synthetic noisy duplicates and hard negatives from seed products.
Uses LLM (gpt-4o-mini) to generate realistic title variants instead of
hardcoded brand/seller/color dictionaries.
"""

import os
import re
import json
import random
import pandas as pd
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _reorder_tokens(title: str) -> str:
    tokens = title.split()
    if len(tokens) < 3:
        return title
    mid = len(tokens) // 2
    return " ".join(tokens[mid:] + tokens[:mid])


def _drop_sku(title: str) -> str:
    return re.sub(r"\b[A-Z]{2,5}-?[A-Z0-9]{3,}(?:/\w+)?\b", "", title).strip()


def _jitter_price(price: float) -> float:
    return round(price * random.uniform(0.85, 1.25))


def _llm_generate_variants(titles: list[str]) -> dict[str, list[str]]:
    """
    Use gpt-4o-mini to generate realistic noisy title variants.
    Returns a dict mapping each original title to a list of variant titles.
    """
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(titles))
    prompt = f"""You are helping build a test dataset for product deduplication on an Israeli price comparison site.

For each product title below, generate 2 realistic variants as a different store might list the same product.
Apply a MIX of these transformations (not all at once):
- Swap brand name between Hebrew and English (e.g. "Bosch" -> "בוש", "אפל" -> "Apple")
- Add seller info in Hebrew (e.g. "יבואן רשמי", "משלוח חינם")
- Add a color descriptor (e.g. "בצבע שחור", "Black")
- Abbreviate terms (e.g. "Bluetooth" -> "BT")
- Reorder words

Return JSON object mapping index to array of variants.
Example: {{"0": ["variant1", "variant2"], "1": ["variant1", "variant2"]}}

Product titles:
{numbered}"""

    try:
        resp = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=2000,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        out = {}
        for k, v in result.items():
            try:
                idx = int(k)
            except ValueError:
                continue
            if 0 <= idx < len(titles) and isinstance(v, list):
                out[titles[idx]] = [s for s in v if isinstance(s, str)]
        return out
    except Exception as e:
        print(f"    LLM variant generation failed: {e}")
        return {}


def generate_noisy_duplicates_llm(seed_df: pd.DataFrame) -> list[dict]:
    """Generate synthetic variants using LLM in batches."""
    all_variants = []
    titles_with_meta = []

    for _, row in seed_df.iterrows():
        if pd.isna(row["min_price"]):
            continue
        titles_with_meta.append(row)

    # Batch titles for LLM calls
    batch_size = 8
    for i in range(0, len(titles_with_meta), batch_size):
        batch = titles_with_meta[i:i + batch_size]
        batch_titles = [r["canonical_title"] for r in batch]

        llm_variants = _llm_generate_variants(batch_titles)

        for row in batch:
            title = row["canonical_title"]
            variants = llm_variants.get(title, [])

            # Also add structural transforms (these are language-agnostic)
            reordered = _reorder_tokens(title)
            if reordered != title:
                variants.append(reordered)

            sku_dropped = _drop_sku(title)
            if sku_dropped and sku_dropped != title and len(sku_dropped) > 5:
                variants.append(sku_dropped)

            for v in variants:
                if v and v != title and len(v) > 5:
                    all_variants.append({
                        "raw_title": v,
                        "price": _jitter_price(row["min_price"]) if row["min_price"] else None,
                        "model_id": row["model_id"],
                        "category": row["category"],
                        "source_type": "synthetic_duplicate",
                        "transform": "llm_variant",
                        "original_title": title,
                    })

    return all_variants


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
    random.seed(42)
    seed_df = pd.read_csv(seed_path)
    print(f"Loaded {len(seed_df)} seed products")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("Generating LLM-based synthetic variants...")
    all_variants = generate_noisy_duplicates_llm(seed_df)
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
