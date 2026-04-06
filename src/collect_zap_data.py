"""
Scrape product listings from Zap public category pages.
Extracts both comparison listings and zapstore direct-buy listings.
"""

import re
import json
import time
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional

CATEGORIES = {
    "e-cellphone": "smartphones",
    "e-headphone": "headphones",
    "e-tv": "tvs",
    "c-pclaptop": "laptops",
    "e-coffeemachine": "coffee_machines",
}

BASE_URL = "https://www.zap.co.il/models.aspx?sog={sog}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _parse_price(text: str) -> Optional[float]:
    """Extract numeric price from Hebrew price strings."""
    text = text.replace(",", "").replace("₪", "").strip()
    text = text.replace("\u202b", "").replace("\u200f", "")
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    return None


def _find_price_in_ancestors(tag) -> Optional[float]:
    """Walk up the DOM tree to find price text near a product link."""
    for ancestor in [tag.parent, getattr(tag.parent, "parent", None)]:
        if ancestor is None:
            continue
        txt = ancestor.get_text(" ", strip=True)
        m = re.search(r"החל מ[- ]*(\d[\d,]*(?:\.\d+)?)\s*₪", txt)
        if m:
            return _parse_price(m.group(0))
        m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*₪", txt)
        if m:
            return _parse_price(m.group(0))
    return None


def _extract_model_id(url: str) -> Optional[str]:
    m = re.search(r"modelid=(\d+)", url)
    return m.group(1) if m else None


def scrape_category(sog: str, category_name: str) -> list[dict]:
    """Scrape a single Zap category page and return product records."""
    url = BASE_URL.format(sog=sog)
    print(f"  Fetching {category_name}: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR fetching {category_name}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        text = a_tag.get_text(strip=True)
        if not text or len(text) < 5:
            continue

        full_url = href if href.startswith("http") else f"https://www.zap.co.il{href}"

        if "/model.aspx?" in href and "modelid=" in href:
            model_id = _extract_model_id(href)
            if not model_id:
                continue

            if any(kw in text for kw in ["השוואת מחירים", "השוואה ב", "חוות דעת", "לפרטים", "ציון"]):
                continue
            if re.fullmatch(r"[\d.\s()]+", text):
                continue

            price = _find_price_in_ancestors(a_tag)

            records.append({
                "raw_title": text,
                "price": price,
                "model_id": model_id,
                "category": category_name,
                "sog": sog,
                "source_url": full_url,
                "source_type": "comparison",
            })

        elif "shop.zap.co.il" in href and "modelid=" in href:
            model_id = _extract_model_id(href)
            if not model_id:
                continue
            if any(kw in text for kw in ["קנו עכשיו", "לפרטים", "קנו ב"]):
                continue

            price = None
            price_match = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*₪", text)
            if price_match:
                price = _parse_price(price_match.group(0))
                text = re.sub(r"\d[\d,]*(?:\.\d+)?\s*₪.*", "", text).strip()

            if not price:
                price = _find_price_in_ancestors(a_tag)

            if len(text) < 5:
                continue

            records.append({
                "raw_title": text,
                "price": price,
                "model_id": model_id,
                "category": category_name,
                "sog": sog,
                "source_url": full_url,
                "source_type": "zapstore",
            })

    seen = set()
    unique_records = []
    for r in records:
        key = (r["raw_title"], r["model_id"], r["source_type"])
        if key not in seen:
            seen.add(key)
            unique_records.append(r)

    print(f"  Found {len(unique_records)} unique listings in {category_name}")
    return unique_records


def scrape_all(output_dir: str = "data/raw") -> pd.DataFrame:
    """Scrape all categories and save to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_records = []
    for sog, name in CATEGORIES.items():
        records = scrape_category(sog, name)
        all_records.extend(records)
        time.sleep(1.5)

    df = pd.DataFrame(all_records)
    csv_path = output_path / "zap_listings.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} total listings to {csv_path}")

    json_path = output_path / "zap_listings.json"
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    return df


if __name__ == "__main__":
    df = scrape_all()
    print(f"\nCategories: {df['category'].value_counts().to_dict()}")
    print(f"Source types: {df['source_type'].value_counts().to_dict()}")
    print(f"Rows with price: {df['price'].notna().sum()}/{len(df)}")
