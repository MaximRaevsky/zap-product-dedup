"""
Scrape product listings from Zap public pages.
Discovers all categories dynamically and samples randomly for diversity.
Also scrapes individual product comparison pages for real per-store title variants.
"""

import re
import random
import time
import requests
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Optional

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
}


def _parse_price(text: str) -> Optional[float]:
    text = text.replace(",", "").replace("₪", "").strip()
    text = re.sub(r"[\u200e\u200f\u202a-\u202e]", "", text)
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def _find_price_near(tag) -> Optional[float]:
    for ancestor in [tag.parent, getattr(tag.parent, "parent", None)]:
        if ancestor is None:
            continue
        txt = ancestor.get_text(" ", strip=True)
        m = re.search(r"(?:החל מ[- ]*)?(\d[\d,]*(?:\.\d+)?)\s*₪", txt)
        if m:
            return _parse_price(m.group(0))
    return None


def _model_id_from_url(url: str) -> Optional[str]:
    m = re.search(r"modelid=(\d+)", url)
    return m.group(1) if m else None


def discover_categories() -> dict[str, str]:
    """Parse all available category sog codes from the Zap homepage."""
    try:
        resp = requests.get("https://www.zap.co.il", headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  WARNING: Could not fetch homepage for category discovery: {e}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    categories = {}
    for a in soup.find_all("a", href=True):
        m = re.search(r"models\.aspx\?sog=([a-z]+-\w+)", a["href"])
        if m:
            sog = m.group(1)
            name = a.get_text(strip=True)
            if name and len(name) > 1 and sog not in categories:
                categories[sog] = name
    return categories


def sample_categories(all_cats: dict[str, str], n: int = 10, seed: Optional[int] = None) -> dict[str, str]:
    """Randomly sample n categories for diversity."""
    if seed is not None:
        random.seed(seed)
    keys = list(all_cats.keys())
    random.shuffle(keys)
    selected = keys[:min(n, len(keys))]
    return {k: all_cats[k] for k in selected}


def scrape_category_page(sog: str, category_name: str) -> list[dict]:
    """Scrape a single Zap category listing page."""
    url = f"https://www.zap.co.il/models.aspx?sog={sog}"
    print(f"  Fetching {category_name}: {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if not text or len(text) < 5:
            continue

        full_url = href if href.startswith("http") else f"https://www.zap.co.il{href}"

        if "/model.aspx?" in href and "modelid=" in href:
            model_id = _model_id_from_url(href)
            if not model_id:
                continue
            if any(kw in text for kw in ["השוואת מחירים", "השוואה ב", "חוות דעת", "לפרטים", "ציון"]):
                continue
            if re.fullmatch(r"[\d.\s()]+", text):
                continue
            records.append({
                "raw_title": text,
                "price": _find_price_near(a),
                "model_id": model_id,
                "category": category_name,
                "sog": sog,
                "source_url": full_url,
                "source_type": "comparison",
            })

        elif "shop.zap.co.il" in href and "modelid=" in href:
            model_id = _model_id_from_url(href)
            if not model_id:
                continue
            if any(kw in text for kw in ["קנו עכשיו", "לפרטים", "קנו ב"]):
                continue
            price = None
            pm = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*₪", text)
            if pm:
                price = _parse_price(pm.group(0))
                text = re.sub(r"\d[\d,]*(?:\.\d+)?\s*₪.*", "", text).strip()
            if not price:
                price = _find_price_near(a)
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
    unique = []
    for r in records:
        key = (r["raw_title"], r["model_id"], r["source_type"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    print(f"  Found {len(unique)} listings in {category_name}")
    return unique


def scrape_product_page(model_id: str, category_name: str) -> list[dict]:
    """Scrape an individual product comparison page for per-store title variants."""
    url = f"https://www.zap.co.il/model.aspx?modelid={model_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []

    nav_noise = ["דלג ל", "תפריט", "חזור ל", "כל הזכויות", "copyright", "שאלות"]
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)
        if not text or len(text) < 8:
            continue
        if "clientcard" in href or "ratemodel" in href:
            continue
        if any(kw in text for kw in ["השוואת", "חוות דעת", "קנו עכשיו", "לפרטים", "ציון"]):
            continue
        if any(text.startswith(n) for n in nav_noise):
            continue
        if re.fullmatch(r"[\d.,\s₪()]+", text):
            continue

        price = None
        pm = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*₪", text)
        if pm:
            price = _parse_price(pm.group(0))
            text_clean = re.sub(r"\d[\d,]*(?:\.\d+)?\s*₪.*", "", text).strip()
            if len(text_clean) >= 8:
                text = text_clean

        if not price:
            price = _find_price_near(a)

        if len(text) < 8 or text.startswith("http"):
            continue

        records.append({
            "raw_title": text,
            "price": price,
            "model_id": model_id,
            "category": category_name,
            "sog": "",
            "source_url": url,
            "source_type": "store_variant",
        })

    seen = set()
    unique = []
    for r in records:
        if r["raw_title"] not in seen:
            seen.add(r["raw_title"])
            unique.append(r)
    return unique


def scrape_all(
    n_categories: int = 10,
    n_product_pages: int = 3,
    seed: Optional[int] = None,
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Scrape Zap: discover all categories, randomly sample n_categories,
    and for each category also scrape n_product_pages individual product pages
    to get real per-store title variants.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Discovering all Zap categories...")
    all_cats = discover_categories()
    if not all_cats:
        print("  Falling back to a small default set")
        all_cats = {"e-cellphone": "טלפונים סלולריים", "e-tv": "טלויזיות", "e-headphone": "אוזניות"}

    print(f"  Found {len(all_cats)} categories")
    selected = sample_categories(all_cats, n=n_categories, seed=seed)
    print(f"  Sampled {len(selected)}: {list(selected.values())}")

    all_records = []
    for sog, name in selected.items():
        records = scrape_category_page(sog, name)
        all_records.extend(records)

        # Pick a few model_ids from this category to scrape per-store variants
        model_ids = list({r["model_id"] for r in records if r["source_type"] == "comparison"})
        random.shuffle(model_ids)
        for mid in model_ids[:n_product_pages]:
            store_variants = scrape_product_page(mid, name)
            all_records.extend(store_variants)
            time.sleep(0.5)

        time.sleep(1)

    df = pd.DataFrame(all_records)
    csv_path = output_path / "zap_listings.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved {len(df)} total listings to {csv_path}")
    print(f"  Categories: {df['category'].nunique()}")
    print(f"  Source types: {df['source_type'].value_counts().to_dict()}")
    print(f"  With price: {df['price'].notna().sum()}/{len(df)}")

    return df


if __name__ == "__main__":
    scrape_all()
