"""
Extract structured attributes (brand, model, series, storage, screen size, color)
from product titles for use in rule-based matching.
"""

import re
from typing import Optional
from rapidfuzz import fuzz

BRAND_DICT = {
    "apple": "Apple", "אפל": "Apple",
    "samsung": "Samsung", "סמסונג": "Samsung",
    "xiaomi": "Xiaomi", "שיאומי": "Xiaomi",
    "sony": "Sony", "סוני": "Sony",
    "lg": "LG", "אל ג'י": "LG",
    "google": "Google", "גוגל": "Google",
    "bose": "Bose", "בוז": "Bose",
    "jbl": "JBL",
    "sennheiser": "Sennheiser",
    "anker": "Anker",
    "beats": "Beats",
    "audio technica": "Audio Technica",
    "beyerdynamic": "Beyerdynamic",
    "nespresso": "Nespresso", "נספרסו": "Nespresso",
    "delonghi": "DeLonghi", "דלונגי": "DeLonghi",
    "philips": "Philips", "פיליפס": "Philips",
    "breville": "Breville", "ברוויל": "Breville",
    "saeco": "Saeco",
    "lenovo": "Lenovo", "לנובו": "Lenovo",
    "hp": "HP",
    "dell": "Dell", "דל": "Dell",
    "asus": "ASUS", "אסוס": "ASUS",
    "acer": "Acer",
    "msi": "MSI",
    "hisense": "Hisense", "היסנס": "Hisense",
    "tcl": "TCL",
    "toshiba": "Toshiba",
    "oppo": "OPPO",
    "oneplus": "OnePlus",
    "nokia": "Nokia",
    "infinix": "Infinix",
    "vivo": "Vivo",
    "huawei": "Huawei",
    "realme": "Realme",
    "poco": "Poco",
    "redmi": "Redmi",
    "dreame": "Dreame",
    "roborock": "Roborock",
    "ecovacs": "Ecovacs",
}

MODEL_NUMBER_PATTERNS = [
    re.compile(r"\bSM-[A-Z]\d{3,}[A-Z]?(?:/\w+)?\b", re.IGNORECASE),
    re.compile(r"\b[A-Z]{2,4}\d{3,}[A-Z]{0,3}(?:/[A-Z]+)?\b"),
    re.compile(r"\bMXP\d{2}[A-Z]{2}/[A-Z]\b", re.IGNORECASE),
    re.compile(r"\bMFHP\d[A-Z]{2}/[A-Z]\b", re.IGNORECASE),
    re.compile(r"\b\d{2}[A-Z]\d{3}\b"),
]

SERIES_PATTERNS = [
    re.compile(r"\b(iPhone\s*\d+\s*(?:Pro\s*Max|Pro|Plus|Mini)?)\b", re.IGNORECASE),
    re.compile(r"\b(Galaxy\s*(?:S|A|Z|M)\d+\s*(?:Ultra|FE|Plus|\+)?)\b", re.IGNORECASE),
    re.compile(r"\b(AirPods\s*(?:Pro\s*\d*|Max|\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(Galaxy\s*Buds\s*\d*\s*(?:Pro|FE|Live|\+)?)\b", re.IGNORECASE),
    re.compile(r"\b(WH-\d{4}XM\d+)\b", re.IGNORECASE),
    re.compile(r"\b(Redmi\s*Buds\s*\d+\s*\w*)\b", re.IGNORECASE),
    re.compile(r"\b(ThinkPad\s*[A-Z]\d+\s*(?:Gen\s*\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(MacBook\s*(?:Air|Pro)\s*(?:\d+)?)\b", re.IGNORECASE),
    re.compile(r"\b(Vertuo\s*(?:Pop|Plus|Next|Lattissima)?)\b", re.IGNORECASE),
    re.compile(r"\b(Pixie|Citiz|Essenza|Inissia|Creatista)\b", re.IGNORECASE),
]

STORAGE_PATTERN = re.compile(r"\b(\d+)\s*(GB|TB)\b", re.IGNORECASE)
RAM_PATTERN = re.compile(r"\b(\d+)\s*GB\s*RAM\b", re.IGNORECASE)
COMBINED_STORAGE_PATTERN = re.compile(r"\b(\d+)GB\s*\+\s*(\d+)(GB|TB)\b", re.IGNORECASE)
SCREEN_SIZE_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:inch|אינטש|אינץ['\u2019]?|\")\b", re.IGNORECASE
)


def extract_brand(title: str) -> tuple[Optional[str], float]:
    """Extract brand name from title. Returns (brand, confidence)."""
    title_lower = title.lower()
    for key, canonical in BRAND_DICT.items():
        if key.lower() in title_lower:
            return canonical, 0.95
    # Fuzzy fallback on first 3 tokens
    tokens = title.split()[:5]
    for token in tokens:
        for key, canonical in BRAND_DICT.items():
            if fuzz.ratio(token.lower(), key.lower()) > 80:
                return canonical, 0.7
    return None, 0.0


def extract_model_number(title: str) -> tuple[Optional[str], float]:
    """Extract model/part number from title."""
    for pattern in MODEL_NUMBER_PATTERNS:
        m = pattern.search(title)
        if m:
            return m.group(0).upper(), 0.9
    return None, 0.0


def extract_series(title: str) -> tuple[Optional[str], float]:
    """Extract product series from title."""
    for pattern in SERIES_PATTERNS:
        m = pattern.search(title)
        if m:
            return m.group(1).strip(), 0.85
    return None, 0.0


def extract_storage(title: str) -> tuple[Optional[str], float]:
    """Extract storage capacity, excluding RAM. Handles '8GB+256GB' format."""
    # Handle combined format like "12GB+256GB" or "8GB+128GB"
    combined_match = COMBINED_STORAGE_PATTERN.search(title)
    if combined_match:
        storage_value = combined_match.group(2)
        storage_unit = combined_match.group(3).upper()
        return f"{storage_value}{storage_unit}", 0.9

    ram_match = RAM_PATTERN.search(title)
    ram_value = ram_match.group(1) if ram_match else None

    for m in STORAGE_PATTERN.finditer(title):
        value = m.group(1)
        unit = m.group(2).upper()
        if value == ram_value and unit == "GB":
            continue
        return f"{value}{unit}", 0.9
    return None, 0.0


def extract_screen_size(title: str) -> tuple[Optional[str], float]:
    m = SCREEN_SIZE_PATTERN.search(title)
    if m:
        return f'{m.group(1)}"', 0.9
    return None, 0.0


def extract_all_attributes(title: str) -> dict:
    """Extract all structured attributes from a product title."""
    brand, brand_conf = extract_brand(title)
    model_num, model_conf = extract_model_number(title)
    series, series_conf = extract_series(title)
    storage, storage_conf = extract_storage(title)
    screen, screen_conf = extract_screen_size(title)

    return {
        "brand": brand,
        "brand_confidence": brand_conf,
        "model_number": model_num,
        "model_number_confidence": model_conf,
        "series": series,
        "series_confidence": series_conf,
        "storage": storage,
        "storage_confidence": storage_conf,
        "screen_size": screen,
        "screen_size_confidence": screen_conf,
    }


if __name__ == "__main__":
    import json
    examples = [
        "טלפון סלולרי Apple iPhone 17 Pro Max 256GB",
        "Samsung Galaxy S26 Ultra SM-S948B/DS 512GB 12GB RAM",
        "אוזניות Apple AirPods Pro 2 MagSafe USB-C True Wireless",
        'טלוויזיה TCL 98C655 4K 98 אינטש',
        "מכונת קפה Nespresso Vertuo Pop ENV92.B DeLonghi",
        "מחשב נייד Lenovo ThinkPad X1 Carbon Gen 12 21KC00BRIV",
    ]
    for ex in examples:
        attrs = extract_all_attributes(ex)
        print(f"Title: {ex}")
        print(f"  {json.dumps({k:v for k,v in attrs.items() if v}, ensure_ascii=False, indent=4)}")
        print()
