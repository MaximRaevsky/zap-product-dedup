"""
Normalize product titles for deduplication.
Handles Hebrew, English, mixed-language, RTL markers, units, filler words.
"""

import re
import unicodedata


FILLER_PHRASES = [
    "טלפון סלולרי",
    "טלפון חכם",
    "מכשיר סלולרי",
    "סלולרי",
    "אוזניות אלחוטיות",
    "אוזניות בלוטות'",
    "אוזניות",
    "טלוויזיה",
    "טלויזיה",
    "מחשב נייד",
    "לפטופ",
    "מכונת קפה אוטומטית",
    "מכונת קפה",
    "מכונת אספרסו",
    "יבואן רשמי",
    "אחריות יבואן רשמי",
    "יבואן מורשה",
    "אחריות מעבדות",
    "משלוח חינם",
    "כולל משלוח",
]

UNIT_NORMALIZATIONS = {
    "אינטש": "inch",
    'אינץ\'': "inch",
    'אינץ': "inch",
    '"': "inch",
    "ג'יגה": "gb",
    "טרה": "tb",
}

COLOR_PATTERN = re.compile(
    r"\bבצבע\s+\S+\b"
    r"|\b(?:black|white|silver|gold|blue|red|green|gray|grey|pink|titanium)\b",
    re.IGNORECASE,
)

SELLER_PATTERN = re.compile(
    r"\b(?:DCS|iDigital|Bug|KSP|Ivory)\b",
    re.IGNORECASE,
)


def normalize_title(title: str) -> str:
    """Full normalization pipeline for a product title."""
    if not title:
        return ""

    text = unicodedata.normalize("NFC", title)

    # Strip RTL/LTR markers and zero-width chars
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u200b-\u200d\ufeff]", "", text)

    # Remove filler phrases (longest first to avoid partial matches)
    for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, " ")

    # Remove color descriptors
    text = COLOR_PATTERN.sub(" ", text)

    # Remove seller names
    text = SELLER_PATTERN.sub(" ", text)

    # Normalize units
    for heb_unit, eng_unit in UNIT_NORMALIZATIONS.items():
        text = text.replace(heb_unit, eng_unit)

    # Lowercase English portions only (preserve Hebrew casing is N/A)
    result = []
    for char in text:
        if "A" <= char <= "Z":
            result.append(char.lower())
        else:
            result.append(char)
    text = "".join(result)

    # Normalize whitespace and punctuation
    text = re.sub(r"[/\\,;:!?(){}[\]]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Keep hyphens in model numbers but normalize double hyphens
    text = re.sub(r"-{2,}", "-", text)

    return text


def normalize_for_comparison(title: str) -> str:
    """Aggressive normalization for fuzzy comparison: also strips hyphens, dots."""
    text = normalize_title(title)
    text = re.sub(r"[-._]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    examples = [
        "טלפון סלולרי Apple iPhone 17 Pro Max 256GB",
        "אפל אייפון 17 פרו מקס 256GB יבואן רשמי",
        "אוזניות ‏אלחוטיות Apple AirPods Pro 3 MagSafe USB-C MFHP4ZM/A",
        "Samsung Galaxy S26 Ultra SM-S948B/DS 512GB 12GB RAM בצבע שחור",
        'טלוויזיה TCL 98C655 4K ‏98 ‏אינטש',
        "מכונת קפה Nespresso Vertuo Pop ENV92.B DeLonghi נספרסו",
    ]
    for ex in examples:
        print(f"  IN:  {ex}")
        print(f"  OUT: {normalize_title(ex)}")
        print()
