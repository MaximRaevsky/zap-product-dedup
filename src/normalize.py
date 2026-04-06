"""
Normalize product titles for deduplication.
Handles Hebrew RTL markers, units, seller noise. Category-agnostic.
"""

import re
import unicodedata

# Only truly generic seller/logistics noise -- NOT category names
NOISE_PHRASES = [
    "יבואן רשמי",
    "אחריות יבואן רשמי",
    "יבואן מורשה",
    "משלוח חינם",
    "כולל משלוח",
]

UNIT_MAP = {
    "אינטש": "inch", "אינץ'": "inch", "אינץ": "inch",
    "ג'יגה": "gb", "טרה": "tb",
}

COLOR_RE = re.compile(
    r"\bבצבע\s+\S+\b"
    r"|\b(?:black|white|silver|gold|blue|red|green|gray|grey|pink|titanium)\b",
    re.IGNORECASE,
)


def normalize_title(title: str) -> str:
    if not title:
        return ""
    text = unicodedata.normalize("NFC", title)
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u200b-\u200d\ufeff]", "", text)

    for phrase in sorted(NOISE_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, " ")
    text = COLOR_RE.sub(" ", text)

    for heb, eng in UNIT_MAP.items():
        text = text.replace(heb, eng)

    text = text.lower()
    text = re.sub(r"[/\\,;:!?(){}[\]]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"-{2,}", "-", text)
    return text


def normalize_for_comparison(title: str) -> str:
    text = normalize_title(title)
    text = re.sub(r"[-._]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
