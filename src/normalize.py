"""
Normalize product titles for deduplication.
Only structural cleanup -- no hardcoded phrase lists or color dictionaries.
"""

import re
import unicodedata

# Hebrew-to-English measurement unit translations (universal technical units)
UNIT_MAP = {
    "אינטש": "inch", "אינץ'": "inch", "אינץ": "inch",
    "ג'יגה": "gb", "טרה": "tb",
}


def normalize_title(title: str) -> str:
    if not title:
        return ""
    text = unicodedata.normalize("NFC", title)
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\u200b-\u200d\ufeff]", "", text)

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
