"""
Extract structured attributes (model number, storage, screen size) from product titles.
No hardcoded brand dictionaries -- brand reasoning is delegated to the LLM judge.
"""

import re
from typing import Optional

# Generic model/SKU number patterns -- structural alphanumeric codes, not tied to any brand
MODEL_PATTERNS = [
    re.compile(r"\b[A-Z]{2,5}-[A-Z0-9]{2,}(?:/\w+)?\b"),          # SM-S948B, WH-1000XM5
    re.compile(r"\b[A-Z]{2,4}\d{3,}[A-Z]{0,4}(?:/[A-Z0-9]+)?\b"), # MXP63ZM, EN85L
]

STORAGE_PATTERN = re.compile(r"\b(\d+)\s*(GB|TB)\b", re.IGNORECASE)
RAM_PATTERN = re.compile(r"\b(\d+)\s*GB\s*RAM\b", re.IGNORECASE)
COMBINED_STORAGE = re.compile(r"\b(\d+)GB\s*\+\s*(\d+)(GB|TB)\b", re.IGNORECASE)
SCREEN_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:inch|אינטש|אינץ['\u2019]?|\")\b", re.IGNORECASE
)


def extract_model_number(title: str) -> tuple[Optional[str], float]:
    for pattern in MODEL_PATTERNS:
        m = pattern.search(title)
        if m:
            return m.group(0).upper(), 0.9
    return None, 0.0


def extract_storage(title: str) -> tuple[Optional[str], float]:
    cm = COMBINED_STORAGE.search(title)
    if cm:
        return f"{cm.group(2)}{cm.group(3).upper()}", 0.9

    ram_match = RAM_PATTERN.search(title)
    ram_val = ram_match.group(1) if ram_match else None
    for m in STORAGE_PATTERN.finditer(title):
        val, unit = m.group(1), m.group(2).upper()
        if val == ram_val and unit == "GB":
            continue
        return f"{val}{unit}", 0.9
    return None, 0.0


def extract_screen_size(title: str) -> tuple[Optional[str], float]:
    m = SCREEN_PATTERN.search(title)
    return (f'{m.group(1)}"', 0.9) if m else (None, 0.0)


def extract_all_attributes(title: str) -> dict:
    model, mc = extract_model_number(title)
    storage, sc = extract_storage(title)
    screen, xc = extract_screen_size(title)
    return {
        "model_number": model, "model_number_confidence": mc,
        "storage": storage, "storage_confidence": sc,
        "screen_size": screen, "screen_size_confidence": xc,
    }
