"""
LLM judge for ambiguous product deduplication cases.
Uses OpenAI GPT-4o with structured JSON output, caching, retries, and full logging.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CACHE_PATH = Path("data/processed/llm_cache.json")
PROMPT_LOG_PATH = Path("results/debug/llm_prompts_sample.json")
RESPONSE_LOG_PATH = Path("results/debug/llm_responses_sample.json")

SYSTEM_PROMPT = """You are an expert product deduplication system for an Israeli e-commerce price comparison platform.

Your task: determine if two product listings refer to the SAME sellable product, even if their names differ in language, wording, or detail level.

Key rules:
- Two listings are duplicates if they refer to the exact same sellable product (same brand, model, configuration).
- Different storage sizes (e.g., 256GB vs 512GB) mean DIFFERENT products.
- Different screen sizes mean DIFFERENT products.
- Different colors of the same model ARE duplicates (color is a variant, not a different product for price comparison).
- Hebrew and English names for the same product ARE duplicates.
- Titles with extra seller info (e.g., "יבואן רשמי") but same product ARE duplicates.
- Different generations (e.g., AirPods Pro 2 vs AirPods Pro 3) are DIFFERENT products.

Respond ONLY with valid JSON in this exact format:
{"is_duplicate": true/false, "confidence": 0.0-1.0, "rationale": "brief explanation"}"""

USER_TEMPLATE = """Compare these two product listings:

Product A: {title_a}
Product B: {title_b}

Category: {category}

Extracted attributes:
A - Brand: {brand_a}, Series: {series_a}, Model: {model_a}, Storage: {storage_a}
B - Brand: {brand_b}, Series: {series_b}, Model: {model_b}, Storage: {storage_b}

Are these the same product? Respond with JSON only."""


def _cache_key(title_a: str, title_b: str) -> str:
    key = f"{min(title_a, title_b)}|||{max(title_a, title_b)}"
    return hashlib.sha256(key.encode()).hexdigest()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _append_to_log(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logs = []
    logs.append(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


def _log_prompt(prompt_data: dict):
    _append_to_log(PROMPT_LOG_PATH, prompt_data)


def _log_response(response_data: dict):
    _append_to_log(RESPONSE_LOG_PATH, response_data)


def _parse_llm_response(text: str) -> Optional[dict]:
    """Safely parse LLM JSON response with fallbacks."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        result = json.loads(text)
        if "is_duplicate" in result and "confidence" in result:
            return {
                "is_duplicate": bool(result["is_duplicate"]),
                "confidence": float(result["confidence"]),
                "rationale": str(result.get("rationale", "")),
            }
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Fallback: try to find JSON object in text
    import re
    m = re.search(r'\{[^}]+\}', text)
    if m:
        try:
            result = json.loads(m.group(0))
            return {
                "is_duplicate": bool(result.get("is_duplicate", False)),
                "confidence": float(result.get("confidence", 0.5)),
                "rationale": str(result.get("rationale", "parsed from partial")),
            }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    return None


def judge_pair(pair: dict, cache: Optional[dict] = None) -> dict:
    """
    Use LLM to judge if a candidate pair is a duplicate.
    Returns the pair dict enriched with llm_* fields.
    """
    title_a = pair.get("title_a", "")
    title_b = pair.get("title_b", "")

    ck = _cache_key(title_a, title_b)

    if cache and ck in cache:
        cached = cache[ck]
        return {
            **pair,
            "llm_is_duplicate": cached["is_duplicate"],
            "llm_confidence": cached["confidence"],
            "llm_rationale": cached["rationale"],
            "llm_cached": True,
        }

    user_msg = USER_TEMPLATE.format(
        title_a=title_a,
        title_b=title_b,
        category=pair.get("category", pair.get("series_a", "")),
        brand_a=pair.get("brand_a", "unknown"),
        brand_b=pair.get("brand_b", "unknown"),
        series_a=pair.get("series_a", "unknown"),
        series_b=pair.get("series_b", "unknown"),
        model_a=pair.get("model_a", "unknown"),
        model_b=pair.get("model_b", "unknown"),
        storage_a=pair.get("storage_a", "unknown"),
        storage_b=pair.get("storage_b", "unknown"),
    )

    _log_prompt({
        "title_a": title_a,
        "title_b": title_b,
        "user_message": user_msg,
    })

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=200,
            )
            raw_text = response.choices[0].message.content
            parsed = _parse_llm_response(raw_text)

            _log_response({
                "title_a": title_a,
                "title_b": title_b,
                "raw_response": raw_text,
                "parsed": parsed,
                "attempt": attempt + 1,
            })

            if parsed:
                if cache is not None:
                    cache[ck] = parsed
                return {
                    **pair,
                    "llm_is_duplicate": parsed["is_duplicate"],
                    "llm_confidence": parsed["confidence"],
                    "llm_rationale": parsed["rationale"],
                    "llm_cached": False,
                }

        except Exception as e:
            print(f"  LLM attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    return {
        **pair,
        "llm_is_duplicate": None,
        "llm_confidence": 0.0,
        "llm_rationale": "LLM failed after 3 attempts",
        "llm_cached": False,
    }


def judge_ambiguous_pairs(
    pairs: list[dict],
    low_threshold: float = 0.4,
    high_threshold: float = 0.75,
    max_llm_calls: int = 80,
) -> list[dict]:
    """
    Send ambiguous pairs (rule_confidence between thresholds) to LLM for judgment.
    Non-ambiguous pairs pass through unchanged. Caps LLM calls for cost control.
    """
    cache = _load_cache()
    results = []
    llm_count = 0

    ambiguous = [p for p in pairs if low_threshold <= p["rule_confidence"] < high_threshold]
    non_ambiguous = [p for p in pairs if p["rule_confidence"] < low_threshold or p["rule_confidence"] >= high_threshold]

    # Prioritize most ambiguous pairs (closest to 0.5)
    ambiguous.sort(key=lambda p: abs(p["rule_confidence"] - 0.55))
    if len(ambiguous) > max_llm_calls:
        overflow = ambiguous[max_llm_calls:]
        ambiguous = ambiguous[:max_llm_calls]
        non_ambiguous.extend(overflow)

    print(f"LLM judge: {len(ambiguous)} ambiguous pairs to evaluate (capped at {max_llm_calls})")

    for p in ambiguous:
        result = judge_pair(p, cache)
        results.append(result)
        llm_count += 1
        if llm_count % 10 == 0:
            print(f"  Processed {llm_count}/{len(ambiguous)} pairs")
            _save_cache(cache)

    _save_cache(cache)
    print(f"  LLM judged {llm_count} pairs ({sum(1 for r in results if r.get('llm_cached'))} from cache)")

    for p in non_ambiguous:
        p["llm_is_duplicate"] = None
        p["llm_confidence"] = 0.0
        p["llm_rationale"] = "not_sent_to_llm"
        p["llm_cached"] = False
        results.append(p)

    return results
