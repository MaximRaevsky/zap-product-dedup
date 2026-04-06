"""
LLM judge for ambiguous product deduplication cases.
Uses GPT-4o with structured JSON output, caching, retries, and logging.
No pre-extracted brand info -- the LLM reads brands directly from the raw titles.
"""

import os
import json
import hashlib
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CACHE_PATH = Path("data/processed/llm_cache.json")
LOG_DIR = Path("results/debug")

SYSTEM_PROMPT = """You are a product deduplication system for a price comparison site.

Determine if two product listings refer to the SAME sellable product, even if names differ in language or wording.

Rules:
- Same brand + model + configuration = duplicate
- Different storage (256GB vs 512GB) = DIFFERENT products
- Different screen sizes = DIFFERENT products
- Different colors of same model = duplicate (color is not a separate product)
- Hebrew and English names for same product = duplicate (e.g. "בוש" = "Bosch")
- Extra seller info = still duplicate
- Different generations (v2 vs v3) = DIFFERENT products

Respond ONLY with JSON: {"is_duplicate": true/false, "confidence": 0.0-1.0, "rationale": "brief"}"""

USER_TEMPLATE = """Product A: {title_a}
Product B: {title_b}

Category: {category}
Extracted specs A: model={model_a}, storage={storage_a}
Extracted specs B: model={model_b}, storage={storage_b}

Same product? JSON only."""


def _cache_key(a: str, b: str) -> str:
    key = f"{min(a, b)}|||{max(a, b)}"
    return hashlib.sha256(key.encode()).hexdigest()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.load(open(CACHE_PATH, encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(cache, open(CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _log(filename: str, data: dict):
    path = LOG_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    if path.exists():
        try:
            logs = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            logs = []
    logs.append(data)
    json.dump(logs, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _parse_response(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        r = json.loads(text)
        if "is_duplicate" in r:
            return {"is_duplicate": bool(r["is_duplicate"]),
                    "confidence": float(r.get("confidence", 0.5)),
                    "rationale": str(r.get("rationale", ""))}
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\{[^}]+\}", text)
    if m:
        try:
            r = json.loads(m.group(0))
            return {"is_duplicate": bool(r.get("is_duplicate", False)),
                    "confidence": float(r.get("confidence", 0.5)),
                    "rationale": str(r.get("rationale", ""))}
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def judge_pair(pair: dict, cache: Optional[dict] = None) -> dict:
    title_a, title_b = pair.get("title_a", ""), pair.get("title_b", "")
    ck = _cache_key(title_a, title_b)

    if cache and ck in cache:
        c = cache[ck]
        return {**pair, "llm_is_duplicate": c["is_duplicate"],
                "llm_confidence": c["confidence"], "llm_rationale": c["rationale"],
                "llm_cached": True}

    user_msg = USER_TEMPLATE.format(
        title_a=title_a, title_b=title_b,
        category=pair.get("category", ""),
        model_a=pair.get("model_a", "?"), model_b=pair.get("model_b", "?"),
        storage_a=pair.get("storage_a", "?"), storage_b=pair.get("storage_b", "?"),
    )

    _log("llm_prompts_sample.json", {"title_a": title_a, "title_b": title_b, "prompt": user_msg})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_msg}],
                temperature=0.1, max_tokens=200,
            )
            raw = resp.choices[0].message.content
            parsed = _parse_response(raw)
            _log("llm_responses_sample.json", {"title_a": title_a, "title_b": title_b,
                                                "raw": raw, "parsed": parsed})
            if parsed:
                if cache is not None:
                    cache[ck] = parsed
                return {**pair, "llm_is_duplicate": parsed["is_duplicate"],
                        "llm_confidence": parsed["confidence"],
                        "llm_rationale": parsed["rationale"], "llm_cached": False}
        except Exception as e:
            print(f"  LLM attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    return {**pair, "llm_is_duplicate": None, "llm_confidence": 0.0,
            "llm_rationale": "LLM failed", "llm_cached": False}


def judge_ambiguous_pairs(
    pairs: list[dict],
    low_threshold: float = 0.3,
    high_threshold: float = 0.85,
    max_llm_calls: int = 150,
) -> list[dict]:
    cache = _load_cache()

    ambiguous = [p for p in pairs if low_threshold <= p["rule_confidence"] < high_threshold]
    clear = [p for p in pairs if p["rule_confidence"] < low_threshold or p["rule_confidence"] >= high_threshold]

    ambiguous.sort(key=lambda p: abs(p["rule_confidence"] - 0.5))
    if len(ambiguous) > max_llm_calls:
        clear.extend(ambiguous[max_llm_calls:])
        ambiguous = ambiguous[:max_llm_calls]

    print(f"LLM judge: {len(ambiguous)} ambiguous pairs (cap {max_llm_calls})")

    results = []
    for i, p in enumerate(ambiguous):
        results.append(judge_pair(p, cache))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(ambiguous)}")
            _save_cache(cache)

    _save_cache(cache)
    cached = sum(1 for r in results if r.get("llm_cached"))
    print(f"  Done ({cached} from cache)")

    for p in clear:
        p.update({"llm_is_duplicate": None, "llm_confidence": 0.0,
                   "llm_rationale": "not_sent", "llm_cached": False})
        results.append(p)

    return results
