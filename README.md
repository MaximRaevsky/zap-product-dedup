# Product Deduplication for E-Commerce Price Comparison

A hybrid deterministic + LLM pipeline that deduplicates product listings from [Zap.co.il](https://www.zap.co.il), groups identical products across inconsistent naming conventions, and surfaces the lowest price for each product.

Built as a take-home assignment for a **GenAI Exploration Lead** role, demonstrating evaluation-driven GenAI engineering with real Israeli e-commerce data.

---

## Problem Statement

Price comparison platforms aggregate product listings from many retailers. The same product often appears under different names:

| Listing A | Listing B | Same Product? |
|---|---|---|
| `Samsung Galaxy S26 Ultra SM-S948B/DS 256GB` | `סמסונג גלקסי S26 Ultra 256GB` | Yes |
| `Bosch Serie 6 SMS6ZCI42E` | `בוש סדרה 6 מדיח כלים SMS6ZCI42E` | Yes |
| `Samsung Galaxy S26 Ultra 256GB` | `Samsung Galaxy S26 Ultra 512GB` | No (different storage) |

Without deduplication, customers see fragmented results and miss the best price. This pipeline solves that by combining deterministic rules with LLM reasoning.

## Why This Matters

- **Cleaner product grouping**: Customers see one entry per product instead of scattered variants
- **Reliable cheapest-price display**: Grouping across retailers ensures the true minimum price is shown
- **Scalable ambiguity handling**: Hard cases go to an LLM judge, easy cases stay fast and cheap
- **Inspectable confidence**: Low-confidence decisions can be routed to human review

## Approach

### Zero Hardcoded Dictionaries

A deliberate design goal: the pipeline contains **no hardcoded brand dictionaries, noise phrase lists, or product-specific mappings**. Where earlier iterations relied on hand-maintained tables (brand Hebrew-English maps, seller noise phrases, navigation keyword filters), this version delegates all language-dependent reasoning to LLMs:

- **Scraper noise filtering**: An LLM (`gpt-4o-mini`) classifies scraped text as product titles vs site UI noise, replacing a hardcoded list of Hebrew navigation keywords
- **Synthetic variant generation**: An LLM generates realistic title variants (brand swaps, seller info, abbreviations) instead of hardcoded transform dictionaries
- **Brand reasoning**: The LLM judge reads brands directly from raw titles instead of relying on a pre-built brand dictionary

This makes the pipeline **category-agnostic** and generalizable to new product domains without code changes.

### Data Collection

The scraper **dynamically discovers all available categories** from the Zap homepage (~59 categories) and **randomly samples 10 per run** for diversity. This ensures the pipeline is tested against a wide variety of product types -- not just electronics.

Categories tested across runs include: smart watches, perfumes, fridges, trampolines, car speakers, dryers, dishwashers, ink cartridges, coffee machines, monitors, AC units, and more.

For each category, the scraper also visits **individual product comparison pages** to collect per-store title variants -- the real-world naming inconsistencies the assignment references (e.g., one store writes "Samsung 23S", another writes "סמסונג גלקסי 23"). An LLM filters these pages to retain only actual product titles.

The dataset is augmented with LLM-generated synthetic noisy duplicates and hand-crafted hard negatives for robust evaluation.

### Why Hybrid (Deterministic + LLM)

| Approach | Strengths | Weaknesses |
|---|---|---|
| Rules only | Fast, cheap, predictable | Fails on ambiguous/multilingual cases |
| LLM only | Handles nuance, multilingual | Slow, expensive, opaque |
| **Hybrid** | Best of both: fast for easy cases, smart for hard ones | Needs careful threshold tuning |

The pipeline uses deterministic rules for the vast majority of decisions (high-confidence matches from shared model IDs, exact matches, and clear non-matches), and sends only the ambiguous cases to GPT-4o. This keeps costs low while maintaining high accuracy.

### Pipeline Architecture

```
Listings → Normalize → Extract Attributes → Candidate Generation (blocking)
    → Rule-Based Matching → [ambiguous cases] → LLM Judge (GPT-4o)
    → Confidence Aggregation → Union-Find Clustering → Min Price Selection
    → Evaluation & Error Analysis
```

**Stage details:**

1. **Normalization** (`normalize.py`): Unicode NFC, RTL marker removal, Hebrew-to-English unit translations (e.g., `ג'יגה` → `gb`). Purely structural cleanup -- no hardcoded noise phrases.
2. **Attribute Extraction** (`extract_attributes.py`): Regex-based extraction of model/SKU numbers, storage capacity, and screen size. No brand dictionary -- brand reasoning is delegated to the LLM judge.
3. **Candidate Generation** (`candidate_generation.py`): Comparing every listing to every other listing would be O(n^2) -- ~500K pairs for 1,000 listings. Instead, three blocking strategies narrow this to ~15K pairs worth checking:
   - **Model ID blocking** (strongest): Listings sharing the same Zap product ID are almost certainly the same product
   - **TF-IDF character n-gram similarity**: Converts each title to a vector of character patterns; titles sharing rare character sequences score high. Uses character chunks (2-4 chars) instead of words, so cross-language matches work (Hebrew and English titles of the same phone share model numbers and specs in character patterns)
   - **Extracted model number**: Titles containing the same SKU code (e.g., both mention `SMS6ZCI42E`)
4. **Rule-Based Matching** (`match_rules.py`): 7 deterministic rules with confidence bands. Same model_id → 0.98. Exact normalized match (same model_id) → 0.99. Conflicting specs (different storage/screen) → 0.05. Cross-product exact matches → reject (likely scraper noise). Everything ambiguous → send to LLM.
5. **LLM Judge** (`llm_judge.py`): GPT-4o with structured JSON output for ambiguous pairs (confidence 0.3--0.85), with caching, retries, and logging. Up to 150 calls per run. Uses `gpt-4o` (not mini) because this task requires reasoning about Hebrew-English brand equivalences and subtle spec differences.
6. **Confidence** (`confidence.py`): Simple policy -- LLM verdict overrides rules when present; no hand-blended weights.
7. **Clustering** (`cluster_products.py`): Union-Find grouping of confirmed duplicates. If A=B and B=C, then {A, B, C} form one product group.
8. **Price Selection** (`select_best_price.py`): Minimum price per cluster -- the cheapest available price across all grouped store listings.

### LLM Usage

| Component | Model | Purpose | Cost |
|---|---|---|---|
| Scraper noise filter | `gpt-4o-mini` | Classify scraped text as product vs UI noise | ~$0.01/run |
| Variant generation | `gpt-4o-mini` | Generate realistic title variants for evaluation | ~$0.02/run |
| Dedup judge | `gpt-4o` | Resolve ambiguous pairs | ~$0.25/run |

Total LLM cost per run: **~$0.28**

### Language Handling

The system handles:
- Hebrew-only titles
- English-only titles
- Mixed Hebrew-English titles
- Brand transliteration (the LLM reasons about `אפל` ↔ `Apple`, `בוש` ↔ `Bosch`, etc.)
- Token reordering across languages

## Evaluation

### Methodology
Ground truth is built from:
1. **Real Zap variants**: Same `model_id`, different titles scraped from product comparison pages (per-store naming)
2. **LLM-generated synthetic duplicates**: Brand swaps, seller noise, abbreviation, reordering (generated by `gpt-4o-mini`)
3. **Hard negatives**: Same category, similar titles, but genuinely different products

### Pairwise Matching (2 runs, different random category samples)

| Run | Categories | F1 | Precision | Recall | Listings | Clusters |
|---|---|---|---|---|---|---|
| 1 | Watches, perfumes, fridges, trampolines, ink, dishwashers... | 0.9837 | 0.9998 | 0.9680 | 1,854 | 151 |
| 2 | Perfumes, coffee, monitors, AC, dryers... | 0.9895 | 0.9982 | 0.9810 | 1,162 | 63 |
| **Average** | | **0.9866** | **0.9990** | **0.9745** | | |

### Price Correctness

The pipeline evaluates whether the customer actually sees the cheapest available price after deduplication. This is measured in two dimensions:

- **Cluster purity** (95.2%): 60 out of 63 multi-listing clusters contain only a single real product. The remaining 3 clusters over-merge different products due to transitive chaining in Union-Find (a few false-positive pairs can chain-merge an entire category).
- **Cheapest price found** (100% on pure clusters): For all 60 correctly-grouped products, the cluster's shown price matches the true cheapest available listing. No customer misses a better price due to incomplete grouping.

## Confidence & Debugging

### Multi-Stage Confidence
- **Candidate confidence**: How likely a pair merits comparison (from blocking scores)
- **Rule confidence**: Deterministic match quality (0.0--1.0)
- **LLM confidence**: Model's self-assessed certainty (used directly as final confidence)
- **Decision source**: Tracked per pair (rules_high, llm, rules_low, rules_fallback)

### Debug Outputs (`results/debug/`, regenerated on each run)
- `normalization_examples.json`: Before/after title normalization
- `candidate_pairs_log.json`: Generated pairs with blocking reasons
- `rule_decisions_log.json`: Per-pair rule results
- `llm_prompts_sample.json`: Actual prompts sent to GPT-4o
- `llm_responses_sample.json`: Raw and parsed LLM responses
- `llm_disagreements.json`: Cases where LLM overrode deterministic rules
- `false_positives.json`, `false_negatives.json`: Evaluation errors

## Where LLMs Added Value vs. Where They Didn't

**LLMs were essential for:**
- Scraper noise filtering (replacing a hardcoded Hebrew keyword list)
- Generating realistic multilingual test data (brand swaps, seller noise)
- Mixed Hebrew-English disambiguation in ambiguous pairs
- Cross-store title variations with substantial rewording

**Deterministic rules were better for:**
- Same model_id matches (the majority of correct decisions)
- Exact normalized matches
- Clear attribute conflicts (different storage/screen size)
- Speed: rules process in milliseconds, LLM takes ~1s per pair

**The sweet spot**: Across all test runs, less than 5% of candidate pairs needed LLM judgment, keeping API costs under $0.30 per run while maintaining F1 > 0.98.

## Limitations

- **Transitive over-merging**: Union-Find clustering is transitive -- if A matches B and B matches C, all three merge. A few false-positive pairs (especially from the 385 `rules_fallback` decisions that bypass the LLM cap of 150) can chain-merge an entire category into one cluster. This affected 3 out of 63 clusters in the sample run. Production mitigation: raise the LLM cap, add cluster-size sanity checks, or use connected-component pruning.
- **Single-page scraping**: Only the first page of each category is scraped. Pagination not implemented.
- **Ground truth**: Partially synthetic. Real-world evaluation would need human-labeled pairs.
- **No image matching**: Titles only; product images could provide additional signal.
- **Model_id dependency**: The strongest signal comes from Zap's model_id. Cross-platform deduplication would need other strategies.
- **LLM filter latency**: The noise filter adds ~1s per product page during scraping.

## What I Would Do Next in Production

1. **Scale data collection**: Paginated scraping across all Zap categories, incremental updates
2. **Embedding-based candidate generation**: Replace TF-IDF with multilingual embeddings (e.g., `intfloat/multilingual-e5-large`) for better cross-lingual matching
3. **Active learning loop**: Route low-confidence pairs to human reviewers, retrain rules
4. **Hebrew morphological analysis**: Use a Hebrew NLP library (e.g., YAP) for proper tokenization
5. **Monitoring & drift detection**: Track precision/recall over time as new products appear
6. **Cross-platform deduplication**: Match products across Zap, Amazon, AliExpress without shared IDs
7. **Image-based matching**: Use product image similarity as an additional signal
8. **A/B testing**: Measure customer impact on price comparison quality

## Setup & Run

### Prerequisites
- Python 3.9+
- OpenAI API key (only needed for `--refresh` or `--multi`)

### Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

### Quick Start (for reviewers)

Cached data from a sample run is included in the repository. The default run reuses it, so **no internet or API key is needed** to inspect the pipeline:

```bash
python src/main.py                    # Runs on cached data -- no scraping, no LLM calls
```

### Full Pipeline (requires internet + API key)
```bash
python src/main.py --refresh          # Re-scrape Zap + regenerate synthetic data
python src/main.py --refresh --seed 42  # Reproducible category selection
python src/main.py --multi 3          # 3 runs with different random categories (always refreshes)
python src/main.py --n-categories 15 --refresh  # Scrape more categories
```

### Caching Behavior

The pipeline is **cache-first** by default:
- If `data/raw/zap_listings.csv` exists, scraping is skipped
- If `data/synthetic/all_listings.csv` exists, synthetic generation is skipped
- The LLM judge caches its verdicts in `data/processed/llm_cache.json` across runs
- Use `--refresh` to force re-scraping and regeneration
- `--multi` always refreshes each run (different random categories per seed)

### Output Files
- `results/grouped_products.csv` -- Final deduplicated product groups with best prices
- `results/metrics/evaluation_metrics.json` -- Precision, recall, F1, cluster purity, price accuracy
- `results/metrics/iteration_log.md` -- Cross-validated results across multiple runs
- `results/debug/` -- Debugging artifacts (regenerated on each run, not committed)
- `results/summary.md` -- Key findings and business insights
- `results/submission_email.md` -- Submission email draft

## Project Structure
```
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── main.py                  # Pipeline orchestrator (supports --multi for cross-validation)
│   ├── collect_zap_data.py      # Zap scraper (dynamic categories + LLM noise filter)
│   ├── collect_seed_data.py     # Seed dataset builder
│   ├── generate_variants.py     # LLM-generated synthetic variants (no hardcoded transforms)
│   ├── normalize.py             # Title normalization (structural only, no phrase lists)
│   ├── extract_attributes.py    # Attribute extraction (model number, storage, screen)
│   ├── candidate_generation.py  # Blocking strategies (model_id, TF-IDF, model number)
│   ├── match_rules.py           # 7 deterministic rules (no brand dictionary)
│   ├── llm_judge.py             # GPT-4o LLM judge with caching
│   ├── confidence.py            # Simple LLM-overrides-rules policy
│   ├── cluster_products.py      # Union-Find clustering
│   ├── select_best_price.py     # Price selection & export
│   ├── evaluate.py              # Evaluation metrics
│   └── analyze_errors.py        # Error analysis
├── data/
│   ├── raw/                     # Scraped Zap data
│   ├── processed/               # Seed dataset, LLM cache
│   └── synthetic/               # Augmented listings & ground truth
└── results/
    ├── grouped_products.csv     # Final output
    ├── summary.md               # Key findings
    ├── submission_email.md      # Email draft
    ├── metrics/                 # Evaluation metrics & iteration log
    └── debug/                   # Debug artifacts (gitignored, regenerated on run)
```
