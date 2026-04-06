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

### Data Collection

The scraper **dynamically discovers all available categories** from the Zap homepage (~59 categories) and **randomly samples 10 per run** for diversity. This ensures the pipeline is tested against a wide variety of product types -- not just electronics.

Categories tested across multiple runs include: smart watches, perfumes, fridges, trampolines, car speakers, dryers, dishwashers, ink cartridges, coffee machines, monitors, AC units, faucets, dolls, drills, cooktops, sunglasses, and more.

For each category, the scraper also visits **individual product comparison pages** to collect per-store title variants -- the real-world naming inconsistencies the assignment references (e.g., one store writes "Samsung 23S", another writes "סמסונג גלקסי 23").

The dataset is augmented with synthetic noisy duplicates and hard negatives for robust evaluation.

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

1. **Normalization** (`normalize.py`): Unicode NFC, RTL marker removal, unit standardization, generic noise removal. Deliberately minimal -- 5 noise phrases, no category-specific stripping.
2. **Attribute Extraction** (`extract_attributes.py`): Brand (via Hebrew-English dictionary), generic model/SKU numbers, storage, screen size. No product-line-specific patterns.
3. **Candidate Generation** (`candidate_generation.py`): Four blocking strategies -- Zap model_id matching (strongest), brand blocking, TF-IDF character n-gram similarity, shared extracted model numbers. Model_id pairs are uncapped; discovery pairs capped at 10K.
4. **Rule-Based Matching** (`match_rules.py`): 7 clean rules with confidence bands. Same model_id → 0.98. Exact match → 0.99. Conflicting specs → 0.05. Everything ambiguous → send to LLM.
5. **LLM Judge** (`llm_judge.py`): GPT-4o with structured JSON output for ambiguous pairs (confidence 0.3–0.85), with caching, retries, and logging. Up to 150 calls per run.
6. **Confidence** (`confidence.py`): Simple policy -- LLM verdict overrides rules when present; no hand-blended weights.
7. **Clustering** (`cluster_products.py`): Union-Find grouping of confirmed duplicates.
8. **Price Selection** (`select_best_price.py`): Minimum price per cluster.

### Language Handling

The system handles:
- Hebrew-only titles
- English-only titles
- Mixed Hebrew-English titles
- Brand transliteration (`אפל` ↔ `Apple`, `סמסונג` ↔ `Samsung`, `בוש` ↔ `Bosch`)
- Token reordering across languages

## Evaluation

### Methodology
Ground truth is built from:
1. **Real Zap variants**: Same `model_id`, different titles scraped from product comparison pages (per-store naming)
2. **Synthetic duplicates**: Brand swaps, token reordering, SKU dropping, seller noise, abbreviation
3. **Hard negatives**: Same category, similar titles, but genuinely different products

### Cross-Validated Results (3 runs, different random category samples)

| Run | Categories | F1 | Precision | Recall | Listings |
|---|---|---|---|---|---|
| 1 | Watches, perfumes, fridges, trampolines... | 0.9958 | 1.0000 | 0.9917 | 2,077 |
| 2 | Coffee, monitors, AC, dryers... | 0.9970 | 0.9995 | 0.9944 | 1,365 |
| 3 | Faucets, dolls, drills, cooktops... | 0.9961 | 1.0000 | 0.9923 | 2,002 |
| **Average** | | **0.9963** | **0.9998** | **0.9928** | |

Price correctness: **100%** across all runs.

## Confidence & Debugging

### Multi-Stage Confidence
- **Candidate confidence**: How likely a pair merits comparison (from blocking scores)
- **Rule confidence**: Deterministic match quality (0.0–1.0)
- **LLM confidence**: Model's self-assessed certainty (used directly as final confidence)
- **Decision source**: Tracked per pair (rules_high, llm, rules_low, rules_fallback)

### Debug Outputs (`results/debug/`)
- `normalization_examples.json`: Before/after title normalization
- `candidate_pairs_log.json`: Generated pairs with blocking reasons
- `rule_decisions_log.json`: Per-pair rule results
- `llm_prompts_sample.json`: Actual prompts sent to GPT-4o
- `llm_responses_sample.json`: Raw and parsed LLM responses
- `llm_disagreements.json`: Cases where LLM overrode deterministic rules
- `false_positives.json`, `false_negatives.json`: Evaluation errors

## Where LLMs Added Value vs. Where They Didn't

**LLMs were essential for:**
- Mixed Hebrew-English disambiguation
- Cases where brand was expressed differently across languages
- Cross-store title variations with substantial rewording

**Deterministic rules were better for:**
- Same model_id matches (the majority of correct decisions)
- Exact normalized matches
- Clear attribute conflicts (different storage/screen size)
- Speed: rules process in milliseconds, LLM takes ~1s per pair

**The sweet spot**: Across all test runs, less than 5% of candidate pairs needed LLM judgment, keeping API costs under $0.30 per run while maintaining F1 > 0.99.

## Limitations

- **Single-page scraping**: Only the first page of each category is scraped. Pagination not implemented.
- **Hebrew NLP**: Relies on dictionary-based brand mapping rather than full morphological analysis.
- **Ground truth**: Partially synthetic. Real-world evaluation would need human-labeled pairs.
- **No image matching**: Titles only; product images could provide additional signal.
- **Model_id dependency**: The strongest signal comes from Zap's model_id. Cross-platform deduplication would need other strategies.

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
- OpenAI API key

### Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

### Run Full Pipeline
```bash
python src/main.py                    # Single run, scrapes 10 random categories
python src/main.py --seed 42          # Reproducible category selection
python src/main.py --multi 3          # 3 runs with different random categories
python src/main.py --skip-scrape      # Re-run with existing data
python src/main.py --n-categories 15  # Scrape more categories
```

### Output Files
- `results/grouped_products.csv` -- Final deduplicated product groups with best prices
- `results/metrics/evaluation_metrics.json` -- Precision, recall, F1, price accuracy
- `results/metrics/iteration_log.md` -- Cross-validated results across multiple runs
- `results/debug/` -- Full debugging artifacts
- `results/summary.md` -- Key findings and business insights
- `results/submission_email.md` -- Submission email draft

## Project Structure
```
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── main.py                  # Pipeline orchestrator (supports --multi for cross-validation)
│   ├── collect_zap_data.py      # Zap scraper (dynamic category discovery + product pages)
│   ├── collect_seed_data.py     # Seed dataset builder
│   ├── generate_variants.py     # Synthetic data generator (category-agnostic)
│   ├── normalize.py             # Title normalization (minimal, generic)
│   ├── extract_attributes.py    # Attribute extraction (brand, model, storage, screen)
│   ├── candidate_generation.py  # Blocking strategies (model_id, brand, TF-IDF, model#)
│   ├── match_rules.py           # 7 deterministic rules
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
    ├── debug/                   # Debug artifacts
    └── examples/                # Sample outputs
```
