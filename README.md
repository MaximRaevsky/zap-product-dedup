# Product Deduplication for E-Commerce Price Comparison

A hybrid deterministic + LLM pipeline that deduplicates product listings from [Zap.co.il](https://www.zap.co.il), groups identical products across inconsistent naming conventions, and surfaces the lowest price for each product.

Built as a take-home assignment for a **GenAI Exploration Lead** role, demonstrating evaluation-driven GenAI engineering with real Israeli e-commerce data.

---

## Problem Statement

Price comparison platforms aggregate product listings from many retailers. The same product often appears under different names:

| Listing A | Listing B | Same Product? |
|---|---|---|
| `טלפון סלולרי Apple iPhone 17 Pro Max 256GB` | `אפל iPhone 17 Pro Max 256GB יבואן רשמי` | Yes |
| `Samsung Galaxy S26 Ultra SM-S948B/DS 256GB 12GB RAM` | `Galaxy S26 Ultra Samsung 256GB` | Yes |
| `Samsung Galaxy S26 Ultra 256GB` | `Samsung Galaxy S26 Ultra 512GB` | No (different storage) |

Without deduplication, customers see fragmented results and miss the best price. This pipeline solves that by combining deterministic rules with LLM reasoning.

## Why This Matters

- **Cleaner product grouping**: Customers see one entry per product instead of scattered variants
- **Reliable cheapest-price display**: Grouping across retailers ensures the true minimum price is shown
- **Scalable ambiguity handling**: Hard cases go to an LLM judge, easy cases stay fast and cheap
- **Inspectable confidence**: Low-confidence decisions can be routed to human review

## Approach

### Data Collection
Real product data scraped from Zap.co.il public pages across 5 diverse categories:
- Smartphones, Headphones, TVs, Laptops, Coffee Machines

Each category page yields both comparison listings and direct-buy (zapstore) listings, providing natural title variations for the same product. The dataset is augmented with synthetic noisy duplicates and hard negatives for robust evaluation.

### Why Hybrid (Deterministic + LLM)

| Approach | Strengths | Weaknesses |
|---|---|---|
| Rules only | Fast, cheap, predictable | Fails on ambiguous/multilingual cases |
| LLM only | Handles nuance, multilingual | Slow, expensive, opaque |
| **Hybrid** | Best of both: fast for easy cases, smart for hard ones | Needs careful threshold tuning |

The pipeline uses deterministic rules for ~70% of decisions (high-confidence matches and clear non-matches), and sends only the ambiguous ~10% to GPT-4o for judgment. This keeps costs low while maintaining high accuracy.

### Pipeline Architecture

```
Listings → Normalize → Extract Attributes → Candidate Generation (blocking)
    → Rule-Based Matching → [ambiguous cases] → LLM Judge (GPT-4o)
    → Confidence Aggregation → Union-Find Clustering → Min Price Selection
    → Evaluation & Error Analysis
```

**Stage details:**

1. **Normalization** (`normalize.py`): Unicode NFC, Hebrew RTL marker removal, unit standardization (אינטש→inch, GB/TB), filler word removal, seller noise stripping
2. **Attribute Extraction** (`extract_attributes.py`): Brand, model number, series, storage, screen size via regex + dictionary lookup with Hebrew mappings
3. **Candidate Generation** (`candidate_generation.py`): Three blocking strategies—brand matching, TF-IDF character n-gram similarity, shared model numbers
4. **Rule-Based Matching** (`match_rules.py`): Cascading rules with confidence bands (0.0–1.0), including SKU-difference detection for near-identical titles
5. **LLM Judge** (`llm_judge.py`): GPT-4o with structured JSON output for ambiguous pairs (confidence 0.4–0.75), with caching, retries, and full prompt/response logging
6. **Clustering** (`cluster_products.py`): Union-Find grouping of confirmed duplicates
7. **Price Selection** (`select_best_price.py`): Minimum price per cluster

### Language Handling

The system explicitly handles:
- Hebrew-only titles (`‏מכונת אספרסו נספרסו`)
- English-only titles (`Apple iPhone 17 Pro Max 256GB`)
- Mixed Hebrew-English (`טלפון סלולרי Samsung Galaxy S26 Ultra`)
- Brand transliteration (`אפל` ↔ `Apple`, `סמסונג` ↔ `Samsung`)
- Token reordering across languages

## Evaluation

### Methodology
Ground truth is built from:
1. **Real Zap variants**: Same `model_id`, different titles from comparison vs. zapstore listings
2. **Synthetic duplicates**: Token reordering, language swaps, abbreviations, seller noise
3. **Hard negatives**: Same brand/series but different model, storage, or generation

### Results (Final Iteration)

| Metric | Value |
|---|---|
| **Pairwise Precision** | 0.937 |
| **Pairwise Recall** | 0.915 |
| **Pairwise F1** | 0.926 |
| **Price Correctness** | 100% (71/71) |
| **Clusters** | 102 (92 multi-listing) |

### Iteration History

| | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| Baseline | 0.876 | 0.912 | 0.893 | 44 | 30 |
| +Storage fix, +Part# detection | 0.884 | 0.918 | 0.900 | 41 | 28 |
| +Broader SKU detection, +Conservative fallback | **0.937** | **0.915** | **0.926** | **21** | 29 |

Key improvements came from:
- Fixing storage parsing for combined formats (`12GB+256GB`)
- Detecting differing SKU/part numbers in near-identical titles
- Conservative fallback for ambiguous cases without LLM judgment

## Confidence & Debugging

### Multi-Stage Confidence
- **Candidate confidence**: How likely a pair merits comparison (from blocking scores)
- **Rule confidence**: Deterministic match quality (0.0–1.0)
- **LLM confidence**: Model's self-assessed certainty
- **Final confidence**: Weighted aggregate, with LLM overriding rules when invoked

### Debug Outputs (`results/debug/`)
- `normalization_examples.json`: Before/after title normalization
- `attribute_extraction_examples.json`: Extracted attributes per title
- `candidate_pairs_log.json`: Generated pairs with reasons
- `rule_decisions_log.json`: Per-pair rule results
- `llm_prompts_sample.json`: Actual prompts sent to GPT-4o
- `llm_responses_sample.json`: Raw and parsed LLM responses
- `llm_disagreements.json`: Cases where LLM overrode deterministic rules
- `false_positives.json`, `false_negatives.json`: Evaluation errors with explanations

## Where LLMs Added Value vs. Where They Didn't

**LLMs were essential for:**
- Mixed Hebrew-English disambiguation
- Cases where brand was expressed differently across languages
- Subtle model-number interpretation (e.g., same series with different color suffixes)

**Deterministic rules were better for:**
- Exact normalized matches (~35% of all decisions, zero LLM cost)
- Clear attribute conflicts (different storage/screen size)
- High fuzzy similarity within same brand
- Speed: rules process in milliseconds, LLM takes ~1s per pair

**The sweet spot**: Only 80 out of 3,000 candidate pairs needed LLM judgment (2.7%), keeping API costs under $0.20 for the full dataset while maintaining F1 > 0.92.

## Limitations

- **Scraping scope**: 5 categories, ~180 raw listings. Production would need broader coverage.
- **Hebrew NLP**: Relies on dictionary-based brand mapping rather than full morphological analysis.
- **Ground truth**: Partially synthetic. Real-world evaluation would need human-labeled pairs.
- **Single-page scraping**: Only first page of each category. Pagination not implemented.
- **No image matching**: Titles only; product images could provide additional signal.

## What I Would Do Next in Production

1. **Scale data collection**: Paginated scraping across all Zap categories, incremental updates
2. **Embedding-based candidate generation**: Replace TF-IDF with multilingual embeddings (e.g., `intfloat/multilingual-e5-large`) for better Hebrew-English cross-lingual matching
3. **Active learning loop**: Route low-confidence pairs to human reviewers, retrain rules
4. **Hebrew morphological analysis**: Use a Hebrew NLP library (e.g., YAP) for proper tokenization
5. **Monitoring & drift detection**: Track precision/recall over time as new products appear
6. **Batch LLM optimization**: Use async API calls and batching for lower latency
7. **Image-based matching**: Use product image similarity as an additional deduplication signal
8. **A/B testing**: Measure customer impact of deduplication on price comparison quality

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
python src/main.py
```

### Options
```bash
python src/main.py --skip-scrape      # Use existing scraped data
python src/main.py --skip-synthetic   # Use existing synthetic data
python src/main.py --threshold 0.6    # Adjust clustering confidence threshold
```

### Output Files
- `results/grouped_products.csv` — Final deduplicated product groups with best prices
- `results/metrics/evaluation_metrics.json` — Precision, recall, F1, price accuracy
- `results/metrics/iteration_log.md` — Improvement history across iterations
- `results/debug/` — Full debugging artifacts
- `results/summary.md` — Key findings and business insights
- `results/submission_email.md` — Submission email draft

## Example Output

**Cluster: Samsung Galaxy S26 Ultra 256GB**
| Listing | Price |
|---|---|
| `טלפון סלולרי Samsung Galaxy S26 Ultra SM-S948B/DS 256GB 12GB RAM` | 3,888 ₪ |
| `Samsung Galaxy S26 Ultra 12GB+256GB (SM-S948B/DS) בצבע כחול` | 4,289 ₪ |
| `Galaxy S26 Ultra Samsung 256GB` | 4,500 ₪ |
→ **Best price: 3,888 ₪**

**Cluster: Nespresso Essenza Mini D30 EN85**
| Listing | Price |
|---|---|
| `‏מכונת אספרסו Nespresso Delonghi Essenza Mini D30 EN85` | 349 ₪ |
| `Essenza Mini D30 EN85 ‏מכונת אספרסו Nespresso Delonghi` | 402 ₪ |
| `נספרסו Essenza Mini D30 EN85 יבואן רשמי` | 380 ₪ |
→ **Best price: 349 ₪**

## Project Structure
```
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── main.py                  # Pipeline orchestrator
│   ├── collect_zap_data.py      # Zap web scraper
│   ├── collect_seed_data.py     # Seed dataset builder
│   ├── generate_variants.py     # Synthetic data generator
│   ├── normalize.py             # Title normalization
│   ├── extract_attributes.py    # Attribute extraction
│   ├── candidate_generation.py  # Blocking & candidate pairs
│   ├── match_rules.py           # Deterministic matching rules
│   ├── llm_judge.py             # GPT-4o LLM judge
│   ├── confidence.py            # Confidence aggregation
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
    ├── metrics/                 # Evaluation metrics
    ├── debug/                   # Debug artifacts
    └── examples/                # Sample outputs
```
