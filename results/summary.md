# Results Summary: Product Deduplication Pipeline

## Key Metrics

| Metric | Value |
|---|---|
| Pairwise Precision | **0.937** |
| Pairwise Recall | **0.915** |
| Pairwise F1 | **0.926** |
| Price Correctness | **100%** (71/71 testable clusters) |
| Total Listings Processed | 471 (180 real + 45 real variants + 292 synthetic) |
| Product Clusters | 102 (92 with multiple listings) |
| LLM Calls Required | 80 out of 3,000 candidates (2.7%) |

## Data Overview

**Source**: 5 categories scraped from Zap.co.il public pages

| Category | Seed Products | Listings | Clusters |
|---|---|---|---|
| Smartphones | 27 | ~95 | 8 |
| Headphones | 24 | ~80 | 23 |
| TVs | 26 | ~90 | 25 |
| Laptops | 29 | ~100 | 21 |
| Coffee Machines | 28 | ~106 | 25 |

## Where the Pipeline Succeeded

1. **Hebrew-English brand matching**: Correctly grouped `אפל` / `Apple`, `סמסונג` / `Samsung` listings
2. **Noisy title handling**: Matched products despite reordered tokens, added prefixes, seller noise
3. **Price accuracy**: 100% of testable clusters showed the correct minimum price
4. **Cost efficiency**: 97.3% of decisions were made by rules alone (zero LLM cost)

## Where It Struggled

1. **Near-identical SKUs**: Products like `HP ProBook 4 G1i AD2M0ET` vs `AD2M3ET` are genuinely different but look almost identical (mitigated in iteration 3 with SKU-diff detection)
2. **Ambiguous sub-models**: `Nespresso Essenza Mini D30 EN85` vs `Essenza Mini C30` — different machines in the same series
3. **Missed candidates**: Some real duplicates weren't generated as candidate pairs due to low initial similarity

## Business Insights

1. **Title inconsistency is the norm, not the exception**: 42 out of 134 products (31%) had naturally occurring title variants on Zap itself — comparison vs. zapstore listings. This is before considering cross-retailer variation.

2. **Hebrew-English mixing is pervasive**: Nearly every product has mixed-language naming. Any deduplication system for the Israeli market must handle this as a first-class concern.

3. **Storage/size is the critical differentiator**: The most common false positive pattern involves products that differ only in storage capacity or screen size. Reliable attribute extraction is more important than sophisticated fuzzy matching.

4. **LLMs are cost-effective for the long tail**: The hardest 3% of cases (mixed-language, ambiguous naming) benefit enormously from LLM reasoning, while the other 97% can be handled deterministically. This makes the hybrid approach practical at scale.

5. **Confidence-based routing enables human-in-the-loop**: Low-confidence predictions can be routed to human reviewers, making the system safe for production use even before achieving near-perfect accuracy.

## Iteration Impact

Three improvement iterations yielded a **+3.3 F1 point gain** (0.893 → 0.926), primarily by reducing false positives through better SKU differentiation. The most impactful single change was detecting near-identical part numbers that indicate different product configurations.
