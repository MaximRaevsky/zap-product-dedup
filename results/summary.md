# Results Summary: Product Deduplication Pipeline

## Key Metrics (Cross-Validated, 2 Runs)

| Metric | Run 1 | Run 2 | Average |
|---|---|---|---|
| **Pairwise F1** | 0.9837 | 0.9895 | **0.9866** |
| **Precision** | 0.9998 | 0.9982 | 0.9990 |
| **Recall** | 0.9680 | 0.9810 | 0.9745 |
| **Price Correctness** | 100% | 100% | **100%** |

## Design Philosophy: Zero Hardcoded Dictionaries

This version of the pipeline contains **no hardcoded brand dictionaries, noise phrase lists, or product-specific mappings**. All language-dependent reasoning is delegated to LLMs:

- **Scraper noise filtering**: `gpt-4o-mini` classifies scraped text as product titles vs UI noise
- **Synthetic variant generation**: `gpt-4o-mini` generates realistic title variants for evaluation
- **Brand/product reasoning**: `gpt-4o` reads brands directly from raw titles during ambiguous pair judgment

This makes the pipeline truly category-agnostic and generalizable to new product domains without code changes.

## Data Overview

The pipeline was tested across 2 independent runs, each scraping 10 randomly sampled categories from Zap's full catalogue (~59 categories). Categories spanned:

- **Run 1**: Smart watches, perfumes, fridges, trampolines, car speakers, dryers, dishwashers, ink cartridges, car multimedia (9 effective categories, 1,854 listings, 151 clusters)
- **Run 2**: Men's fragrances, coffee machines, monitors, AC units, dryers, car multimedia (6 effective categories, 1,162 listings, 63 clusters)

## Where the Pipeline Succeeded

1. **Category-agnostic generalization**: F1 > 0.98 across very different product types (electronics, home appliances, beauty, ink cartridges, trampolines)
2. **Real per-store variant matching**: Product page scraping captures genuine retailer naming differences, not just synthetic noise
3. **Price accuracy**: 100% of testable clusters showed the correct minimum price across all runs
4. **Cost efficiency**: Less than 5% of decisions required LLM calls, total cost ~$0.28/run
5. **No manual maintenance**: Zero hardcoded lists mean no need to update brand tables or noise phrases when adding new categories

## Where It Struggled

1. **LLM JSON reliability**: Some `gpt-4o-mini` variant generation batches produce malformed JSON (~2 failures per run), reducing synthetic data volume slightly
2. **Missing categories**: Some Zap categories (furniture, dog food, sports equipment, bags) returned 0 listings, reducing effective coverage
3. **Cross-product noise**: Identical normalized titles across different products (navigation remnants) required a dedicated rule to prevent transitive over-merging
4. **Model_id dependency**: The strongest deduplication signal is Zap's model_id. Cross-platform scenarios without shared IDs would need different strategies

## Business Insights

1. **Title inconsistency is the norm**: Per-store title variants show significant naming variation -- different word order, Hebrew vs English brand names, extra seller information. Any production system must handle this.

2. **Hebrew-English mixing is pervasive across all categories**: Not just electronics. Perfume brands, tool brands, appliance brands all appear in both languages on the same platform.

3. **Structured attributes are the critical differentiator**: Storage size, screen size, and model numbers distinguish truly different products. Reliable attribute extraction matters more than sophisticated fuzzy matching.

4. **LLMs are cost-effective for the long tail**: Less than 5% of decisions need LLM reasoning, but those decisions handle the genuinely ambiguous cases that rules cannot resolve. The hybrid approach is practical at scale.

5. **LLMs replace maintenance burden**: Using LLMs for noise filtering and variant generation eliminates the need to maintain hardcoded dictionaries -- a significant operational advantage as product catalogs grow.

6. **Confidence-based routing enables human-in-the-loop**: Low-confidence predictions can be routed to human reviewers, making the system safe for production even before achieving perfect accuracy.
