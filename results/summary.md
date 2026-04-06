# Results Summary: Product Deduplication Pipeline

## Key Metrics (Cross-Validated, 3 Runs)

| Metric | Run 1 | Run 2 | Run 3 | Average |
|---|---|---|---|---|
| **Pairwise F1** | 0.9958 | 0.9970 | 0.9961 | **0.9963** |
| **Precision** | 1.0000 | 0.9995 | 1.0000 | 0.9998 |
| **Recall** | 0.9917 | 0.9944 | 0.9923 | 0.9928 |
| **Price Correctness** | 100% | 100% | 100% | **100%** |

## Data Overview

The pipeline was tested across 3 independent runs, each scraping 10 randomly sampled categories from Zap's full catalogue (~59 categories). Categories spanned:

- **Run 1**: Smart watches, perfumes, fridges, trampolines, car speakers, dryers, dishwashers, ink cartridges
- **Run 2**: Coffee machines, monitors, AC units, men's fragrances, car multimedia, dryers
- **Run 3**: Car amplifiers, perfumes, cooktops, faucets, oral hygiene, drills, ink, sunglasses, dolls

Each run processed 1,300--2,100 listings and produced 100--220 deduplicated product clusters with the correct minimum price.

## Where the Pipeline Succeeded

1. **Category-agnostic generalization**: F1 > 0.995 across very different product types (electronics, home appliances, beauty, toys, tools)
2. **Real per-store variant matching**: Product page scraping captures genuine retailer naming differences, not just synthetic noise
3. **Price accuracy**: 100% of testable clusters showed the correct minimum price across all runs
4. **Cost efficiency**: Less than 5% of decisions required LLM calls

## Where It Struggled

1. **Scraper noise**: Some navigation links from product pages ("דלג לתפריט") were initially scraped as titles, creating invalid evaluation pairs (fixed with navigation-text filtering)
2. **Missing categories**: Some Zap categories (furniture, dog food, sports) returned 0 listings, reducing effective coverage
3. **Model_id dependency**: The strongest deduplication signal is Zap's model_id. Cross-platform scenarios without shared IDs would need different strategies.

## Business Insights

1. **Title inconsistency is the norm**: Per-store title variants on product comparison pages show significant naming variation -- different word order, Hebrew vs English brand names, extra seller information. Any production system must handle this.

2. **Hebrew-English mixing is pervasive across all categories**: Not just electronics. Perfume brands, tool brands, appliance brands all appear in both languages on the same platform.

3. **Structured attributes are the critical differentiator**: Storage size, screen size, and model numbers distinguish truly different products. Reliable attribute extraction matters more than sophisticated fuzzy matching.

4. **LLMs are cost-effective for the long tail**: Less than 5% of decisions need LLM reasoning, but those decisions handle the genuinely ambiguous cases that rules cannot resolve. The hybrid approach is practical at scale.

5. **Confidence-based routing enables human-in-the-loop**: Low-confidence predictions can be routed to human reviewers, making the system safe for production even before achieving perfect accuracy.
