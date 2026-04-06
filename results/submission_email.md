# Submission Email

**Subject:** Take-Home Assignment: Product Deduplication Pipeline

---

Hi,

Please find my submission for the GenAI Exploration Lead take-home assignment.

I built a hybrid product deduplication pipeline that combines deterministic rules with GPT-4o for ambiguous cases. The system processes real product data scraped from Zap.co.il across randomly sampled categories (a different mix each run), handles Hebrew/English/mixed-language titles, and outputs deduplicated product groups with the lowest price per product.

Key results:
- **Average F1: 0.9866** across cross-validated runs with different random category samples
- **95% cluster purity**, with 100% price accuracy on correctly-grouped products
- Customers save **17.5% on average** (median 164 NIS) by seeing the grouped cheapest price
- Less than **5% of decisions** required LLM calls -- the rest were handled by deterministic rules
- **Zero hardcoded dictionaries** -- all language-dependent reasoning (brand mapping, noise filtering, variant generation) is delegated to LLMs
- Tested on diverse categories: electronics, appliances, perfumes, ink cartridges, trampolines, and more

The pipeline deliberately avoids hardcoded brand dictionaries or noise phrase lists. Instead, it uses GPT-4o-mini for scraper noise filtering and synthetic data generation, and GPT-4o for ambiguous pair judgment -- making it category-agnostic and generalizable without code changes.

The repository includes full source code, evaluation dataset, comprehensive debug artifacts, and a detailed README explaining design decisions.

GitHub: https://github.com/MaximRaevsky/zap-product-dedup

Happy to discuss the approach, trade-offs, or next steps for productionization.

Best regards,
Maxim Raevsky
