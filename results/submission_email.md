# Submission Email

**Subject:** Take-Home Assignment: Product Deduplication Pipeline

---

Hi,

Please find my submission for the GenAI Exploration Lead take-home assignment.

I built a hybrid product deduplication pipeline that combines deterministic rules with GPT-4o for ambiguous cases. The system processes real product data scraped from Zap.co.il across randomly sampled categories (not hardcoded -- a different mix each run), handles Hebrew/English/mixed-language titles, and outputs deduplicated product groups with the lowest price per product.

Key results:
- **Average F1: 0.9963** across 3 cross-validated runs with different random category samples
- **100% price correctness** on all testable clusters
- Less than **5% of decisions** required LLM calls -- the rest were handled by deterministic rules
- Tested on diverse categories: electronics, appliances, perfumes, tools, toys, and more

The pipeline scrapes real per-store title variants from Zap product pages (the actual naming inconsistencies referenced in the assignment), augments with synthetic variants for evaluation, and uses a minimal 7-rule matching system backed by GPT-4o for edge cases.

The repository includes full source code, evaluation dataset, comprehensive debug artifacts, and a detailed README explaining design decisions.

GitHub: https://github.com/MaximRaevsky/zap-product-dedup

Happy to discuss the approach, trade-offs, or next steps for productionization.

Best regards,
Maxim Raevsky
