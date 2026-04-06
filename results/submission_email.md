# Submission Email

**Subject:** Take-Home Assignment: Product Deduplication Pipeline

---

Hi,

Please find my submission for the GenAI Exploration Lead take-home assignment.

I built a hybrid pipeline that deduplicates product listings scraped from Zap.co.il. It uses deterministic rules for the easy cases and GPT-4o for ambiguous ones (Hebrew/English mixed titles, brand transliterations, etc.). No hardcoded brand dictionaries or product-specific mappings -- all language-dependent reasoning is delegated to LLMs, so it works across categories without code changes.

Results: **F1=0.9866** averaged across runs on randomly sampled categories (perfumes, AC units, coffee machines, monitors, dryers, and more). **95% cluster purity**, 100% price accuracy on correctly-grouped products. Less than 5% of decisions needed an LLM call.

The repo includes cached data so you can run `python src/main.py` without an API key. The README covers design decisions, evaluation methodology, and known limitations.

GitHub: https://github.com/MaximRaevsky/zap-product-dedup

Happy to discuss.

Best regards,
Maxim Raevsky
