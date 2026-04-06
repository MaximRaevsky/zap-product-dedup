# Submission Email

**Subject:** Take-Home Assignment: Product Deduplication Pipeline

---

Hi,

Please find attached my submission for the GenAI Exploration Lead take-home assignment.

I built a hybrid product deduplication pipeline that combines deterministic rules with GPT-4o for the ambiguous cases. The system processes real product data scraped from Zap.co.il across 5 categories (smartphones, headphones, TVs, laptops, coffee machines), handles Hebrew/English/mixed-language titles, and outputs deduplicated product groups with the lowest price per product.

Key results:
- **F1: 0.926** (Precision: 0.937, Recall: 0.915)
- **100% price correctness** on testable clusters
- Only **2.7% of decisions** required LLM calls — the rest were handled by deterministic rules
- Three evaluation-driven iterations, each improving on concrete failure analysis

The repository includes full source code, a realistic evaluation dataset (real Zap data + synthetic augmentation), comprehensive debug artifacts, and a detailed README explaining the design decisions.

GitHub: [link to repo]

Happy to discuss the approach, trade-offs, or next steps for productionization.

Best regards,
Maxim Raevsky
