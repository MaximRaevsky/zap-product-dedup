# Iteration Log

## Iteration 1 (Baseline)
- **F1: 0.8934** (P=0.8757, R=0.9118)
- TP=310, FP=44, FN=30, TN=209
- Price accuracy: 100%
- Clusters: 78 (70 grouped)
- Issues found:
  - Storage parsing failed for "12GB+256GB" format → false "conflicting storage"
  - Very similar part numbers (AD2M0ET vs AD2M3ET) treated as same product
  - `brand_high_token_sort` rule too aggressive for near-identical SKUs

## Iteration 2
- **F1: 0.9004** (P=0.8839, R=0.9176)
- TP=312, FP=41, FN=28, TN=212
- Improvements:
  - Fixed combined storage format extraction (8GB+256GB)
  - Added `_has_differing_part_numbers` heuristic to detect SKU differences
  - Reduced FPs by 3, FNs by 2

## Iteration 3 (Final)
- **F1: 0.9256** (P=0.9367, R=0.9147)
- TP=311, FP=21, FN=29, TN=232
- Improvements:
  - Broadened part-number detection pattern to catch more SKU formats
  - Made rules_fallback more conservative (threshold 0.55 vs 0.5)
  - FPs halved from 41→21, significant precision gain
  - Price accuracy: 100% (71/71 testable clusters)

## Summary
| Metric     | Iter 1 | Iter 2 | Iter 3 |
|------------|--------|--------|--------|
| Precision  | 0.876  | 0.884  | 0.937  |
| Recall     | 0.912  | 0.918  | 0.915  |
| F1         | 0.893  | 0.900  | 0.926  |
| FP         | 44     | 41     | 21     |
| FN         | 30     | 28     | 29     |
| Price Acc. | 100%   | 100%   | 100%   |
