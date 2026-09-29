# Changelog

## 2026-09-29

- Added "Who is being compared": 31 of 35 ADHD children were medicated, no
  control was, so every group contrast is medicated-ADHD vs unmedicated-control;
  age is matched (d = −0.19) and age adjustment moves the effect sizes by ≤0.03;
  usable trials are 555 (ADHD) vs 672 (control). Bootstrap CIs on the LOO AUC
  table: the model parameters' 0.027 gap to accuracy is −0.021 to +0.078, so
  "worse" is corrected to "no better".
- Published `results/` (per-child fits, group comparison, added-value AUCs,
  the new `confounds_check.json`) and `tests/test_readme_numbers.py` (7 tests
  pinning the README numbers to the result files).
