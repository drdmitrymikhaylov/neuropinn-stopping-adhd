# Changelog

## 2026-10-30

- Correction: the page said the boundary *a* "correlates with reaction-time
  variability (−0.55) and not primarily with accuracy". By the repository's
  own `added_value.json` it correlates 0.67 with accuracy; the partial
  correlation with variability given accuracy is −0.20 (p = 0.07), and the
  boundary's group effect size falls from −0.82 to −0.23 once accuracy is
  residualised out (drift −0.87 → +0.08, t₀ −0.55 → −0.11). Accuracy + a
  classifies at 0.729, the same as accuracy alone. New section "What the
  boundary restates"; the drift correlates 0.98 with logit(accuracy).
- The four unmedicated ADHD children, with numbers: drift p = 0.18 against
  the 44 controls (three at the 30th–52nd control percentile, one below every
  control), boundary p = 0.020 and RT variability p = 0.028 (exact
  Mann-Whitney, ten measures, uncorrected); power of 4 vs 44 at d = 0.87 is
  0.37. Age-residualised LOO AUCs added (accuracy 0.729 → 0.715, model
  parameters 0.703 → 0.700; order kept).
- `results/boundary_check.json`, `tests/test_boundary_check.py`.

## 2026-10-19

- LICENSE now holds the standard CC BY 4.0 legal text, so GitHub recognises the licence (it showed "Other" before). The terms are unchanged; the note on what each licence covers moved to `NOTICE.md`.

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
