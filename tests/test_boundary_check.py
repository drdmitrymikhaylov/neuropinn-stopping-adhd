"""Pin the "What the boundary restates" section and the 2026-10-30 additions
to results/boundary_check.json and results/cohort_fits.json.

Standard library plus numpy; the LOO AUCs and exact tests themselves are
recomputed privately (src/boundary_check.py) and read here from the JSON, but
every correlation, R² and residualised effect size is recomputed from the
per-child fits.
"""
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = " ".join((ROOT / "README.md").read_text().split())
FITS = json.loads((ROOT / "results" / "cohort_fits.json").read_text())
BC = json.loads((ROOT / "results" / "boundary_check.json").read_text())
ADDED = json.loads((ROOT / "results" / "added_value.json").read_text())
GROUPS = json.loads((ROOT / "results" / "group_comparison.json").read_text())


def col(k):
    return np.array([float(r[k]) for r in FITS])


def resid(x, c):
    return x - np.polyval(np.polyfit(c, x, 1), c)


def cohens_d(x, g):
    x1, x0 = x[g == 1], x[g == 0]
    sp = np.sqrt(((x1.size - 1) * x1.var(ddof=1) + (x0.size - 1) * x0.var(ddof=1)) / (x1.size + x0.size - 2))
    return (x1.mean() - x0.mean()) / sp


Y = col("adhd").astype(int)
ACC, V, A, T0, W, CV = (col(k) for k in ("accuracy", "v", "a", "t0", "w", "cv_rt"))


def test_identity_sigmoid_w_is_accuracy():
    dev = 1 / (1 + np.exp(-W)) - ACC
    assert abs(dev.mean()) < 0.0005 and round(dev.std(ddof=1), 3) == 0.064 and round(np.abs(dev).max(), 3) == 0.138
    logit = np.log(ACC / (1 - ACC))
    assert round(np.corrcoef(V, logit)[0, 1], 2) == 0.98
    assert round(np.corrcoef(W, logit)[0, 1], 2) == 0.97
    assert round(np.corrcoef(V, ACC)[0, 1], 2) == 0.95  # the figure the page quotes
    assert round(BC["identity"]["corr_v_logit_accuracy"], 2) == 0.98
    assert "differs from accuracy by 0.0002 on average over the 79 children, sd 0.064, at most 0.138" in README
    assert "0.98 with logit(accuracy)** (*w* 0.97)" in README


def test_boundary_correlates_more_with_accuracy_than_variability():
    r_acc, r_cv = np.corrcoef(A, ACC)[0, 1], np.corrcoef(A, CV)[0, 1]
    assert round(r_acc, 2) == 0.67 and round(r_cv, 2) == -0.55
    assert abs(r_acc) > abs(r_cv)
    assert abs(r_acc - ADDED["correlations"]["a~accuracy"]) < 1e-9
    assert abs(r_cv - ADDED["correlations"]["a~cv_rt"]) < 1e-9
    assert round(np.corrcoef(ACC, CV)[0, 1], 2) == -0.65
    # partial correlations, recomputed
    p_cv = np.corrcoef(resid(A, ACC), resid(CV, ACC))[0, 1]
    p_acc = np.corrcoef(resid(A, CV), resid(ACC, CV))[0, 1]
    assert round(p_cv, 2) == -0.20 and round(p_acc, 2) == 0.49
    b = BC["boundary"]
    assert abs(b["partial_a_cv_rt_given_accuracy"]["r"] - p_cv) < 1e-9
    lo, hi = b["partial_a_cv_rt_given_accuracy"]["ci95"]
    assert round(lo, 2) == -0.41 and round(hi, 2) == 0.02 and hi > 0  # interval reaches zero
    assert round(b["partial_a_cv_rt_given_accuracy"]["p"], 2) == 0.07
    lo, hi = b["partial_a_accuracy_given_cv_rt"]["ci95"]
    assert round(lo, 2) == 0.30 and round(hi, 2) == 0.64
    assert "partial correlation with variability is **−0.20** (95 % CI −0.41 to +0.02, *p* = 0.07)" in README
    assert "partial correlation with accuracy is **+0.49** (+0.30 to +0.64)" in README
    # the old sentence is gone from the live text (kept only as a quoted correction)
    assert README.count("not primarily with accuracy") == 1
    assert 'An earlier version of this page claimed the boundary *a* "correlates with reaction-time variability' in README


def test_variance_shares_and_within_group():
    def r2(y, *xs):
        X = np.column_stack([np.ones(y.size)] + list(xs))
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert round(100 * r2(A, ACC)) == 45 and round(100 * r2(A, CV)) == 31 and round(100 * r2(A, ACC, CV)) == 47
    assert "Accuracy explains 45 % of the variance of *a*, variability 31 %, both together 47 %" in README
    adhd, ctrl = Y == 1, Y == 0
    assert round(np.corrcoef(A[adhd], CV[adhd])[0, 1], 2) == -0.63 and round(np.corrcoef(A[adhd], ACC[adhd])[0, 1], 2) == 0.54
    assert round(np.corrcoef(A[ctrl], ACC[ctrl])[0, 1], 2) == 0.71 and round(np.corrcoef(A[ctrl], CV[ctrl])[0, 1], 2) == -0.49
    assert "(variability −0.63, accuracy 0.54; controls 0.71 and −0.49)" in README


def test_group_effects_vanish_once_accuracy_is_removed():
    expect = {"a": (-0.82, -0.23), "v": (-0.87, 0.08), "t0": (-0.55, -0.11), "cv_rt": (0.64, 0.05)}
    for k, (raw, res) in expect.items():
        x = col(k)
        assert round(cohens_d(x, Y), 2) == raw
        assert abs(cohens_d(x, Y) - GROUPS[k]["cohens_d"]) < 1e-9
        assert round(cohens_d(resid(x, ACC), Y), 2) == res
        g = BC["boundary"]["group_effect_residualised_on_accuracy"][k]
        assert abs(g["cohens_d_residualised"] - cohens_d(resid(x, ACC), Y)) < 1e-9
        assert g["p_mannwhitney_residualised"] > 0.5
    assert round(BC["boundary"]["group_effect_residualised_on_accuracy"]["a"]["p_mannwhitney_residualised"], 2) == 0.68
    assert "goes from −0.82 to **−0.23** (*p* = 0.68)" in README
    assert "drift −0.87 → +0.08, non-decision time −0.55 → −0.11, and reaction-time variability itself +0.64 → +0.05" in README


def test_loo_auc_accuracy_plus_one_table():
    t = BC["boundary"]["loo_auc_accuracy_plus_one"]
    assert abs(t["accuracy alone"]["loo_auc"] - ADDED["accuracy alone"]["loo_auc"]) < 1e-12
    rows = [("accuracy alone", 0.729), ("accuracy + a", 0.729), ("accuracy + v", 0.726), ("accuracy + t0", 0.710),
            ("w alone", 0.715), ("v alone", 0.704), ("a alone", 0.679), ("a + RT variability", 0.681),
            ("RT variability alone", 0.640), ("t0 alone", 0.606)]
    for name, val in rows:
        assert round(t[name]["loo_auc"], 3) == val, name
    assert round(t["accuracy alone"]["loo_auc"], 4) == 0.7292 and round(t["accuracy + a"]["loo_auc"], 4) == 0.7286
    assert round(t["a + RT variability"]["loo_auc"] - t["a alone"]["loo_auc"], 3) == 0.001
    for cell in ("| accuracy + boundary *a* | 0.729 |", "| accuracy + drift *v* | 0.726 |", "| boundary *a* alone | 0.679 |",
                 "| boundary *a* + RT variability | 0.681 |", "| non-decision *t₀* alone | 0.606 |", "| *w* = *v*·*a* alone | 0.715 |"):
        assert cell in README, cell
    assert "(0.7292 → 0.7286)" in README


def test_unmedicated_four_against_controls():
    u = BC["unmedicated"]
    assert u["n_unmedicated"] == 4 and u["n_control"] == 44 and u["n_medicated"] == 31
    unmed = [r for r in FITS if r["adhd"] == 1 and not r["medicated"]]
    assert [r["subject"] for r in unmed] == u["subjects"]
    ctrl_v = np.array([r["v"] for r in FITS if r["adhd"] == 0])
    v4 = [r["v"] for r in unmed]
    assert [round(x, 2) for x in v4] == [1.12, 0.13, 1.32, 1.58]
    assert [round(100 * np.mean(ctrl_v < x)) for x in v4] == [30, 0, 34, 52]
    assert round(np.mean(v4), 2) == 1.04
    ctrl_a = np.array([r["a"] for r in FITS if r["adhd"] == 0])
    a4 = [r["a"] for r in unmed]
    assert [round(x, 2) for x in a4] == [1.33, 1.33, 1.52, 1.34]
    assert [round(100 * np.mean(ctrl_a < x)) for x in a4] == [7, 7, 41, 7]
    m = u["measures"]
    assert round(m["v"]["p_exact_unmedicated_vs_control"], 2) == 0.18
    assert round(m["a"]["p_exact_unmedicated_vs_control"], 3) == 0.020
    assert round(m["cv_rt"]["p_exact_unmedicated_vs_control"], 3) == 0.028
    assert round(m["t0"]["p_exact_unmedicated_vs_control"], 2) == 0.36
    assert round(m["accuracy"]["p_exact_unmedicated_vs_control"], 2) == 0.24
    assert [round(p) for p in m["cv_rt"]["percentile_among_controls"]] == [95, 100, 59, 77]
    assert len(m) == 10
    assert round(min(x["p_exact_medicated_vs_unmedicated"] for x in m.values()), 2) == 0.19
    assert round(1 / u["min_two_sided_exact_p"]) == 97290
    assert "1.12, 0.13, 1.32 and 1.58, which sit at the 30th, 0th, 34th and 52nd percentiles" in README
    assert "(1.33, 1.33, 1.52, 1.34 — three at the 7th percentile of controls; *p* = 0.020)" in README
    assert "(95th, 100th, 59th, 77th percentiles; *p* = 0.028)" in README
    assert "*p* ≥ 0.19 on every measure" in README and "1 in 97 290" in README


def test_power_of_four_against_44():
    pw = BC["unmedicated"]["power"]
    assert round(pw["drift"]["power_4_vs_44"], 2) == 0.37
    assert round(pw["drift"]["power_31_vs_44"], 2) == 0.96
    assert round(pw["mde_80pct_4_vs_44"], 2) == 1.49
    assert round(pw["power_at_drift_d_by_n_unmedicated"]["16"], 2) == 0.84
    assert "power of **0.37** to find the published drift effect (*d* = 0.87) and needs *d* = 1.49 for 80 %; sixteen unmedicated children would give 0.84" in README


def test_age_residualised_loo_aucs():
    ar = BC["age_residualised_loo_auc"]
    expect = {"accuracy alone": (0.729, 0.715), "accuracy + RT variability": (0.712, 0.701),
              "classical summary": (0.712, 0.700), "model parameters": (0.703, 0.700), "model + accuracy": (0.713, 0.705)}
    for k, (raw, res) in expect.items():
        assert round(ar[k]["loo_auc_raw"], 3) == raw and round(ar[k]["loo_auc_age_residualised"], 3) == res, k
        assert abs(ar[k]["loo_auc_raw"] - ADDED[k]["loo_auc"]) < 1e-12
        assert abs(ar[k]["loo_auc_raw"] - ar[k]["loo_auc_age_residualised"]) <= 0.015 + 1e-9
    order_raw = sorted(expect, key=lambda k: -ar[k]["loo_auc_raw"])
    assert order_raw[0] == "accuracy alone" and sorted(expect, key=lambda k: -ar[k]["loo_auc_age_residualised"])[0] == "accuracy alone"
    gap = ar["accuracy alone"]["loo_auc_age_residualised"] - ar["model parameters"]["loo_auc_age_residualised"]
    assert round(gap, 3) == 0.015
    assert round(ar["age alone"]["loo_auc"], 2) == 0.43
    plus = [ar[k]["loo_auc_raw_plus_age"] for k in expect]
    assert all(ar[k]["loo_auc_raw_plus_age"] < ar[k]["loo_auc_raw"] for k in expect)
    assert round(min(plus), 3) == 0.687 and round(max(plus), 3) == 0.709
    assert "accuracy alone 0.729 → **0.715**, accuracy + RT variability 0.712 → 0.701, four classical measures 0.712 → 0.700, three model parameters 0.703 → 0.700, model + accuracy 0.713 → 0.705" in README
    assert "narrows from 0.027 to 0.015" in README and "(0.687–0.709)" in README
