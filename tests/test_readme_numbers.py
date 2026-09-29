"""Pin the numbers quoted in README.md to the result files, so they cannot drift.

Run with:  python3 -m pytest tests/
Needs only the standard library plus numpy.
"""
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
FITS = json.loads((ROOT / "results" / "cohort_fits.json").read_text())
CHECK = json.loads((ROOT / "results" / "confounds_check.json").read_text())
ADDED = json.loads((ROOT / "results" / "added_value.json").read_text())
GROUPS = json.loads((ROOT / "results" / "group_comparison.json").read_text())


def r3(x):
    return float(np.floor(x * 1000 + 0.5) / 1000)  # half-up, not Python's half-even


def test_cohort_composition():
    adhd = [r for r in FITS if r["adhd"] == 1]
    ctrl = [r for r in FITS if r["adhd"] == 0]
    assert len(FITS) == 79 and len(adhd) == 35 and len(ctrl) == 44
    assert sum(r["medicated"] for r in adhd) == 31
    assert sum(r["medicated"] for r in ctrl) == 0
    assert "35 with an ADHD diagnosis (31 of them medicated" in README
    assert CHECK["composition"]["adhd"]["medicated"] == 31


def test_trial_counts_by_group():
    adhd = [r["n_trials"] for r in FITS if r["adhd"] == 1]
    ctrl = [r["n_trials"] for r in FITS if r["adhd"] == 0]
    assert round(np.median([r["n_trials"] for r in FITS])) == 639
    assert round(np.median(adhd)) == 555 and round(np.median(ctrl)) == 672
    assert "555 in the ADHD group, 672 in controls" in README
    assert CHECK["composition"]["n_trials_p_mannwhitney"] < 0.005
    assert abs(CHECK["correlations"]["n_trials~accuracy"] - 0.76) < 0.005
    assert abs(CHECK["correlations"]["v~n_trials"] - 0.75) < 0.005


def test_age_is_matched():
    c = CHECK["composition"]
    assert round(c["adhd"]["age_mean"], 1) == 10.3 and round(c["control"]["age_mean"], 1) == 10.5
    assert round(c["age_cohens_d"], 2) == -0.19
    assert round(c["age_p_mannwhitney"], 2) == 0.44
    adj = CHECK["age_adjusted"]
    assert round(adj["v"]["cohens_d_age_adjusted"], 2) == -0.85
    assert round(adj["a"]["cohens_d_age_adjusted"], 2) == -0.79
    assert round(adj["t0"]["cohens_d_age_adjusted"], 2) == -0.56
    # raw d in the README table equals the raw d recomputed here
    for k in ("v", "a", "t0"):
        assert abs(adj[k]["cohens_d_raw"] - GROUPS[k]["cohens_d"]) < 0.005
    # the largest shift from age adjustment is 0.03
    shifts = [abs(adj[k]["cohens_d_raw"] - adj[k]["cohens_d_age_adjusted"]) for k in ("v", "a", "t0")]
    assert max(shifts) < 0.035
    assert "drift −0.87 → −0.85, boundary −0.82 → −0.79, non-decision −0.55 → −0.56" in README


def test_medication_split_inside_adhd():
    s = CHECK["adhd_medication_split"]
    assert s["n_medicated"] == 31 and s["n_unmedicated"] == 4
    assert round(s["v"]["unmedicated_mean"], 2) == 1.04 and round(s["v"]["medicated_mean"], 2) == 0.90
    assert round(s["a"]["unmedicated_mean"], 2) == 1.38 and round(s["a"]["medicated_mean"], 2) == 1.47


def test_auc_table_and_intervals():
    sets = CHECK["loo_auc"]["sets"]
    # the point estimates are the ones in added_value.json (same LOO procedure)
    for name, s in sets.items():
        assert abs(s["loo_auc"] - ADDED[name]["loo_auc"]) < 1e-9
    assert r3(sets["accuracy alone"]["loo_auc"]) == 0.729
    assert r3(sets["model parameters"]["loo_auc"]) == 0.703
    lo, hi = sets["accuracy alone"]["ci95"]
    assert round(lo, 2) == 0.61 and round(hi, 2) == 0.83
    lo, hi = sets["model parameters"]["ci95"]
    assert round(lo, 2) == 0.59 and round(hi, 2) == 0.81
    d = CHECK["loo_auc"]["differences_vs_accuracy"]["model parameters"]
    assert r3(d["accuracy_minus_this"]) == 0.027
    assert round(d["ci95"][0], 3) == -0.021 and round(d["ci95"][1], 3) == 0.078
    assert round(d["share_of_resamples_this_beats_accuracy"], 2) == 0.14
    assert "bootstrap interval of −0.021 to +0.078" in README
    assert "ahead in 14 % of resamples" in README


def test_only_one_difference_clears_zero():
    diffs = CHECK["loo_auc"]["differences_vs_accuracy"]
    clears = [n for n, d in diffs.items() if d["ci95"][0] > 0 or d["ci95"][1] < 0]
    assert clears == ["accuracy + RT variability"]
    assert round(diffs["accuracy + RT variability"]["ci95"][0], 3) == 0.002
    # every interval on an AUC is about two tenths wide
    for s in CHECK["loo_auc"]["sets"].values():
        assert 0.18 < s["ci95"][1] - s["ci95"][0] < 0.25


def test_readme_no_longer_says_worse():
    m = re.search(r"produces a classifier that is (.+?) than the proportion", README)
    assert m and m.group(1) == "no better"
