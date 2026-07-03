# /// script
# requires-python = ">=3.10"
# dependencies = ["deepsig", "scipy", "numpy<2"]
# ///
"""Reference values for statskit's hypothesis-test layer.

- ASO violation ratio: deepsig.aso.compute_violation_ratio (Dror et al. 2019
  reference implementation), deterministic given scores and dt.
- Mann-Whitney U: scipy.stats.mannwhitneyu, two-sided, method="asymptotic"
  (matching statskit's normal approximation; scipy's default would switch to
  the exact method on small tie-free samples), use_continuity=True.
"""
import json

import numpy as np
from deepsig.aso import compute_violation_ratio
from scipy import stats

rng = np.random.default_rng(7)

score_cases = [
    ("a_dominates", rng.normal(0.6, 0.1, 40), rng.normal(0.4, 0.1, 40)),
    ("overlapping", rng.normal(0.52, 0.12, 35), rng.normal(0.48, 0.12, 45)),
    ("b_dominates", rng.normal(0.4, 0.08, 30), rng.normal(0.55, 0.1, 30)),
    ("with_ties", np.repeat([0.2, 0.5, 0.5, 0.8], 8), np.repeat([0.3, 0.5, 0.7], 10)),
]

out = {"cases": []}
for name, a, b in score_cases:
    a = np.round(np.asarray(a, dtype=float), 8)
    b = np.round(np.asarray(b, dtype=float), 8)
    ratio = compute_violation_ratio(scores_a=a, scores_b=b, dt=0.005)
    mw = stats.mannwhitneyu(
        a, b, alternative="two-sided", method="asymptotic", use_continuity=True
    )
    out["cases"].append(
        {
            "name": name,
            "a": a.tolist(),
            "b": b.tolist(),
            "violation_ratio": float(ratio),
            "mannwhitney_u_a": float(mw.statistic),
            "mannwhitney_p": float(mw.pvalue),
        }
    )
    print(f"{name}: ratio={ratio:.6f} U_a={mw.statistic} p={mw.pvalue:.6g}")

import pathlib

with open(pathlib.Path(__file__).parent / "statskit_hypothesis.json", "w") as f:
    json.dump(out, f)
print("wrote stats_reference.json")
