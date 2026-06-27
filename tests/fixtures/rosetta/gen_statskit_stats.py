# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "numpy"]
# ///
"""Rosetta fixture generator for statskit regression + descriptive statistics.

Provenance for statskit_stats.json. All EXACT library oracles:
  regression: mse/rmse/mae/r_squared vs sklearn.metrics (rmse = sqrt(mse))
  descriptive: mean/variance/stddev (population ddof=0, sample ddof=1) vs numpy

Deferred (convention-sensitive, need an impl read first): the statistical tests
in stats.rs (wilcoxon, mann_whitney, cohens_d, friedman, mcnemar) whose p-value
method / pooled-std / continuity-correction conventions must be matched to scipy
before a fixture is safe.

Regenerate: uv run tests/fixtures/rosetta/gen_statskit_stats.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 0
rng = np.random.default_rng(SEED)

xs = rng.normal(5.0, 2.0, size=30)
y_true = rng.normal(0.0, 1.0, size=30)
y_pred = y_true + rng.normal(0.0, 0.5, size=30)

mse = float(mean_squared_error(y_true, y_pred))

expected = {
    "mean": float(np.mean(xs)),
    "variance_population": float(np.var(xs, ddof=0)),
    "variance_sample": float(np.var(xs, ddof=1)),
    "stddev_population": float(np.std(xs, ddof=0)),
    "stddev_sample": float(np.std(xs, ddof=1)),
    "mse": mse,
    "rmse": float(np.sqrt(mse)),
    "mae": float(mean_absolute_error(y_true, y_pred)),
    "r_squared": float(r2_score(y_true, y_pred)),
}

fixture = {
    "provenance": {
        "generator": "gen_statskit_stats.py",
        "library": "scikit-learn + numpy",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "regression vs sklearn.metrics; descriptive vs numpy (ddof 0/1).",
    },
    "xs": xs.tolist(),
    "y_true": y_true.tolist(),
    "y_pred": y_pred.tolist(),
    "expected": expected,
}

out = Path(__file__).parent / "statskit_stats.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
for key, val in expected.items():
    print(f"{key:22s} {val:.12f}")
print(f"wrote {out}")
