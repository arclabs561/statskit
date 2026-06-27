# /// script
# requires-python = ">=3.10"
# dependencies = ["scikit-learn", "numpy"]
# ///
"""Rosetta fixture generator for statskit classification + calibration metrics.

This script IS the provenance for statskit_classify.json: every expected value
is computed by scikit-learn at the version recorded in the fixture's provenance
block. The Rust side (tests/rosetta_statskit.rs) asserts statskit matches these
references within tolerance. Regenerate with:

    uv run tests/fixtures/rosetta/gen_statskit.py

Deferred from this wave (no canonical sklearn reference): ECE, MCE. They need a
documented reference implementation before they get a fixture.
"""

import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 0
rng = np.random.default_rng(SEED)

# Binary dataset. Distinct scores (no ties) so ROC/PR curve construction is
# unambiguous across implementations. Probabilities kept inside (0.02, 0.98) so
# neither statskit's nor sklearn's log-loss epsilon clip activates.
n_bin = 50
bin_y_true = rng.integers(0, 2, size=n_bin)
bin_signal = bin_y_true * 0.8 + rng.normal(0.0, 0.6, size=n_bin)
# Make every score distinct.
bin_y_score = bin_signal + np.linspace(0, 1e-6, n_bin)
bin_y_pred = (bin_y_score > np.median(bin_y_score)).astype(int)
bin_y_prob = np.clip(1.0 / (1.0 + np.exp(-bin_signal)), 0.02, 0.98)

# Multiclass dataset, 3 classes, all present in y_true, with noisy predictions.
n_multi = 60
n_classes = 3
multi_y_true = rng.integers(0, n_classes, size=n_multi)
multi_y_true[:3] = [0, 1, 2]  # guarantee all classes present
flip = rng.random(n_multi) < 0.3
multi_y_pred = np.where(flip, rng.integers(0, n_classes, size=n_multi), multi_y_true)

# sklearn log_loss wants per-class probabilities; pass [P(0), P(1)] with explicit
# labels so it is unambiguous and matches statskit's binary P(class 1) formula.
bin_prob_2col = np.column_stack([1.0 - bin_y_prob, bin_y_prob])

expected = {
    "accuracy_bin": accuracy_score(bin_y_true, bin_y_pred),
    "mcc_bin": matthews_corrcoef(bin_y_true, bin_y_pred),
    "mcc_multi": matthews_corrcoef(multi_y_true, multi_y_pred),
    "roc_auc_bin": roc_auc_score(bin_y_true, bin_y_score),
    "average_precision_bin": average_precision_score(bin_y_true, bin_y_score),
    "log_loss_bin": log_loss(bin_y_true, bin_prob_2col, labels=[0, 1]),
    "brier_bin": brier_score_loss(bin_y_true, bin_y_prob),
    "cohen_kappa_multi": cohen_kappa_score(multi_y_true, multi_y_pred),
    "balanced_accuracy_multi": balanced_accuracy_score(multi_y_true, multi_y_pred),
    "precision_micro_multi": precision_score(multi_y_true, multi_y_pred, average="micro"),
    "precision_macro_multi": precision_score(multi_y_true, multi_y_pred, average="macro"),
    "recall_macro_multi": recall_score(multi_y_true, multi_y_pred, average="macro"),
    "f1_micro_multi": f1_score(multi_y_true, multi_y_pred, average="micro"),
    "f1_macro_multi": f1_score(multi_y_true, multi_y_pred, average="macro"),
    "f1_weighted_multi": f1_score(multi_y_true, multi_y_pred, average="weighted"),
    "jaccard_macro_multi": jaccard_score(multi_y_true, multi_y_pred, average="macro"),
}

confusion_multi = confusion_matrix(
    multi_y_true, multi_y_pred, labels=list(range(n_classes))
).tolist()

fixture = {
    "provenance": {
        "generator": "gen_statskit.py",
        "library": "scikit-learn",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "seed": SEED,
        "note": "External ground truth. statskit must match within tolerance.",
    },
    "datasets": {
        "bin": {
            "y_true": bin_y_true.tolist(),
            "y_pred": bin_y_pred.tolist(),
            "y_score": bin_y_score.tolist(),
            "y_prob": bin_y_prob.tolist(),
        },
        "multi": {
            "y_true": multi_y_true.tolist(),
            "y_pred": multi_y_pred.tolist(),
            "n_classes": n_classes,
        },
    },
    "expected": {k: float(v) for k, v in expected.items()},
    "confusion_multi": confusion_multi,
}

out = Path(__file__).parent / "statskit_classify.json"
out.write_text(json.dumps(fixture, indent=2) + "\n")
for k, v in expected.items():
    print(f"{k:28s} {v:.12f}")
print(f"\nwrote {out}")
