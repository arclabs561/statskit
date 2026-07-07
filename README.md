# statskit

[![crates.io](https://img.shields.io/crates/v/statskit.svg)](https://crates.io/crates/statskit)
[![Documentation](https://docs.rs/statskit/badge.svg)](https://docs.rs/statskit)

Statistical metrics and tests.

`statskit` works on plain `f64` and `usize` slices. It provides classification,
calibration, regression, descriptive statistics, and paired or unpaired system
comparison tests. Optional features add serde support and SIMD reductions.

## Quickstart

```toml
[dependencies]
statskit = "0.2"
```

```rust
use statskit::{accuracy, f1, Average, mse, cohens_d, bootstrap_bca, mean_diff, BootstrapConfig};

// Classification: parallel integer label slices.
let y_true = [0usize, 1, 1, 0, 1];
let y_pred = [0usize, 1, 0, 0, 1];
let acc = accuracy(&y_true, &y_pred);          // 0.8
let macro_f1 = f1(&y_true, &y_pred, Average::Macro);

// Regression error.
let err = mse(&[1.0, 2.0, 3.0], &[1.1, 1.9, 3.2]);

// Compare two systems: effect size plus a bootstrap CI on the mean difference.
let a = [0.81, 0.79, 0.88, 0.90, 0.85];
let b = [0.74, 0.71, 0.80, 0.77, 0.69];
let d = cohens_d(&a, &b);
let ci = bootstrap_bca(&a, &b, mean_diff, BootstrapConfig::default());
// ci.point_estimate, ci.lower, ci.upper, ci.p_value
```

## Modules

- `statskit::classify`: `accuracy`, `precision`, `recall`, `f1`, `fbeta`, `mcc`,
  `roc_curve` / `roc_auc`, `pr_curve` / `average_precision`, `confusion_matrix`,
  `classification_report`, `log_loss`, `balanced_accuracy`, `specificity`,
  `cohen_kappa`, `hamming_loss`, `jaccard_score`. Multi-class metrics take an
  `Average` (`Micro`, `Macro`, `Weighted`).
- `statskit::calibration`: `brier_score`, `expected_calibration_error`,
  `maximum_calibration_error`, `reliability_diagram`.
- `statskit::regression`: `mse`, `rmse`, `mae`, `r_squared`.
- `statskit::stats`: descriptive moments (`mean`, `variance_population` /
  `variance_sample`, `stddev_population` / `stddev_sample`) plus tests and effect
  sizes (`bootstrap_bca`, `wilcoxon`, `mann_whitney`, `permutation_test`,
  `mcnemar`, `friedman`, `aso`, `cohens_d`, `rank_biserial`, `mean_diff`,
  `benjamini_hochberg`, `bonferroni`).

## Features

- `serde`: derive `Serialize` / `Deserialize` on result structs.
- `simd`: SIMD-accelerated `f64` reductions in the regression metrics, via `innr`.

## Status

Experimental, intentionally small surface. A metric is added only when it has a
downstream use case and tests; functions making a statistical claim (CI,
p-value) state their assumptions in the rustdoc.
