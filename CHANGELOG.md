# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- `wilcoxon`, `friedman`, `balanced_accuracy` and macro-averaged
  classification metrics can return different values; signatures are
  unchanged.

### Fixed

- `wilcoxon` uses the exact null distribution for at most 20 nonzero,
  untied differences, as documented. Five positive differences now give
  p = 0.0625 (was 0.043 from the normal approximation), matching
  `scipy.stats.wilcoxon`.
- `friedman` divides the statistic by the tie correction
  `1 - sum(t^3 - t) / (n k (k^2 - 1))`, matching
  `scipy.stats.friedmanchisquare`.
- `Average::Macro` averages over labels that occur in `y_true` or `y_pred`,
  not over every id up to the largest label. A perfect prediction on labels
  {0, 2} now scores 1.0 (was 0.667), matching scikit-learn.
- `balanced_accuracy` averages recall over classes present in `y_true`
  only; classes seen only in predictions no longer count as recall 0.

## 0.3.1 - Pending publication

### Changed

- Corrected ranking metrics and statistical tests can return different numeric
  results. ROC/PR curves group tied scores into distinct thresholds, so curve
  lengths can change; public function signatures are unchanged.
- Classification and statistical sorting uses floating-point total ordering.
  This does not broaden the supported input domains or guarantee meaningful
  results for nonfinite data.
- Update README examples and documentation links, and add a bounded BCa
  interval coverage regression.

### Fixed

- `roc_curve`, `roc_auc`, `pr_curve` and `average_precision` treat tied
  scores as one threshold, matching scikit-learn. A pure tie now gives AUC
  0.5 (was 0), and partial ties no longer inflate AUC. `pr_curve` returns
  one point per distinct threshold.
- `permutation_test` reports `(b + 1) / (m + 1)`, so the p-value is never 0.
- `wilcoxon` applies the tie correction to the normal-approximation
  variance, matching `scipy.stats.wilcoxon(method="approx")`.

- Remove the absolute tolerance from R² constant-target detection. Small nonzero sums of squares use the usual ratio; zero target variance returns 1 only when the residual sum of squares is also zero. This does not remove floating-point underflow or overflow limits.
  ([source change](https://github.com/arclabs561/statskit/commit/9fab9c9b3dabf9b177a14f929a7210288ca436bf))

## [0.3.0] - 2026-09-11

### Added

- `conformal`: split-conformal rank selection for finite scalar scores, with
  exact rational or binary floating-point levels and an explicit unbounded
  threshold. Score construction and prediction sets remain with callers.

### Fixed

- Return R² of 1.0 for effectively constant targets with negligible residual
  error, and 0.0 otherwise.
- Reject invalid labels, probabilities, p-values and statistical-test inputs
  instead of silently computing results. Validate binning, resampling and
  permutation counts; preserve NaN in variance calculations.

## [0.2.0] - 2026-07-03

### Fixed
- `aso` now computes the ASO statistic Dror et al. (2019) actually define:
  the W2 violation ratio (violation mass over total squared
  quantile-difference mass) with a bootstrap upper-confidence margin,
  mirroring the reference implementation (deep-significance) exactly. The
  previous implementation returned the bootstrap mean of a one-sided
  Kolmogorov-Smirnov-style CDF-difference supremum, a different statistic
  that happened to share the [0, 1] range and the under-dominance
  behavior. Values change for effectively all inputs; the deterministic
  point statistic is exposed as `aso_violation_ratio` and pinned against
  deepsig-generated fixtures.
- `mann_whitney` now applies the continuity correction and the tie
  correction that scipy's asymptotic method applies by default; p-values
  previously overstated significance on tied data and small samples.
  Pinned against scipy-generated fixtures (`method="asymptotic"`).

## [0.1.1] - 2026-06-11

### Added

- SIMD-accelerated MSE, MAE, and R² via `innr::dense_f64` behind a `simd` feature.

### Changed

- Updated the `innr` dependency to 0.6.

## [0.1.0] - 2026-04-15

### Added

- Classification metrics: log loss, balanced accuracy, Cohen kappa, specificity, Hamming, Jaccard, and MCE.
- Statistical tests: McNemar, Friedman, and Mann-Whitney.
- Regression metrics and a `classification_eval` example.
- Calibration, bootstrap, and system-comparison utilities.

### Fixed

- Used Welford's algorithm for variance computation.

[Unreleased]: https://github.com/arclabs561/statskit/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/arclabs561/statskit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/arclabs561/statskit/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/arclabs561/statskit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/arclabs561/statskit/releases/tag/v0.1.0
