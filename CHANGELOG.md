# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[0.1.1]: https://github.com/arclabs561/statskit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/arclabs561/statskit/releases/tag/v0.1.0
