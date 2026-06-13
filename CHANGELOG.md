# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
