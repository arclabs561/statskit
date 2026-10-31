# statskit

[![crates.io](https://img.shields.io/crates/v/statskit.svg)](https://crates.io/crates/statskit)
[![Documentation](https://docs.rs/statskit/badge.svg)](https://docs.rs/statskit)

Statistical metrics and tests.

`statskit` works on `f64` and `usize` slices.

## Quickstart

```toml
[dependencies]
statskit = "0.3"
```

```rust
use statskit::{
    accuracy, bootstrap_bca, cohens_d, f1, mean_diff, mse, wilcoxon, Average,
    BootstrapConfig,
};

// Classification: parallel integer label slices.
let y_true = [0usize, 1, 1, 0, 1];
let y_pred = [0usize, 1, 0, 0, 1];
let acc = accuracy(&y_true, &y_pred);
assert!((acc - 0.8).abs() < 1e-12);
let macro_f1 = f1(&y_true, &y_pred, Average::Macro);
assert!((macro_f1 - 0.8).abs() < 1e-12);

// Regression error.
let err = mse(&[1.0, 2.0, 3.0], &[1.1, 1.9, 3.2]);
assert!((err - 0.02).abs() < 1e-12);

// Compare systems by effect size and a bootstrap CI on their mean difference.
let a = [0.81, 0.79, 0.88, 0.90, 0.85];
let b = [0.74, 0.70, 0.80, 0.77, 0.69];
let d = cohens_d(&a, &b);
assert!(d > 0.0);
let ci = bootstrap_bca(&a, &b, mean_diff, BootstrapConfig::default());
assert!(ci.lower <= ci.point_estimate && ci.point_estimate <= ci.upper);

// With five pairs, take the p-value from an exact test: all five differences
// favour `a`, and the smallest two-sided p-value is 2 / 2^5 = 0.0625.
let w = wilcoxon(&a, &b, 0.05);
assert!((w.p_value - 0.0625).abs() < 1e-12);
```

`BootstrapResult::p_value` is unreliable for small samples; prefer
`permutation_test` or the exact `wilcoxon` for a p-value.

See the [API reference](https://docs.rs/statskit) for the full set of metrics and tests.

## Conformal thresholds

Scores can be signed; larger values mean worse agreement. Supply a fixed
scoring rule and calibration scores exchangeable with the future score.

```rust
use statskit::conformal::{calibrate_in_place, Coverage, Threshold};

let mut scores = [0.2, 0.8, 0.4, 1.2, 0.5, 0.3, 0.6, 0.9, 0.7];
let coverage = Coverage::from_ratio(9, 10).unwrap();
let threshold = calibrate_in_place(&mut scores, coverage).unwrap();
assert_eq!(threshold, Threshold::Finite(1.2));
```

The ratio requests exactly 90% coverage. `Coverage::from_miscoverage(alpha)`
instead uses the exact value represented by the supplied `f64`. Calibration
reorders the scores and returns `Threshold::Unbounded` if the required rank
exceeds their count. Empty or non-finite scores return an error.

The threshold supplies a marginal coverage guarantee under exchangeability.
Callers define scores and turn thresholds into prediction sets. Model fitting
and scale estimation remain in the calling code.

## Features

The default API and `serde` support Rust 1.85. The `simd` feature requires
Rust 1.89 or newer.

- `serde`: derive `Serialize` / `Deserialize` on result structs.
- `simd`: SIMD-accelerated `f64` reductions in the regression metrics, via `innr`.

## Status

Experimental, intentionally small surface. A metric is added only when it has a
downstream use case and tests; functions making a statistical claim (CI,
p-value) state their assumptions in the rustdoc.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or
the [MIT license](LICENSE-MIT), at your option.
