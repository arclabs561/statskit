# statskit

[![crates.io](https://img.shields.io/crates/v/statskit.svg)](https://crates.io/crates/statskit)
[![Documentation](https://docs.rs/statskit/badge.svg)](https://docs.rs/statskit)
[![CI](https://github.com/arclabs561/statskit/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/statskit/actions/workflows/ci.yml)

Statistical judgment and evaluation primitives.

## Quickstart

```toml
[dependencies]
statskit = "0.1"
```

```rust
use statskit::{accuracy, variance_population};

// accuracy takes parallel label slices and returns the fraction correct.
let acc = accuracy(&[0, 1, 1, 0], &[0, 1, 0, 0]);
assert_eq!(acc, 0.75);

let xs = [1.0, 2.0, 3.0, 4.0];
assert_eq!(variance_population(&xs).unwrap(), 1.25);
```

## Modules

- `statskit::classify`: Classification metrics (accuracy, precision, recall, F1, confusion matrix, log loss).
- `statskit::regression`: Regression metrics (MSE, MAE, R^2, and related).
- `statskit::calibration`: Probability-calibration metrics.
- `statskit::stats`: Basic moments and means (small helpers; numerically stable where practical).

## Status

- Experimental, intentionally small surface.
