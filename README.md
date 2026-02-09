# statskit

Statistical judgment and evaluation primitives (metrics + small stats helpers).

## Quickstart

```toml
[dependencies]
# Not on crates.io yet; depend via git (pin `rev` for reproducibility).
statskit = { git = "https://github.com/arclabs561/statskit" }
```

```rust
use statskit::{accuracy, variance_population};

let acc = accuracy(2, 3);
assert_eq!(acc, 2.0 / 3.0);

let xs = [1.0, 2.0, 3.0, 4.0];
assert_eq!(variance_population(&xs).unwrap(), 1.25);
```

## Modules

- `statskit::metrics`: Evaluation metrics (intentionally small; grows with downstream usage).
- `statskit::stats`: Basic moments and means (small helpers; numerically stable where practical).

## Status

- Experimental, intentionally small surface.
- Not published on crates.io yet (`Cargo.toml` sets `publish = false`).
