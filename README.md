# statskit

Statistical judgment and evaluation: auditable metrics and uncertainty quantification.

## Quickstart

```toml
[dependencies]
statskit = "0.1.0"
```

```rust
use statskit::metrics::accuracy;

let acc = accuracy(2, 3);
assert_eq!(acc, 2.0 / 3.0);
```

## Modules

- `statskit::metrics`: Evaluation metrics (intentionally small; grows with downstream usage).
- `statskit::stats`: Basic moments and means (small helpers).
