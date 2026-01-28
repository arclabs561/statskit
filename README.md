# statskit

Statistical judgment and evaluation.

This crate exists to make evaluation claims **auditable**: define metrics precisely, compute them
correctly, and expose uncertainty when a result is inherently statistical.

## Best starting points

- **Module overview**: read the crate docs in `statskit/src/lib.rs` first.
- **Metrics**: `statskit::metrics::*` (kept small; grows only when used downstream)
- **Basic statistics**: `statskit::stats::*` (means/moments and related helpers)
