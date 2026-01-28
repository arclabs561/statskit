//! Metrics for statistical evaluation and judgement.
//!
//! This crate is intentionally small: `statskit` is meant to be the "statistical mirror"
//! of the stack, not a grab-bag. Add metrics only when they are used by at least one
//! downstream crate and have tests.

/// A trivial accuracy metric.
///
/// This is here mostly as a placeholder so the crate builds and has at least one
/// concrete API surface; it will likely move/expand as `weigh` is integrated.
pub fn accuracy(correct: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    correct as f64 / total as f64
}
