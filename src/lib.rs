//! # weigh
//!
//! Tekne L4: Statistical Judgment and Evaluation.
//!
//! This layer provides the statistical mirror for the stack, verifying performance
//! through rigorous metrics and confidence intervals.

pub mod metrics;
pub mod stats;

pub use metrics::*;
pub use stats::*;
