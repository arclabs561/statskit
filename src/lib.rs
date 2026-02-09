//! # statskit
//!
//! Statistical judgment and evaluation.
//!
//! This layer is the statistical mirror for the stack: turn “it seems better” into
//! “it is better, under a stated metric, with assumptions stated.”
//!
//! ## Contract
//!
//! - **No metric without a use case**: add a metric only when it is used downstream and has tests.
//! - **Uncertainty is explicit when present**: when a function makes a statistical claim (CI,
//!   p-value), assumptions must be spelled out in the rustdoc. (Note: this crate currently focuses
//!   on point-estimate primitives.)
//! - **Small surface**: prefer a narrow set of well-specified primitives over a grab-bag.
//!
//! ## What’s here (today)
//!
//! - `stats`: small helpers (mean, moments, etc.)
//! - `metrics`: evaluation metrics (currently minimal; grows with usage)

#![forbid(unsafe_code)]

pub mod metrics;
pub mod stats;

pub use metrics::*;
pub use stats::*;
