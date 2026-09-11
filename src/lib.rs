//! # statskit
//!
//! Statistical metrics and tests for plain `f64` and `usize` slices.
//!
//! The crate covers classification, probability calibration, regression,
//! descriptive statistics, system-comparison tests, and score-only conformal
//! threshold selection. Functions that make a statistical claim state their
//! assumptions in rustdoc.
//!
//! ## Contract
//!
//! - **No metric without a use case**: add a metric only when it is used downstream and has tests.
//! - **Uncertainty is explicit when present**: when a function makes a statistical claim (CI,
//!   p-value), assumptions must be spelled out in the rustdoc.
//! - **Small surface**: prefer a narrow set of well-specified primitives over a grab-bag.
//!
//! ## What's here
//!
//! - `stats`: descriptive statistics (mean, variance, stddev) and statistical tests
//!   (bootstrap BCa, Wilcoxon, permutation, ASO, multiple-comparison corrections, effect sizes)
//! - `classify`: classification metrics (precision, recall, F1, MCC, ROC-AUC, PR-AUC,
//!   confusion matrix, classification report, log loss, balanced accuracy, specificity,
//!   Cohen's kappa, Hamming loss, Jaccard score)
//! - `calibration`: calibration metrics (Brier score, ECE, MCE, reliability diagram)
//! - `conformal`: finite-sample score thresholds for split conformal calibration
//! - `regression`: regression metrics (MSE, RMSE, MAE, R-squared)

#![forbid(unsafe_code)]

pub mod calibration;
pub mod classify;
pub mod conformal;
pub mod regression;
pub mod stats;

pub use calibration::*;
pub use classify::*;
pub use regression::*;
pub use stats::*;
