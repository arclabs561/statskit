//! # statskit
//!
//! Statistical judgment and evaluation.
//!
//! This layer is the statistical mirror for the stack: turn "it seems better" into
//! "it is better, under a stated metric, with assumptions stated."
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
//! - `regression`: regression metrics (MSE, RMSE, MAE, R-squared)

#![forbid(unsafe_code)]

pub mod calibration;
pub mod classify;
pub mod regression;
pub mod stats;

pub use calibration::*;
pub use classify::*;
pub use regression::*;
pub use stats::*;
