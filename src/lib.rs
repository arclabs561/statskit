//! # judge
//!
//! Tekne L4: Unified Evaluation and Statistical Judgment.
//!
//! This crate provides the mathematical tools for passing judgment on 
//! representations. It sits between L3 (Structures) and L5 (Domain).
//!
//! ## Key Responsibilities
//!
//! - **Statistical Rigor**: Computing moments, intervals, and significance.
//! - **Conformal Prediction**: Providing guaranteed error bounds for ML outputs.
//! - **Calibration**: Ensuring that confidence scores match empirical accuracy.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("statistical error: {0}")]
    Statistics(String),
    #[error("calibration error: {0}")]
    Calibration(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// A Judgment is a value paired with its statistical justification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Judgment<T> {
    /// The value being judged.
    pub value: T,
    /// The confidence score in [0, 1].
    pub confidence: f64,
    /// The statistical interval (e.g. for conformal prediction).
    pub interval: Option<(f64, f64)>,
}

impl<T> Judgment<T> {
    /// Create a new judgment with a confidence score.
    pub fn new(value: T, confidence: f64) -> Self {
        Self {
            value,
            confidence,
            interval: None,
        }
    }

    /// Add a conformal interval to the judgment.
    pub fn with_interval(mut self, low: f64, high: f64) -> Self {
        self.interval = Some((low, high));
        self
    }
}

/// Statistics for high-dimensional embedding drift.
pub mod stats {
    /// Compute the drift (Euclidean distance between means) of two samples.
    pub fn mean_drift(a: &[&[f32]], b: &[&[f32]]) -> f64 {
        if a.is_empty() || b.is_empty() { return 0.0; }
        let dim = a[0].len();
        
        let mut mean_a = vec![0.0; dim];
        for v in a {
            for (i, &x) in v.iter().enumerate() { mean_a[i] += x; }
        }
        for x in &mut mean_a { *x /= a.len() as f32; }

        let mut mean_b = vec![0.0; dim];
        for v in b {
            for (i, &x) in v.iter().enumerate() { mean_b[i] += x; }
        }
        for x in &mut mean_b { *x /= b.len() as f32; }

        let mut sum_sq = 0.0;
        for i in 0..dim {
            sum_sq += (mean_a[i] - mean_b[i]).powi(2);
        }
        sum_sq.sqrt() as f64
    }
}

/// Conformal Prediction implementation.
pub mod conformal {
    /// Scores for calibration.
    pub struct CalibrationSet {
        pub non_conformity_scores: Vec<f64>,
    }

    impl CalibrationSet {
        pub fn new(mut scores: Vec<f64>) -> Self {
            scores.sort_by(|a, b| a.total_cmp(b));
            Self { non_conformity_scores: scores }
        }

        /// Returns the threshold for a given significance level alpha.
        pub fn quantile(&self, alpha: f64) -> f64 {
            let n = self.non_conformity_scores.len();
            let q = ((n + 1) as f64 * (1.0 - alpha)).ceil() as usize;
            self.non_conformity_scores[q.min(n - 1)]
        }
    }
}

/// Statistical Significance tests for judgment.
pub mod significance {
    /// Perform a simple Chi-Squared test for categorical distribution alignment.
    /// Returns the statistic.
    pub fn chi_squared(observed: &[f64], expected: &[f64]) -> f64 {
        observed.iter().zip(expected.iter())
            .map(|(&o, &e)| {
                if e == 0.0 { 0.0 } else { (o - e).powi(2) / e }
            })
            .sum()
    }
}
