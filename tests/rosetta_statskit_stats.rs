//! Rosetta correctness fixtures: statskit regression metrics and descriptive
//! statistics asserted against scikit-learn and numpy. Companion to
//! rosetta_statskit.rs (classification/calibration).
//!
//! Reference values in `fixtures/rosetta/statskit_stats.json` come from
//! `gen_statskit_stats.py` (their provenance). All EXACT library oracles:
//! regression mse/rmse/mae/r_squared vs sklearn.metrics; descriptive
//! mean/variance/stddev (population ddof=0, sample ddof=1) vs numpy.
//!
//! Deferred: the statistical tests in stats.rs (wilcoxon, mann_whitney,
//! cohens_d, friedman, mcnemar), whose p-value / pooled-std / continuity
//! conventions must be matched to scipy before a fixture is safe.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_statskit_stats.py`.

use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/statskit_stats.json");

#[derive(Deserialize)]
struct Fixture {
    xs: Vec<f64>,
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    mean: f64,
    variance_population: f64,
    variance_sample: f64,
    stddev_population: f64,
    stddev_sample: f64,
    mse: f64,
    rmse: f64,
    mae: f64,
    r_squared: f64,
}

fn close(got: f64, want: f64, label: &str) {
    let tol = 1e-9 * (1.0 + want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label}: statskit={got} ref={want} diff={diff} tol={tol}"
    );
}

#[test]
fn rosetta_stats_match_sklearn_numpy() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let xs = &fx.xs;
    let e = &fx.expected;

    // Descriptive statistics (numpy, ddof 0 = population, ddof 1 = sample).
    close(statskit::mean(xs).unwrap(), e.mean, "mean");
    close(
        statskit::variance_population(xs).unwrap(),
        e.variance_population,
        "variance_population",
    );
    close(
        statskit::variance_sample(xs).unwrap(),
        e.variance_sample,
        "variance_sample",
    );
    close(
        statskit::stddev_population(xs).unwrap(),
        e.stddev_population,
        "stddev_population",
    );
    close(
        statskit::stddev_sample(xs).unwrap(),
        e.stddev_sample,
        "stddev_sample",
    );

    // Regression metrics (sklearn.metrics).
    close(statskit::mse(&fx.y_true, &fx.y_pred), e.mse, "mse");
    close(statskit::rmse(&fx.y_true, &fx.y_pred), e.rmse, "rmse");
    close(statskit::mae(&fx.y_true, &fx.y_pred), e.mae, "mae");
    close(
        statskit::r_squared(&fx.y_true, &fx.y_pred),
        e.r_squared,
        "r_squared",
    );
}
