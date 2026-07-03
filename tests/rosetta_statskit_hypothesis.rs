//! Rosetta correctness fixtures: statskit's hypothesis-test layer asserted
//! against the reference implementations. Companion to rosetta_statskit.rs
//! (classification/calibration) and rosetta_statskit_stats.rs
//! (regression/descriptive).
//!
//! Reference values in `fixtures/rosetta/statskit_hypothesis.json` come from
//! `gen_statskit_hypothesis.py` (their provenance):
//! - ASO violation ratio from deepsig's `compute_violation_ratio` (the Dror
//!   et al. 2019 reference implementation), deterministic given scores, so
//!   asserted TIGHT.
//! - Mann-Whitney U and two-sided p from scipy `mannwhitneyu` with
//!   `method="asymptotic"`, `use_continuity=True` (the convention statskit
//!   implements; scipy's auto method switches to exact on small tie-free
//!   samples, which statskit does not implement). p compared at 1e-6
//!   relative: statskit's erf is an Abramowitz-Stegun approximation.
//!
//! `aso` itself adds a seeded-bootstrap upper-confidence margin on the
//! ratio, so it is asserted as bounds (>= ratio, <= 1) plus side-of-0.5
//! agreement on the clearly ordered cases, not as an exact value (the
//! bootstrap RNG differs from numpy's).

use serde::Deserialize;
use statskit::{aso, aso_violation_ratio, mann_whitney};

const FIXTURE: &str = include_str!("fixtures/rosetta/statskit_hypothesis.json");

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    a: Vec<f64>,
    b: Vec<f64>,
    violation_ratio: f64,
    mannwhitney_u_a: f64,
    mannwhitney_p: f64,
}

#[test]
fn aso_violation_ratio_matches_deepsig() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("fixture parses");
    for case in &fixture.cases {
        let got = aso_violation_ratio(&case.a, &case.b);
        assert!(
            (got - case.violation_ratio).abs() < 1e-9,
            "{}: violation ratio diverges from deepsig: got {got}, want {}",
            case.name,
            case.violation_ratio
        );
    }
}

#[test]
fn aso_upper_bound_brackets_the_ratio() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("fixture parses");
    for case in &fixture.cases {
        let ratio = aso_violation_ratio(&case.a, &case.b);
        let eps_min = aso(&case.a, &case.b, 500, Some(13));
        assert!(
            (ratio..=1.0).contains(&eps_min),
            "{}: eps_min {eps_min} should lie in [ratio {ratio}, 1]",
            case.name
        );
        match case.name.as_str() {
            "a_dominates" => assert!(eps_min < 0.5, "{}: eps_min={eps_min}", case.name),
            "b_dominates" => assert!(eps_min > 0.5, "{}: eps_min={eps_min}", case.name),
            _ => {}
        }
    }
}

#[test]
fn mann_whitney_matches_scipy_asymptotic() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("fixture parses");
    for case in &fixture.cases {
        let r = mann_whitney(&case.a, &case.b, 0.05);
        // statskit reports min(U_a, U_b); scipy reports U_a. Equivalent
        // under U_a + U_b = n_a * n_b.
        let n_ab = (case.a.len() * case.b.len()) as f64;
        let scipy_u_min = case.mannwhitney_u_a.min(n_ab - case.mannwhitney_u_a);
        assert!(
            (r.statistic - scipy_u_min).abs() < 1e-9,
            "{}: U diverges: got {}, scipy min-U {scipy_u_min}",
            case.name,
            r.statistic
        );
        // statskit's normal_cdf uses the Abramowitz-Stegun 7.1.26 erf
        // approximation (|error| <= 1.5e-7 absolute), so p carries an
        // absolute error floor around 3e-7 plus a small relative term.
        // A missing continuity or tie correction shifts p by orders of
        // magnitude more than this at these values.
        let tol = 3e-7 + 1e-5 * case.mannwhitney_p;
        assert!(
            (r.p_value - case.mannwhitney_p).abs() < tol,
            "{}: p diverges: got {}, scipy {}",
            case.name,
            r.p_value,
            case.mannwhitney_p
        );
    }
}
