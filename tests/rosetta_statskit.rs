//! Rosetta correctness fixtures: statskit classification + calibration metrics
//! asserted against scikit-learn reference values.
//!
//! The reference values in `fixtures/rosetta/statskit_classify.json` are computed
//! by scikit-learn (see the sibling `gen_statskit.py`, which is their provenance).
//! This proves agreement with a specific, well-trusted reference implementation at
//! a specific parameterization; it does not prove absolute correctness, and it
//! deliberately covers only deterministic functions with a clean sklearn oracle.
//! ECE and MCE are excluded because scikit-learn ships no canonical reference.
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_statskit.py`.

use serde::Deserialize;
use statskit::Average;

const FIXTURE: &str = include_str!("fixtures/rosetta/statskit_classify.json");

#[derive(Deserialize)]
struct Fixture {
    datasets: Datasets,
    expected: Expected,
    confusion_multi: Vec<Vec<u64>>,
}

#[derive(Deserialize)]
struct Datasets {
    bin: BinSet,
    multi: MultiSet,
}

#[derive(Deserialize)]
struct BinSet {
    y_true: Vec<usize>,
    y_pred: Vec<usize>,
    y_score: Vec<f64>,
    y_prob: Vec<f64>,
}

#[derive(Deserialize)]
struct MultiSet {
    y_true: Vec<usize>,
    y_pred: Vec<usize>,
    n_classes: usize,
}

#[derive(Deserialize)]
struct Expected {
    accuracy_bin: f64,
    mcc_bin: f64,
    mcc_multi: f64,
    roc_auc_bin: f64,
    average_precision_bin: f64,
    log_loss_bin: f64,
    brier_bin: f64,
    cohen_kappa_multi: f64,
    balanced_accuracy_multi: f64,
    precision_micro_multi: f64,
    precision_macro_multi: f64,
    recall_macro_multi: f64,
    f1_micro_multi: f64,
    f1_macro_multi: f64,
    f1_weighted_multi: f64,
    jaccard_macro_multi: f64,
}

/// EXACT-class tolerance: these are deterministic functions, so agreement with
/// the sklearn reference should be at the floating-point-noise level. A loose
/// tolerance here would hide a real divergence, which is the whole point.
fn close(got: f64, want: f64, label: &str) {
    let tol = 1e-9 * (1.0 + want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label}: statskit={got} sklearn={want} diff={diff} tol={tol}"
    );
}

#[test]
fn rosetta_classify_matches_sklearn() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let b = &fx.datasets.bin;
    let m = &fx.datasets.multi;
    let e = &fx.expected;

    // Binary metrics.
    close(
        statskit::accuracy(&b.y_true, &b.y_pred),
        e.accuracy_bin,
        "accuracy_bin",
    );
    close(statskit::mcc(&b.y_true, &b.y_pred), e.mcc_bin, "mcc_bin");
    close(
        statskit::roc_auc(&b.y_true, &b.y_score),
        e.roc_auc_bin,
        "roc_auc_bin",
    );
    close(
        statskit::average_precision(&b.y_true, &b.y_score),
        e.average_precision_bin,
        "average_precision_bin",
    );
    close(
        statskit::log_loss(&b.y_true, &b.y_prob),
        e.log_loss_bin,
        "log_loss_bin",
    );
    close(
        statskit::brier_score(&b.y_true, &b.y_prob),
        e.brier_bin,
        "brier_bin",
    );

    // Multiclass metrics.
    close(
        statskit::mcc(&m.y_true, &m.y_pred),
        e.mcc_multi,
        "mcc_multi",
    );
    close(
        statskit::cohen_kappa(&m.y_true, &m.y_pred),
        e.cohen_kappa_multi,
        "cohen_kappa_multi",
    );
    close(
        statskit::balanced_accuracy(&m.y_true, &m.y_pred),
        e.balanced_accuracy_multi,
        "balanced_accuracy_multi",
    );
    close(
        statskit::precision(&m.y_true, &m.y_pred, Average::Micro),
        e.precision_micro_multi,
        "precision_micro_multi",
    );
    close(
        statskit::precision(&m.y_true, &m.y_pred, Average::Macro),
        e.precision_macro_multi,
        "precision_macro_multi",
    );
    close(
        statskit::recall(&m.y_true, &m.y_pred, Average::Macro),
        e.recall_macro_multi,
        "recall_macro_multi",
    );
    close(
        statskit::f1(&m.y_true, &m.y_pred, Average::Micro),
        e.f1_micro_multi,
        "f1_micro_multi",
    );
    close(
        statskit::f1(&m.y_true, &m.y_pred, Average::Macro),
        e.f1_macro_multi,
        "f1_macro_multi",
    );
    close(
        statskit::f1(&m.y_true, &m.y_pred, Average::Weighted),
        e.f1_weighted_multi,
        "f1_weighted_multi",
    );
    close(
        statskit::jaccard_score(&m.y_true, &m.y_pred, Average::Macro),
        e.jaccard_macro_multi,
        "jaccard_macro_multi",
    );

    // Confusion matrix is an exact integer match (cm[true][pred]).
    let cm = statskit::confusion_matrix(&m.y_true, &m.y_pred, m.n_classes);
    assert_eq!(cm, fx.confusion_multi, "confusion_multi");
}
