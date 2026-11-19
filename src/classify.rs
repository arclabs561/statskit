use std::collections::BTreeMap;

/// Averaging strategy for multi-class metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Average {
    /// Global TP/FP/FN across all classes.
    Micro,
    /// Per-class metric, then unweighted mean over the labels that occur
    /// in `y_true` or `y_pred`.
    Macro,
    /// Per-class metric, weighted by support (true count).
    Weighted,
}

/// Per-class precision, recall, F1, and support.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PerClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// Full classification report across all classes.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClassificationReport {
    pub per_class: Vec<PerClassMetrics>,
    pub accuracy: f64,
    pub macro_avg: PerClassMetrics,
    pub weighted_avg: PerClassMetrics,
}

fn assert_binary_labels(labels: &[usize]) {
    assert!(
        labels.iter().all(|&label| label <= 1),
        "labels must be binary"
    );
}

fn assert_finite_scores(scores: &[f64]) {
    assert!(
        scores.iter().all(|score| score.is_finite()),
        "scores must be finite"
    );
}

fn assert_probabilities(probabilities: &[f64]) {
    assert!(
        probabilities.iter().all(|p| p.is_finite()),
        "probabilities must be finite"
    );
    assert!(
        probabilities.iter().all(|p| (0.0..=1.0).contains(p)),
        "probabilities must be in [0, 1]"
    );
}

/// Confusion matrix. `cm[true_class][pred_class]` = count.
pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize], n_classes: usize) -> Vec<Vec<u64>> {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    assert!(n_classes > 0, "n_classes must be positive");
    assert!(
        y_true.iter().chain(y_pred).all(|&label| label < n_classes),
        "label out of range"
    );
    let mut cm = vec![vec![0u64; n_classes]; n_classes];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        cm[t][p] += 1;
    }
    cm
}

#[derive(Clone, Copy, Default)]
struct ClassCounts {
    tp: u64,
    fp: u64,
    missed: u64,
}

impl ClassCounts {
    fn support(self) -> u64 {
        self.tp + self.missed
    }

    fn predicted(self) -> u64 {
        self.tp + self.fp
    }
}

fn observed_classes(y_true: &[usize], y_pred: &[usize]) -> BTreeMap<usize, ClassCounts> {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    let mut classes = BTreeMap::<usize, ClassCounts>::new();
    for (&actual, &predicted) in y_true.iter().zip(y_pred) {
        if actual == predicted {
            classes.entry(actual).or_default().tp += 1;
        } else {
            classes.entry(actual).or_default().missed += 1;
            classes.entry(predicted).or_default().fp += 1;
        }
    }
    classes
}

/// Precision with the given averaging strategy.
pub fn precision(y_true: &[usize], y_pred: &[usize], average: Average) -> f64 {
    compute_averaged(y_true, y_pred, average, |tp, fp, _fn| {
        if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        }
    })
}

/// Recall with the given averaging strategy.
pub fn recall(y_true: &[usize], y_pred: &[usize], average: Average) -> f64 {
    compute_averaged(y_true, y_pred, average, |tp, _fp, fn_| {
        if tp + fn_ == 0 {
            0.0
        } else {
            tp as f64 / (tp + fn_) as f64
        }
    })
}

/// F1 score (harmonic mean of precision and recall).
pub fn f1(y_true: &[usize], y_pred: &[usize], average: Average) -> f64 {
    fbeta(y_true, y_pred, 1.0, average)
}

/// F-beta score. `beta > 1` weights recall higher; `beta < 1` weights precision higher.
pub fn fbeta(y_true: &[usize], y_pred: &[usize], beta: f64, average: Average) -> f64 {
    let beta2 = beta * beta;
    compute_averaged(y_true, y_pred, average, |tp, fp, fn_| {
        let num = (1.0 + beta2) * tp as f64;
        let den = (1.0 + beta2) * tp as f64 + beta2 * fn_ as f64 + fp as f64;
        if den == 0.0 { 0.0 } else { num / den }
    })
}

/// Overall accuracy: correct predictions / total.
pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    if y_true.is_empty() {
        return 0.0;
    }
    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Matthews Correlation Coefficient (multiclass generalization).
///
/// Returns 0.0 when any denominator term is zero.
pub fn mcc(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let classes = observed_classes(y_true, y_pred);
    let n = y_true.len() as f64;
    let mut trace = 0.0;
    let mut pt_sum = 0.0;
    let mut p2_sum = 0.0;
    let mut t2_sum = 0.0;
    for counts in classes.values() {
        let predicted = counts.predicted() as f64;
        let actual = counts.support() as f64;
        trace += counts.tp as f64;
        pt_sum += predicted * actual;
        p2_sum += predicted * predicted;
        t2_sum += actual * actual;
    }

    let numerator = n * trace - pt_sum;
    let denom_a = n * n - p2_sum;
    let denom_b = n * n - t2_sum;

    if denom_a <= 0.0 || denom_b <= 0.0 {
        return 0.0;
    }
    numerator / (denom_a * denom_b).sqrt()
}

/// ROC curve points as `(fpr, tpr, threshold)`, sorted by decreasing threshold.
///
/// Binary classification only (class 0 vs 1).
pub fn roc_curve(y_true: &[usize], y_scores: &[f64]) -> Vec<(f64, f64, f64)> {
    assert_eq!(y_true.len(), y_scores.len(), "length mismatch");
    assert_binary_labels(y_true);
    assert_finite_scores(y_scores);
    let total_pos = y_true.iter().filter(|&&y| y == 1).count() as f64;
    let total_neg = y_true.iter().filter(|&&y| y == 0).count() as f64;

    if total_pos == 0.0 || total_neg == 0.0 {
        return vec![(0.0, 0.0, f64::INFINITY), (1.0, 1.0, f64::NEG_INFINITY)];
    }

    let mut points = vec![(0.0, 0.0, f64::INFINITY)];
    for (threshold, tp, fp) in cumulative_counts_by_threshold(y_true, y_scores) {
        points.push((fp / total_neg, tp / total_pos, threshold));
    }
    points
}

/// Cumulative `(threshold, tp, fp)` at each distinct score, in decreasing
/// score order. Tied scores are one threshold: a tie group yields a single
/// entry carrying the counts of the whole group, so the result does not
/// depend on input order within ties.
fn cumulative_counts_by_threshold(y_true: &[usize], y_scores: &[f64]) -> Vec<(f64, f64, f64)> {
    let mut indices: Vec<usize> = (0..y_scores.len()).collect();
    indices.sort_by(|&a, &b| y_scores[b].total_cmp(&y_scores[a]));

    let mut out = Vec::new();
    let mut tp = 0.0;
    let mut fp = 0.0;
    for (k, &i) in indices.iter().enumerate() {
        if y_true[i] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let group_ends = indices
            .get(k + 1)
            .is_none_or(|&next| y_scores[next] != y_scores[i]);
        if group_ends {
            out.push((y_scores[i], tp, fp));
        }
    }
    out
}

/// Area under the ROC curve (binary classification).
pub fn roc_auc(y_true: &[usize], y_scores: &[f64]) -> f64 {
    let curve = roc_curve(y_true, y_scores);
    trapezoidal_auc(
        &curve
            .iter()
            .map(|&(fpr, tpr, _)| (fpr, tpr))
            .collect::<Vec<_>>(),
    )
}

/// Precision-recall curve as `(precision, recall, threshold)`.
pub fn pr_curve(y_true: &[usize], y_scores: &[f64]) -> Vec<(f64, f64, f64)> {
    assert_eq!(y_true.len(), y_scores.len(), "length mismatch");
    assert_binary_labels(y_true);
    assert_finite_scores(y_scores);
    let total_pos = y_true.iter().filter(|&&y| y == 1).count() as f64;

    if total_pos == 0.0 {
        return vec![(0.0, 0.0, f64::INFINITY)];
    }

    cumulative_counts_by_threshold(y_true, y_scores)
        .into_iter()
        .map(|(threshold, tp, fp)| (tp / (tp + fp), tp / total_pos, threshold))
        .collect()
}

/// Average precision (area under PR curve).
pub fn average_precision(y_true: &[usize], y_scores: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_scores.len(), "length mismatch");
    assert_binary_labels(y_true);
    assert_finite_scores(y_scores);
    let total_pos = y_true.iter().filter(|&&y| y == 1).count() as f64;

    if total_pos == 0.0 {
        return 0.0;
    }

    let mut ap = 0.0;
    let mut prev_recall = 0.0;

    for (_, tp, fp) in cumulative_counts_by_threshold(y_true, y_scores) {
        let prec = tp / (tp + fp);
        let rec = tp / total_pos;
        // Trapezoidal integration over recall
        if rec > prev_recall {
            ap += prec * (rec - prev_recall);
            prev_recall = rec;
        }
    }

    ap
}

/// Full classification report with per-class and averaged metrics.
pub fn classification_report(
    y_true: &[usize],
    y_pred: &[usize],
    n_classes: usize,
) -> ClassificationReport {
    assert!(n_classes > 0, "n_classes must be positive");
    let classes = observed_classes(y_true, y_pred);
    assert!(
        classes.keys().next_back().is_none_or(|&c| c < n_classes),
        "label out of range"
    );

    let per_class: Vec<PerClassMetrics> = (0..n_classes)
        .map(|c| {
            let counts = classes.get(&c).copied().unwrap_or_default();
            let (tp, fp, fn_) = (counts.tp, counts.fp, counts.missed);
            let p = if tp + fp == 0 {
                0.0
            } else {
                tp as f64 / (tp + fp) as f64
            };
            let r = if tp + fn_ == 0 {
                0.0
            } else {
                tp as f64 / (tp + fn_) as f64
            };
            let f = if p + r == 0.0 {
                0.0
            } else {
                2.0 * p * r / (p + r)
            };
            PerClassMetrics {
                precision: p,
                recall: r,
                f1: f,
                support: counts.support() as usize,
            }
        })
        .collect();

    let acc = accuracy(y_true, y_pred);
    let total_support = y_true.len();

    let macro_avg = PerClassMetrics {
        precision: per_class.iter().map(|m| m.precision).sum::<f64>() / n_classes as f64,
        recall: per_class.iter().map(|m| m.recall).sum::<f64>() / n_classes as f64,
        f1: per_class.iter().map(|m| m.f1).sum::<f64>() / n_classes as f64,
        support: total_support,
    };

    let weighted_avg = if total_support == 0 {
        PerClassMetrics {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            support: 0,
        }
    } else {
        let ts = total_support as f64;
        PerClassMetrics {
            precision: per_class
                .iter()
                .map(|m| m.precision * m.support as f64)
                .sum::<f64>()
                / ts,
            recall: per_class
                .iter()
                .map(|m| m.recall * m.support as f64)
                .sum::<f64>()
                / ts,
            f1: per_class
                .iter()
                .map(|m| m.f1 * m.support as f64)
                .sum::<f64>()
                / ts,
            support: total_support,
        }
    };

    ClassificationReport {
        per_class,
        accuracy: acc,
        macro_avg,
        weighted_avg,
    }
}

/// Log loss (cross-entropy). Lower is better.
/// `y_true`: binary labels (0 or 1), `y_prob`: predicted probabilities.
/// Clips probabilities to `[1e-15, 1-1e-15]` to avoid `log(0)`.
pub fn log_loss(y_true: &[usize], y_prob: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_prob.len(), "length mismatch");
    assert_binary_labels(y_true);
    assert_probabilities(y_prob);
    if y_true.is_empty() {
        return 0.0;
    }
    let eps = 1e-15;
    let n = y_true.len() as f64;
    let sum: f64 = y_true
        .iter()
        .zip(y_prob.iter())
        .map(|(&y, &p)| {
            let p = p.clamp(eps, 1.0 - eps);
            let y = y as f64;
            -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        })
        .sum();
    sum / n
}

/// Balanced accuracy: average of per-class recall over the classes present
/// in `y_true`. Classes that occur only in `y_pred` have no recall and are
/// not averaged (matching scikit-learn's `balanced_accuracy_score`).
pub fn balanced_accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let classes = observed_classes(y_true, y_pred);
    let mut recalls = 0.0;
    let mut present = 0usize;
    for counts in classes.values().filter(|c| c.support() > 0) {
        recalls += counts.tp as f64 / counts.support() as f64;
        present += 1;
    }
    if present == 0 {
        return 0.0;
    }
    recalls / present as f64
}

/// Specificity (true negative rate) for binary classification.
pub fn specificity(y_true: &[usize], y_pred: &[usize]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    assert_binary_labels(y_true);
    assert_binary_labels(y_pred);
    if y_true.is_empty() {
        return 0.0;
    }
    let mut tn = 0u64;
    let mut fp = 0u64;
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if t == 0 && p == 0 {
            tn += 1;
        } else if t == 0 && p != 0 {
            fp += 1;
        }
    }
    if tn + fp == 0 {
        return 0.0;
    }
    tn as f64 / (tn + fp) as f64
}

/// Cohen's kappa coefficient. Measures agreement corrected for chance.
/// Range: -1 to 1. 1 = perfect, 0 = chance, negative = worse than chance.
pub fn cohen_kappa(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let classes = observed_classes(y_true, y_pred);
    if classes.is_empty() {
        return 0.0;
    }
    let n = y_true.len() as f64;
    let p_o = classes.values().map(|c| c.tp as f64).sum::<f64>() / n;
    let p_e: f64 = classes
        .values()
        .map(|c| (c.support() as f64 * c.predicted() as f64) / (n * n))
        .sum();

    if (1.0 - p_e).abs() < 1e-15 {
        return 0.0;
    }
    (p_o - p_e) / (1.0 - p_e)
}

/// Hamming loss: fraction of labels that are incorrectly predicted.
/// For single-label: equals `1 - accuracy`.
pub fn hamming_loss(y_true: &[usize], y_pred: &[usize]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    if y_true.is_empty() {
        return 0.0;
    }
    1.0 - accuracy(y_true, y_pred)
}

/// Jaccard similarity coefficient (IoU) with the given averaging strategy.
pub fn jaccard_score(y_true: &[usize], y_pred: &[usize], average: Average) -> f64 {
    compute_averaged(y_true, y_pred, average, |tp, fp, fn_| {
        let denom = tp + fp + fn_;
        if denom == 0 {
            0.0
        } else {
            tp as f64 / denom as f64
        }
    })
}

fn compute_averaged(
    y_true: &[usize],
    y_pred: &[usize],
    average: Average,
    metric_fn: impl Fn(u64, u64, u64) -> f64,
) -> f64 {
    let classes = observed_classes(y_true, y_pred);
    if classes.is_empty() {
        return 0.0;
    }
    match average {
        Average::Micro => {
            let correct = classes.values().map(|c| c.tp).sum();
            let incorrect = y_true.len() as u64 - correct;
            metric_fn(correct, incorrect, incorrect)
        }
        Average::Macro => {
            let total: f64 = classes
                .values()
                .map(|c| metric_fn(c.tp, c.fp, c.missed))
                .sum();
            total / classes.len() as f64
        }
        Average::Weighted => {
            let total: f64 = classes
                .values()
                .map(|c| metric_fn(c.tp, c.fp, c.missed) * c.support() as f64)
                .sum();
            total / y_true.len() as f64
        }
    }
}

fn trapezoidal_auc(points: &[(f64, f64)]) -> f64 {
    let mut area = 0.0;
    for i in 1..points.len() {
        let dx = points[i].0 - points[i - 1].0;
        let avg_y = (points[i].1 + points[i - 1].1) / 2.0;
        area += dx * avg_y;
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_binary() {
        // y_true: [0, 0, 1, 1], y_pred: [0, 1, 1, 0]
        // TN=1, FP=1, FN=1, TP=1
        let y_true = [0, 0, 1, 1];
        let y_pred = [0, 1, 1, 0];
        let cm = confusion_matrix(&y_true, &y_pred, 2);
        assert_eq!(cm[0][0], 1); // TN
        assert_eq!(cm[0][1], 1); // FP
        assert_eq!(cm[1][0], 1); // FN
        assert_eq!(cm[1][1], 1); // TP
    }

    #[test]
    fn test_confusion_matrix_multiclass() {
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 2, 1, 0, 2, 1];
        let cm = confusion_matrix(&y_true, &y_pred, 3);
        assert_eq!(cm[0][0], 1);
        assert_eq!(cm[0][2], 1);
        assert_eq!(cm[1][1], 1);
        assert_eq!(cm[1][0], 1);
        assert_eq!(cm[2][2], 1);
        assert_eq!(cm[2][1], 1);
    }

    #[test]
    #[should_panic(expected = "n_classes must be positive")]
    fn confusion_matrix_rejects_zero_classes() {
        confusion_matrix(&[], &[], 0);
    }

    #[test]
    #[should_panic(expected = "label out of range")]
    fn confusion_matrix_rejects_labels_outside_declared_classes() {
        confusion_matrix(&[2], &[0], 2);
    }

    #[test]
    #[should_panic(expected = "n_classes must be positive")]
    fn classification_report_rejects_zero_classes() {
        classification_report(&[], &[], 0);
    }

    #[test]
    fn test_precision_recall_f1_micro() {
        // For single-label classification, micro precision = micro recall = accuracy
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 0, 1, 2, 2, 1];
        let acc = accuracy(&y_true, &y_pred);
        let p = precision(&y_true, &y_pred, Average::Micro);
        let r = recall(&y_true, &y_pred, Average::Micro);
        let f = f1(&y_true, &y_pred, Average::Micro);
        assert!((p - acc).abs() < 1e-10);
        assert!((r - acc).abs() < 1e-10);
        assert!((f - acc).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_f1_macro() {
        // Hand-computed: y_true=[0,0,1,1], y_pred=[0,1,1,1]
        // Class 0: TP=1, FP=0, FN=1 -> P=1.0, R=0.5
        // Class 1: TP=2, FP=1, FN=0 -> P=2/3, R=1.0
        // Macro P = (1.0 + 2/3)/2 = 5/6
        // Macro R = (0.5 + 1.0)/2 = 0.75
        let y_true = [0, 0, 1, 1];
        let y_pred = [0, 1, 1, 1];
        let p = precision(&y_true, &y_pred, Average::Macro);
        let r = recall(&y_true, &y_pred, Average::Macro);
        assert!((p - 5.0 / 6.0).abs() < 1e-10);
        assert!((r - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_precision_recall_f1_weighted() {
        // Same data as macro test
        // Class 0 support=2, Class 1 support=2
        // Equal support -> weighted == macro
        let y_true = [0, 0, 1, 1];
        let y_pred = [0, 1, 1, 1];
        let pw = precision(&y_true, &y_pred, Average::Weighted);
        let pm = precision(&y_true, &y_pred, Average::Macro);
        assert!((pw - pm).abs() < 1e-10);

        // Unequal support: y_true=[0,1,1,1], y_pred=[0,1,1,0]
        // Class 0: TP=1, FP=1, FN=0 -> P=0.5, support=1
        // Class 1: TP=2, FP=0, FN=1 -> P=1.0, support=3
        // Weighted P = (0.5*1 + 1.0*3)/4 = 3.5/4 = 0.875
        let y_true2 = [0, 1, 1, 1];
        let y_pred2 = [0, 1, 1, 0];
        let pw2 = precision(&y_true2, &y_pred2, Average::Weighted);
        assert!((pw2 - 0.875).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_binary() {
        // TP=3, TN=2, FP=1, FN=1
        // MCC = (3*2 - 1*1) / sqrt((3+1)(3+1)(2+1)(2+1)) = 5/sqrt(144) = 5/12
        let y_true = [1, 1, 1, 0, 0, 0, 1];
        let y_pred = [1, 1, 1, 0, 0, 1, 0];
        let m = mcc(&y_true, &y_pred);
        assert!((m - 5.0 / 12.0).abs() < 1e-10, "got {m}");
    }

    #[test]
    fn test_mcc_perfect() {
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 0, 1, 1, 2, 2];
        let m = mcc(&y_true, &y_pred);
        assert!((m - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mcc_random() {
        // All predicted as class 0 -> MCC = 0
        let y_true = [0, 1, 2, 0, 1, 2];
        let y_pred = [0, 0, 0, 0, 0, 0];
        let m = mcc(&y_true, &y_pred);
        assert!(m.abs() < 1e-10, "expected ~0.0, got {m}");
    }

    #[test]
    fn test_roc_auc_perfect() {
        let y_true = [0, 0, 1, 1];
        let y_scores = [0.1, 0.2, 0.9, 0.95];
        let auc = roc_auc(&y_true, &y_scores);
        assert!((auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_roc_auc_random() {
        // Inverse ranking: every positive scored below every negative -> AUC = 0
        let y_true = [0, 0, 1, 1];
        let y_scores = [0.9, 0.8, 0.2, 0.1];
        let auc = roc_auc(&y_true, &y_scores);
        assert!(auc < 0.01, "expected ~0.0, got {auc}");
    }

    #[test]
    fn test_average_precision() {
        // Perfect ranking
        let y_true = [0, 0, 1, 1];
        let y_scores = [0.1, 0.2, 0.8, 0.9];
        let ap = average_precision(&y_true, &y_scores);
        assert!((ap - 1.0).abs() < 1e-10);
    }

    // Tied scores form one threshold: the ROC/PR point after a tie group uses
    // the cumulative counts of the whole group. Expected values are
    // sklearn's roc_auc_score / average_precision_score on the same inputs.

    #[test]
    fn roc_auc_pure_tie_is_half() {
        for y_true in [[0, 1], [1, 0]] {
            assert!((roc_auc(&y_true, &[0.5, 0.5]) - 0.5).abs() < 1e-12);
            assert!((average_precision(&y_true, &[0.5, 0.5]) - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn roc_auc_partial_tie_gives_half_credit() {
        let y_true = [0, 1, 0, 1];
        let y_scores = [1.0, 2.0, 2.0, 3.0];
        assert!((roc_auc(&y_true, &y_scores) - 0.875).abs() < 1e-12);
        assert!((average_precision(&y_true, &y_scores) - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn ranking_metrics_ignore_order_within_ties() {
        let y_true = [1, 0, 0, 1, 1, 0];
        let y_scores = [0.7, 0.7, 0.3, 0.3, 0.9, 0.1];
        let y_true_swapped = [0, 1, 1, 0, 1, 0];
        let y_scores_swapped = [0.7, 0.7, 0.3, 0.3, 0.9, 0.1];
        assert_eq!(
            roc_auc(&y_true, &y_scores),
            roc_auc(&y_true_swapped, &y_scores_swapped)
        );
        assert_eq!(
            average_precision(&y_true, &y_scores),
            average_precision(&y_true_swapped, &y_scores_swapped)
        );
        // One curve point per distinct threshold, plus the ROC origin.
        assert_eq!(roc_curve(&y_true, &y_scores).len(), 5);
        assert_eq!(pr_curve(&y_true, &y_scores).len(), 4);
    }

    #[test]
    #[should_panic(expected = "labels must be binary")]
    fn binary_ranking_metrics_reject_non_binary_labels() {
        roc_curve(&[2], &[0.5]);
    }

    #[test]
    #[should_panic(expected = "scores must be finite")]
    fn binary_ranking_metrics_reject_non_finite_scores() {
        average_precision(&[1], &[f64::INFINITY]);
    }

    #[test]
    fn test_all_same_class() {
        let y_true = [0, 0, 0];
        let y_pred = [0, 0, 0];
        let acc = accuracy(&y_true, &y_pred);
        assert!((acc - 1.0).abs() < 1e-10);

        // MCC is 0 when all same class (no variation)
        let m = mcc(&y_true, &y_pred);
        assert_eq!(m, 0.0);
    }

    macro_rules! empty_left_length_mismatch_test {
        ($name:ident, $call:expr) => {
            #[test]
            #[should_panic(expected = "length mismatch")]
            fn $name() {
                $call;
            }
        };
    }

    empty_left_length_mismatch_test!(precision_rejects_empty_left_mismatch, {
        precision(&[], &[0], Average::Macro)
    });
    empty_left_length_mismatch_test!(recall_rejects_empty_left_mismatch, {
        recall(&[], &[0], Average::Macro)
    });
    empty_left_length_mismatch_test!(f1_rejects_empty_left_mismatch, {
        f1(&[], &[0], Average::Macro)
    });
    empty_left_length_mismatch_test!(fbeta_rejects_empty_left_mismatch, {
        fbeta(&[], &[0], 2.0, Average::Macro)
    });
    empty_left_length_mismatch_test!(accuracy_rejects_empty_left_mismatch, {
        accuracy(&[], &[0])
    });
    empty_left_length_mismatch_test!(mcc_rejects_empty_left_mismatch, { mcc(&[], &[0]) });
    empty_left_length_mismatch_test!(balanced_accuracy_rejects_empty_left_mismatch, {
        balanced_accuracy(&[], &[0])
    });
    empty_left_length_mismatch_test!(specificity_rejects_empty_left_mismatch, {
        specificity(&[], &[0])
    });
    empty_left_length_mismatch_test!(cohen_kappa_rejects_empty_left_mismatch, {
        cohen_kappa(&[], &[0])
    });
    empty_left_length_mismatch_test!(hamming_loss_rejects_empty_left_mismatch, {
        hamming_loss(&[], &[0])
    });
    empty_left_length_mismatch_test!(jaccard_rejects_empty_left_mismatch, {
        jaccard_score(&[], &[0], Average::Macro)
    });

    #[test]
    fn test_log_loss_perfect() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.0, 0.0, 1.0, 1.0];
        let ll = log_loss(&y_true, &y_prob);
        assert!(ll < 1e-10, "expected near 0, got {ll}");
    }

    #[test]
    fn test_log_loss_worst() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.99, 0.99, 0.01, 0.01];
        let ll = log_loss(&y_true, &y_prob);
        assert!(ll > 2.0, "expected high loss, got {ll}");
    }

    #[test]
    fn test_log_loss_uniform() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.5, 0.5, 0.5, 0.5];
        let ll = log_loss(&y_true, &y_prob);
        assert!(
            (ll - 2.0_f64.ln()).abs() < 1e-10,
            "expected ln(2), got {ll}"
        );
    }

    #[test]
    #[should_panic(expected = "probabilities must be finite")]
    fn log_loss_rejects_non_finite_probabilities() {
        log_loss(&[1], &[f64::NAN]);
    }

    #[test]
    #[should_panic(expected = "probabilities must be in [0, 1]")]
    fn log_loss_rejects_out_of_range_probabilities() {
        log_loss(&[1], &[-0.1]);
    }

    #[test]
    fn test_balanced_accuracy_imbalanced() {
        // 90 negatives all correct, 10 positives all wrong
        let mut y_true = vec![0usize; 90];
        y_true.extend(vec![1usize; 10]);
        let mut y_pred = vec![0usize; 90];
        y_pred.extend(vec![0usize; 10]);
        // Class 0 recall = 1.0, Class 1 recall = 0.0 -> balanced = 0.5
        let ba = balanced_accuracy(&y_true, &y_pred);
        assert!((ba - 0.5).abs() < 1e-10, "expected 0.5, got {ba}");
    }

    /// Balanced accuracy is the mean recall over classes present in
    /// `y_true`. Class 2 appears only in `y_pred`, so the mean is over
    /// classes 0 and 1: (1 + 1/2) / 2 = 0.75
    /// (sklearn.metrics.balanced_accuracy_score).
    #[test]
    fn balanced_accuracy_ignores_prediction_only_classes() {
        let y_true = [0usize, 0, 1, 1];
        let y_pred = [0usize, 0, 1, 2];
        let ba = balanced_accuracy(&y_true, &y_pred);
        assert!((ba - 0.75).abs() < 1e-12, "balanced accuracy = {ba}");
    }

    /// Macro averages run over labels that occur in `y_true` or `y_pred`.
    /// Label 1 occurs in neither slice, so a perfect prediction on {0, 2}
    /// scores 1.0 (sklearn `average="macro"`).
    #[test]
    fn macro_average_ignores_labels_absent_from_both_slices() {
        let y_true = [0usize, 2, 2, 0];
        let y_pred = [0usize, 2, 2, 0];
        for (name, v) in [
            ("precision", precision(&y_true, &y_pred, Average::Macro)),
            ("recall", recall(&y_true, &y_pred, Average::Macro)),
            ("f1", f1(&y_true, &y_pred, Average::Macro)),
            ("jaccard", jaccard_score(&y_true, &y_pred, Average::Macro)),
        ] {
            assert!((v - 1.0).abs() < 1e-12, "macro {name} = {v}");
        }
        // Label 1 predicted but never true still counts (precision 0 for it):
        // per-class precision {1, 0, 1} -> 2/3.
        let p = precision(&[0usize, 2, 2], &[0usize, 2, 1], Average::Macro);
        assert!((p - 2.0 / 3.0).abs() < 1e-12, "macro precision = {p}");
    }

    #[test]
    fn aggregate_metrics_accept_the_full_label_range() {
        let dense_true = [0, 0, 1, 2];
        let dense_pred = [0, 1, 1, 3];
        let labels = [usize::MAX, 8, 0, 42];
        let sparse_true = dense_true.map(|i| labels[i]);
        let sparse_pred = dense_pred.map(|i| labels[i]);
        for average in [Average::Micro, Average::Macro, Average::Weighted] {
            for metric in [precision, recall, f1, jaccard_score] {
                let expected = metric(&dense_true, &dense_pred, average);
                let actual = metric(&sparse_true, &sparse_pred, average);
                assert!((actual - expected).abs() < 1e-12);
            }
            let expected = fbeta(&dense_true, &dense_pred, 2.0, average);
            let actual = fbeta(&sparse_true, &sparse_pred, 2.0, average);
            assert!((actual - expected).abs() < 1e-12);
        }
        assert!((mcc(&sparse_true, &sparse_pred) - 0.4).abs() < 1e-12);
        assert!((cohen_kappa(&sparse_true, &sparse_pred) - 1.0 / 3.0).abs() < 1e-12);
        assert!((balanced_accuracy(&sparse_true, &sparse_pred) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn report_includes_declared_but_unobserved_classes() {
        let report = classification_report(&[0, 2], &[0, 2], 4);
        assert_eq!(report.per_class.len(), 4);
        assert_eq!(report.per_class[1].support, 0);
        assert_eq!(report.per_class[3].support, 0);
        assert_eq!(report.macro_avg.f1, 0.5);
        assert_eq!(report.weighted_avg.f1, 1.0);
    }

    #[test]
    fn test_specificity_binary() {
        // TN=2, FP=1, FN=1, TP=2
        let y_true = [0, 0, 0, 1, 1, 1];
        let y_pred = [0, 0, 1, 1, 1, 0];
        // Specificity = TN/(TN+FP) = 2/3
        let s = specificity(&y_true, &y_pred);
        assert!((s - 2.0 / 3.0).abs() < 1e-10, "expected 2/3, got {s}");
    }

    #[test]
    #[should_panic(expected = "labels must be binary")]
    fn specificity_rejects_non_binary_labels() {
        specificity(&[2], &[0]);
    }

    #[test]
    fn test_cohen_kappa_perfect() {
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 0, 1, 1, 2, 2];
        let k = cohen_kappa(&y_true, &y_pred);
        assert!((k - 1.0).abs() < 1e-10, "expected 1.0, got {k}");
    }

    #[test]
    fn test_cohen_kappa_chance() {
        // All predicted as class 0 with balanced classes -> kappa near 0
        let y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        let y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 2];
        let k = cohen_kappa(&y_true, &y_pred);
        assert!(k.abs() < 0.1, "expected near 0, got {k}");
    }

    #[test]
    fn test_hamming_loss_equals_1_minus_accuracy() {
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 1, 1, 2, 2, 0];
        let hl = hamming_loss(&y_true, &y_pred);
        let acc = accuracy(&y_true, &y_pred);
        assert!((hl - (1.0 - acc)).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_score_binary() {
        // Class 0: TP=1, FP=1, FN=1 -> IoU = 1/3
        // Class 1: TP=1, FP=1, FN=1 -> IoU = 1/3
        // Macro = 1/3
        let y_true = [0, 0, 1, 1];
        let y_pred = [0, 1, 1, 0];
        let j = jaccard_score(&y_true, &y_pred, Average::Macro);
        assert!((j - 1.0 / 3.0).abs() < 1e-10, "expected 1/3, got {j}");
    }

    #[test]
    fn test_classification_report() {
        let y_true = [0, 0, 1, 1, 2, 2];
        let y_pred = [0, 0, 1, 2, 2, 2];
        let report = classification_report(&y_true, &y_pred, 3);

        assert_eq!(report.per_class.len(), 3);
        assert_eq!(report.per_class[0].support, 2);
        assert_eq!(report.per_class[1].support, 2);
        assert_eq!(report.per_class[2].support, 2);
        assert!((report.accuracy - 5.0 / 6.0).abs() < 1e-10);
        assert_eq!(report.macro_avg.support, 6);
        assert_eq!(report.weighted_avg.support, 6);
    }
}
