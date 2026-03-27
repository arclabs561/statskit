/// Averaging strategy for multi-class metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Average {
    /// Global TP/FP/FN across all classes.
    Micro,
    /// Per-class metric, then unweighted mean.
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

/// Confusion matrix. `cm[true_class][pred_class]` = count.
pub fn confusion_matrix(y_true: &[usize], y_pred: &[usize], n_classes: usize) -> Vec<Vec<u64>> {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    let mut cm = vec![vec![0u64; n_classes]; n_classes];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        cm[t][p] += 1;
    }
    cm
}

fn per_class_tp_fp_fn(cm: &[Vec<u64>], n_classes: usize) -> Vec<(u64, u64, u64)> {
    let mut result = vec![(0u64, 0u64, 0u64); n_classes];
    for c in 0..n_classes {
        let tp = cm[c][c];
        let fp: u64 = (0..n_classes)
            .map(|r| if r != c { cm[r][c] } else { 0 })
            .sum();
        let fn_: u64 = (0..n_classes)
            .map(|p| if p != c { cm[c][p] } else { 0 })
            .sum();
        result[c] = (tp, fp, fn_);
    }
    result
}

fn support_per_class(cm: &[Vec<u64>], n_classes: usize) -> Vec<usize> {
    (0..n_classes)
        .map(|c| cm[c].iter().sum::<u64>() as usize)
        .collect()
}

/// Precision with the given averaging strategy.
pub fn precision(y_true: &[usize], y_pred: &[usize], average: Average) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let n_classes = n_classes_from(y_true, y_pred);
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let stats = per_class_tp_fp_fn(&cm, n_classes);
    let supports = support_per_class(&cm, n_classes);
    compute_averaged(&stats, &supports, average, |tp, fp, _fn| {
        if tp + fp == 0 {
            0.0
        } else {
            tp as f64 / (tp + fp) as f64
        }
    })
}

/// Recall with the given averaging strategy.
pub fn recall(y_true: &[usize], y_pred: &[usize], average: Average) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let n_classes = n_classes_from(y_true, y_pred);
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let stats = per_class_tp_fp_fn(&cm, n_classes);
    let supports = support_per_class(&cm, n_classes);
    compute_averaged(&stats, &supports, average, |tp, _fp, fn_| {
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
    if y_true.is_empty() {
        return 0.0;
    }
    let n_classes = n_classes_from(y_true, y_pred);
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let stats = per_class_tp_fp_fn(&cm, n_classes);
    let supports = support_per_class(&cm, n_classes);
    let beta2 = beta * beta;
    compute_averaged(&stats, &supports, average, |tp, fp, fn_| {
        let num = (1.0 + beta2) * tp as f64;
        let den = (1.0 + beta2) * tp as f64 + beta2 * fn_ as f64 + fp as f64;
        if den == 0.0 { 0.0 } else { num / den }
    })
}

/// Overall accuracy: correct predictions / total.
pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
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
    if y_true.is_empty() {
        return 0.0;
    }
    let n_classes = n_classes_from(y_true, y_pred);
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let n = y_true.len() as f64;

    // Gorodkin (2004) multiclass MCC:
    // MCC = (n * sum_k(cm[k][k]) - sum_k(sum_i(cm[k][i]) * sum_i(cm[i][k])))
    //     / sqrt((n^2 - sum_k(sum_i(cm[i][k])^2)) * (n^2 - sum_k(sum_i(cm[k][i])^2)))

    let trace: f64 = (0..n_classes).map(|k| cm[k][k] as f64).sum();

    // p_k = sum over rows of column k (predicted as k)
    let p: Vec<f64> = (0..n_classes)
        .map(|k| (0..n_classes).map(|i| cm[i][k] as f64).sum())
        .collect();

    // t_k = sum over columns of row k (true class k)
    let t: Vec<f64> = (0..n_classes)
        .map(|k| (0..n_classes).map(|i| cm[k][i] as f64).sum())
        .collect();

    let pt_sum: f64 = (0..n_classes).map(|k| p[k] * t[k]).sum();
    let p2_sum: f64 = p.iter().map(|x| x * x).sum::<f64>();
    let t2_sum: f64 = t.iter().map(|x| x * x).sum::<f64>();

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
    let total_pos = y_true.iter().filter(|&&y| y == 1).count() as f64;
    let total_neg = y_true.iter().filter(|&&y| y == 0).count() as f64;

    if total_pos == 0.0 || total_neg == 0.0 {
        return vec![(0.0, 0.0, f64::INFINITY), (1.0, 1.0, f64::NEG_INFINITY)];
    }

    let mut indices: Vec<usize> = (0..y_scores.len()).collect();
    indices.sort_by(|&a, &b| y_scores[b].partial_cmp(&y_scores[a]).unwrap());

    let mut points = Vec::new();
    let mut tp = 0.0;
    let mut fp = 0.0;

    points.push((0.0, 0.0, f64::INFINITY));

    for &i in &indices {
        if y_true[i] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let fpr = fp / total_neg;
        let tpr = tp / total_pos;
        points.push((fpr, tpr, y_scores[i]));
    }

    // Deduplicate consecutive points with same threshold
    points.dedup_by(|a, b| (a.2 - b.2).abs() < f64::EPSILON);
    points
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
    let total_pos = y_true.iter().filter(|&&y| y == 1).count() as f64;

    if total_pos == 0.0 {
        return vec![(0.0, 0.0, f64::INFINITY)];
    }

    let mut indices: Vec<usize> = (0..y_scores.len()).collect();
    indices.sort_by(|&a, &b| y_scores[b].partial_cmp(&y_scores[a]).unwrap());

    let mut points = Vec::new();
    let mut tp = 0.0;
    let mut fp = 0.0;

    for &i in &indices {
        if y_true[i] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let prec = tp / (tp + fp);
        let rec = tp / total_pos;
        points.push((prec, rec, y_scores[i]));
    }

    points
}

/// Average precision (area under PR curve).
pub fn average_precision(y_true: &[usize], y_scores: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_scores.len(), "length mismatch");
    let total_pos = y_true.iter().filter(|&&y| y == 1).count() as f64;

    if total_pos == 0.0 {
        return 0.0;
    }

    let mut indices: Vec<usize> = (0..y_scores.len()).collect();
    indices.sort_by(|&a, &b| y_scores[b].partial_cmp(&y_scores[a]).unwrap());

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut ap = 0.0;
    let mut prev_recall = 0.0;

    for &i in &indices {
        if y_true[i] == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
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
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let stats = per_class_tp_fp_fn(&cm, n_classes);
    let supports = support_per_class(&cm, n_classes);

    let per_class: Vec<PerClassMetrics> = (0..n_classes)
        .map(|c| {
            let (tp, fp, fn_) = stats[c];
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
                support: supports[c],
            }
        })
        .collect();

    let acc = accuracy(y_true, y_pred);
    let total_support: usize = supports.iter().sum();

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

/// Balanced accuracy: average of per-class recall.
pub fn balanced_accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let n_classes = n_classes_from(y_true, y_pred);
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let stats = per_class_tp_fp_fn(&cm, n_classes);
    let sum_recall: f64 = stats
        .iter()
        .map(|&(tp, _fp, fn_)| {
            if tp + fn_ == 0 {
                0.0
            } else {
                tp as f64 / (tp + fn_) as f64
            }
        })
        .sum();
    sum_recall / n_classes as f64
}

/// Specificity (true negative rate) for binary classification.
pub fn specificity(y_true: &[usize], y_pred: &[usize]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
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
    if y_true.is_empty() {
        return 0.0;
    }
    let n_classes = n_classes_from(y_true, y_pred);
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let n = y_true.len() as f64;

    // Observed agreement
    let p_o: f64 = (0..n_classes).map(|k| cm[k][k] as f64).sum::<f64>() / n;

    // Expected agreement by chance
    let p_e: f64 = (0..n_classes)
        .map(|k| {
            let row_sum: f64 = cm[k].iter().sum::<u64>() as f64;
            let col_sum: f64 = (0..n_classes).map(|r| cm[r][k] as f64).sum();
            (row_sum * col_sum) / (n * n)
        })
        .sum();

    if (1.0 - p_e).abs() < 1e-15 {
        return 0.0;
    }
    (p_o - p_e) / (1.0 - p_e)
}

/// Hamming loss: fraction of labels that are incorrectly predicted.
/// For single-label: equals `1 - accuracy`.
pub fn hamming_loss(y_true: &[usize], y_pred: &[usize]) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    1.0 - accuracy(y_true, y_pred)
}

/// Jaccard similarity coefficient (IoU) with the given averaging strategy.
pub fn jaccard_score(y_true: &[usize], y_pred: &[usize], average: Average) -> f64 {
    if y_true.is_empty() {
        return 0.0;
    }
    let n_classes = n_classes_from(y_true, y_pred);
    let cm = confusion_matrix(y_true, y_pred, n_classes);
    let stats = per_class_tp_fp_fn(&cm, n_classes);
    let supports = support_per_class(&cm, n_classes);
    compute_averaged(&stats, &supports, average, |tp, fp, fn_| {
        let denom = tp + fp + fn_;
        if denom == 0 {
            0.0
        } else {
            tp as f64 / denom as f64
        }
    })
}

fn n_classes_from(y_true: &[usize], y_pred: &[usize]) -> usize {
    let max_true = y_true.iter().copied().max().unwrap_or(0);
    let max_pred = y_pred.iter().copied().max().unwrap_or(0);
    max_true.max(max_pred) + 1
}

fn compute_averaged(
    stats: &[(u64, u64, u64)],
    supports: &[usize],
    average: Average,
    metric_fn: impl Fn(u64, u64, u64) -> f64,
) -> f64 {
    let n_classes = stats.len();
    match average {
        Average::Micro => {
            let (total_tp, total_fp, total_fn) = stats
                .iter()
                .fold((0u64, 0u64, 0u64), |(a, b, c), &(tp, fp, fn_)| {
                    (a + tp, b + fp, c + fn_)
                });
            metric_fn(total_tp, total_fp, total_fn)
        }
        Average::Macro => {
            let sum: f64 = stats
                .iter()
                .map(|&(tp, fp, fn_)| metric_fn(tp, fp, fn_))
                .sum();
            sum / n_classes as f64
        }
        Average::Weighted => {
            let total_support: usize = supports.iter().sum();
            if total_support == 0 {
                return 0.0;
            }
            let weighted: f64 = stats
                .iter()
                .zip(supports.iter())
                .map(|(&(tp, fp, fn_), &s)| metric_fn(tp, fp, fn_) * s as f64)
                .sum();
            weighted / total_support as f64
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

    #[test]
    fn test_all_same_class() {
        let y_true = [0, 0, 0];
        let y_pred = [0, 0, 0];
        let acc = accuracy(&y_true, &y_pred);
        assert!((acc - 1.0).abs() < 1e-10);

        // MCC is 0 when all same class (no variation)
        let m = mcc(&y_true, &y_pred);
        assert!(m.abs() < 1e-10 || m.is_nan() || (m - 0.0).abs() < 1e-10);
    }

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
