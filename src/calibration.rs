/// A single bin in a reliability diagram.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CalibrationBin {
    pub mean_predicted: f64,
    pub fraction_positive: f64,
    pub count: usize,
}

fn validate_binary_probabilities(y_true: &[usize], y_prob: &[f64]) {
    assert!(
        y_true.iter().all(|&label| label <= 1),
        "labels must be binary"
    );
    assert!(
        y_prob.iter().all(|p| p.is_finite()),
        "probabilities must be finite"
    );
    assert!(
        y_prob.iter().all(|p| (0.0..=1.0).contains(p)),
        "probabilities must be in [0, 1]"
    );
}

/// Brier score: mean squared error between predicted probability and outcome.
///
/// Binary classification. Returns 0.0 for empty input.
pub fn brier_score(y_true: &[usize], y_prob: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_prob.len(), "length mismatch");
    validate_binary_probabilities(y_true, y_prob);
    if y_true.is_empty() {
        return 0.0;
    }
    let sum: f64 = y_true
        .iter()
        .zip(y_prob.iter())
        .map(|(&y, &p)| {
            let diff = p - y as f64;
            diff * diff
        })
        .sum();
    sum / y_true.len() as f64
}

/// Expected Calibration Error: weighted average of |accuracy - confidence| per bin.
///
/// Returns 0.0 for empty input.
pub fn expected_calibration_error(y_true: &[usize], y_prob: &[f64], n_bins: usize) -> f64 {
    assert_eq!(y_true.len(), y_prob.len(), "length mismatch");
    assert!(n_bins > 0, "n_bins must be positive");
    validate_binary_probabilities(y_true, y_prob);
    if y_true.is_empty() {
        return 0.0;
    }
    let bins = reliability_diagram(y_true, y_prob, n_bins);
    let n = y_true.len() as f64;
    bins.iter()
        .map(|b| (b.count as f64 / n) * (b.fraction_positive - b.mean_predicted).abs())
        .sum()
}

/// Reliability diagram: bin predictions and compute observed fraction of positives per bin.
pub fn reliability_diagram(y_true: &[usize], y_prob: &[f64], n_bins: usize) -> Vec<CalibrationBin> {
    assert_eq!(y_true.len(), y_prob.len(), "length mismatch");
    assert!(n_bins > 0, "n_bins must be positive");
    validate_binary_probabilities(y_true, y_prob);

    let mut bin_sums = vec![0.0; n_bins];
    let mut bin_pos = vec![0.0; n_bins];
    let mut bin_counts = vec![0usize; n_bins];

    for (&y, &p) in y_true.iter().zip(y_prob.iter()) {
        let bin = ((p * n_bins as f64) as usize).min(n_bins - 1);
        bin_sums[bin] += p;
        bin_pos[bin] += y as f64;
        bin_counts[bin] += 1;
    }

    (0..n_bins)
        .filter(|&i| bin_counts[i] > 0)
        .map(|i| CalibrationBin {
            mean_predicted: bin_sums[i] / bin_counts[i] as f64,
            fraction_positive: bin_pos[i] / bin_counts[i] as f64,
            count: bin_counts[i],
        })
        .collect()
}

/// Maximum Calibration Error: worst-case bin calibration gap.
///
/// Same binning as ECE, but takes the max gap instead of the weighted average.
pub fn maximum_calibration_error(y_true: &[usize], y_prob: &[f64], n_bins: usize) -> f64 {
    assert_eq!(y_true.len(), y_prob.len(), "length mismatch");
    assert!(n_bins > 0, "n_bins must be positive");
    validate_binary_probabilities(y_true, y_prob);
    if y_true.is_empty() {
        return 0.0;
    }
    let bins = reliability_diagram(y_true, y_prob, n_bins);
    bins.iter()
        .map(|b| (b.fraction_positive - b.mean_predicted).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brier_perfect() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.0, 0.0, 1.0, 1.0];
        let bs = brier_score(&y_true, &y_prob);
        assert!((bs - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_brier_worst() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [1.0, 1.0, 0.0, 0.0];
        let bs = brier_score(&y_true, &y_prob);
        assert!((bs - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ece_perfect() {
        // Perfectly calibrated: probability matches outcome exactly
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.0, 0.0, 1.0, 1.0];
        let ece = expected_calibration_error(&y_true, &y_prob, 10);
        assert!((ece - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mce_perfectly_calibrated() {
        let y_true = [0, 0, 1, 1];
        let y_prob = [0.0, 0.0, 1.0, 1.0];
        let mce = maximum_calibration_error(&y_true, &y_prob, 10);
        assert!((mce - 0.0).abs() < 1e-10, "expected 0, got {mce}");
    }

    #[test]
    fn test_mce_worst_bin() {
        // Bin near 0.8: predictions ~0.8, all actually negative -> gap = 0.8
        // Bin near 0.1: predictions ~0.1, all actually positive -> gap = 0.9
        // MCE should pick the worst bin
        let y_true = [1, 1, 0, 0];
        let y_prob = [0.1, 0.15, 0.8, 0.85];
        let mce = maximum_calibration_error(&y_true, &y_prob, 10);
        assert!(mce > 0.5, "expected large MCE, got {mce}");
    }

    #[test]
    fn test_reliability_bins() {
        let y_true = [0, 0, 1, 1, 1];
        let y_prob = [0.1, 0.2, 0.7, 0.8, 0.9];
        let bins = reliability_diagram(&y_true, &y_prob, 5);
        // Should have bins with counts summing to 5
        let total: usize = bins.iter().map(|b| b.count).sum();
        assert_eq!(total, 5);
        // All fractions should be in [0, 1]
        for b in &bins {
            assert!(b.fraction_positive >= 0.0 && b.fraction_positive <= 1.0);
            assert!(b.mean_predicted >= 0.0 && b.mean_predicted <= 1.0);
        }
    }

    #[test]
    #[should_panic(expected = "labels must be binary")]
    fn brier_rejects_non_binary_labels() {
        brier_score(&[2], &[0.5]);
    }

    #[test]
    #[should_panic(expected = "probabilities must be finite")]
    fn reliability_rejects_non_finite_probabilities() {
        reliability_diagram(&[1], &[f64::NAN], 10);
    }

    #[test]
    #[should_panic(expected = "probabilities must be in [0, 1]")]
    fn brier_rejects_out_of_range_probabilities() {
        brier_score(&[1], &[1.1]);
    }

    #[test]
    #[should_panic(expected = "n_bins must be positive")]
    fn calibration_errors_reject_zero_bins() {
        expected_calibration_error(&[1], &[0.5], 0);
    }
}
