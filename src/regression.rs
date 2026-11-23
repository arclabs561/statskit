/// Mean Squared Error.
pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    if y_true.is_empty() {
        return 0.0;
    }
    // SIMD-accelerated reduction via innr when the `simd` feature is enabled;
    // portable fallback uses the iterator path.
    #[cfg(feature = "simd")]
    let sum: f64 = innr::dense_f64::l2_distance_squared_f64(y_true, y_pred);
    #[cfg(not(feature = "simd"))]
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    sum / y_true.len() as f64
}

/// Root Mean Squared Error.
pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    mse(y_true, y_pred).sqrt()
}

/// Mean Absolute Error.
pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    if y_true.is_empty() {
        return 0.0;
    }
    #[cfg(feature = "simd")]
    let sum: f64 = innr::dense_f64::l1_distance_f64(y_true, y_pred);
    #[cfg(not(feature = "simd"))]
    let sum: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).abs())
        .sum();
    sum / y_true.len() as f64
}

/// R-squared (coefficient of determination).
///
/// Returns 1.0 for perfect predictions. Can be negative when the model is
/// worse than predicting the mean.
/// When the total sum of squares is zero, returns 1.0 if the residual sum of
/// squares is also zero, and 0.0 otherwise. No absolute tolerance is applied:
/// small nonzero variance still uses the usual ratio of sums of squares.
pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    if y_true.is_empty() {
        return 0.0;
    }
    // Test constancy before rounding the mean or squaring a tiny residual.
    if y_true[0].is_finite() && y_true.iter().all(|&value| value == y_true[0]) {
        return if y_true == y_pred { 1.0 } else { 0.0 };
    }
    if !y_true.iter().chain(y_pred).all(|x| x.is_finite()) {
        let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let total = y_true.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        let residual = y_true
            .iter()
            .zip(y_pred)
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>();
        return if total == 0.0 {
            0.0
        } else {
            1.0 - residual / total
        };
    }
    // A power-of-two unit preserves neighboring representable observations.
    // Center relative to the first observation before taking the mean, so a
    // mean between adjacent floats remains representable in relative units.
    let max = y_true.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let scale = f64::from_bits(max.to_bits() & 0x7ff0_0000_0000_0000).max(f64::MIN_POSITIVE);
    let origin = y_true[0] / scale;
    let offset = |x: f64| x / scale - origin;
    let mean = y_true.iter().map(|&x| offset(x)).sum::<f64>() / y_true.len() as f64;
    let total = y_true
        .iter()
        .map(|&x| (offset(x) - mean).powi(2))
        .sum::<f64>();
    let ratio: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(&t, &p)| {
            let delta = t - p;
            let scaled = if delta.is_finite() {
                delta / scale
            } else {
                t / scale - p / scale
            };
            let square = scaled * scaled;
            if square.is_finite() {
                square / total
            } else {
                (scaled / total.sqrt()).powi(2)
            }
        })
        .sum();
    1.0 - ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_perfect() {
        let y_true = [1.0, 2.0, 3.0, 4.0];
        let y_pred = [1.0, 2.0, 3.0, 4.0];
        assert!((mse(&y_true, &y_pred) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rmse_known() {
        let y_true = [1.0, 2.0, 3.0];
        let y_pred = [1.5, 2.5, 3.5];
        // MSE = (0.25 + 0.25 + 0.25) / 3 = 0.25
        // RMSE = 0.5
        let r = rmse(&y_true, &y_pred);
        assert!((r - 0.5).abs() < 1e-10, "expected 0.5, got {r}");
    }

    #[test]
    fn test_mae_known() {
        let y_true = [1.0, 2.0, 3.0];
        let y_pred = [1.5, 2.5, 3.5];
        // MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        let m = mae(&y_true, &y_pred);
        assert!((m - 0.5).abs() < 1e-10, "expected 0.5, got {m}");
    }

    #[test]
    fn test_r_squared_perfect() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r2 = r_squared(&y_true, &y_pred);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_r_squared_mean_predictor() {
        let y_true = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean_val = 3.0;
        let y_pred = [mean_val; 5];
        let r2 = r_squared(&y_true, &y_pred);
        assert!(r2.abs() < 1e-10, "expected 0.0, got {r2}");
    }

    #[test]
    fn constant_target_score_does_not_depend_on_rounded_sums() {
        let target = [0.1; 3];
        assert_eq!(r_squared(&target, &target), 1.0);
        assert_eq!(r_squared(&target, &[0.0; 3]), 0.0);

        // Squaring this nonzero prediction error underflows.
        assert_eq!(r_squared(&[1e-200; 3], &[0.0; 3]), 0.0);
    }

    #[test]
    fn adjacent_observations_keep_their_midpoint_in_any_units() {
        for x in [2f64.powi(-600), 1.0, 2f64.powi(600)] {
            let z = f64::from_bits(x.to_bits() - 1);
            assert!((r_squared(&[z, x], &[x, x]) + 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn r_squared_is_invariant_to_extreme_units() {
        for scale in [1e-200, 1.0, 1e200] {
            assert_eq!(r_squared(&[0.0, scale], &[0.0, 0.0]), -1.0);
        }
    }

    #[test]
    fn r_squared_constant_target_distinguishes_perfect_prediction() {
        let y_true = [3.0, 3.0, 3.0];

        assert_eq!(r_squared(&y_true, &y_true), 1.0);
        assert_eq!(r_squared(&y_true, &[2.0, 2.0, 2.0]), 0.0);
    }
}
