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
pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "length mismatch");
    if y_true.is_empty() {
        return 0.0;
    }
    let mean_y: f64 = y_true.iter().sum::<f64>() / y_true.len() as f64;
    #[cfg(feature = "simd")]
    let ss_res: f64 = innr::dense_f64::l2_distance_squared_f64(y_true, y_pred);
    #[cfg(not(feature = "simd"))]
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&t, &p)| (t - p).powi(2))
        .sum();
    let ss_tot: f64 = y_true.iter().map(|&t| (t - mean_y).powi(2)).sum();
    if ss_tot < 1e-15 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
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
}
