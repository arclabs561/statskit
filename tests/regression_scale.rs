//! R² = 1 - SS_res / SS_tot is unchanged by multiplying both inputs by a
//! nonzero scale when the squared sums remain representable.

#[test]
fn r_squared_preserves_negative_score_at_small_scale() {
    // Target mean is 2s, SS_tot = 2s², SS_res = 5s²: R² = -1.5.
    for scale in [1.0, 2.0_f64.powi(-30), 2.0_f64.powi(30)] {
        let actual =
            statskit::r_squared(&[scale, 2.0 * scale, 3.0 * scale], &[scale, scale, scale]);
        assert_eq!(actual, -1.5, "scale {scale}");
    }
}

#[test]
fn constant_target_rejects_even_small_nonzero_prediction_error() {
    let scale = 2.0_f64.powi(-30);
    assert_eq!(statskit::r_squared(&[scale; 3], &[scale; 3]), 1.0);
    assert_eq!(statskit::r_squared(&[scale; 3], &[0.0; 3]), 0.0);
}
