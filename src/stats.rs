//! Small statistical helpers (means, moments, etc.).
//!
//! Keep this module narrow and well-tested. Prefer reusing existing crates when a
//! dependency is already present in the workspace.

/// Compute the mean of a slice.
pub fn mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    Some(xs.iter().copied().sum::<f64>() / xs.len() as f64)
}

/// Numerically stable one-pass mean/second-moment accumulator (Welford).
///
/// Returns `(mean, m2)` where `m2` is the sum of squared deviations from the mean.
fn welford_mean_m2(xs: &[f64]) -> Option<(f64, f64)> {
    if xs.is_empty() {
        return None;
    }
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut n: u64 = 0;
    for x in xs.iter().copied() {
        n += 1;
        let delta = x - mean;
        mean += delta / (n as f64);
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    Some((mean, m2))
}

/// Compute the (population) variance of a slice.
///
/// Returns `None` for an empty slice.
pub fn variance_population(xs: &[f64]) -> Option<f64> {
    let (_mean, m2) = welford_mean_m2(xs)?;
    let var = m2 / (xs.len() as f64);
    Some(var.max(0.0))
}

/// Compute the (sample) variance of a slice (Bessel corrected).
///
/// Returns `None` if `xs.len() < 2`.
pub fn variance_sample(xs: &[f64]) -> Option<f64> {
    if xs.len() < 2 {
        return None;
    }
    let (_mean, m2) = welford_mean_m2(xs)?;
    let var = m2 / (xs.len() as f64 - 1.0);
    Some(var.max(0.0))
}

/// Compute the (population) standard deviation of a slice.
pub fn stddev_population(xs: &[f64]) -> Option<f64> {
    Some(variance_population(xs)?.sqrt())
}

/// Compute the (sample) standard deviation of a slice.
pub fn stddev_sample(xs: &[f64]) -> Option<f64> {
    Some(variance_sample(xs)?.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_variance_smoke() {
        let xs = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mean(&xs).unwrap(), 2.5);
        let vp = variance_population(&xs).unwrap();
        assert!((vp - 1.25).abs() < 1e-12);
        let vs = variance_sample(&xs).unwrap();
        assert!((vs - (5.0 / 3.0)).abs() < 1e-12);
        assert!((stddev_population(&xs).unwrap() - vp.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn variance_is_stable_under_large_offsets() {
        // A dataset with large mean and small variance (relative scale).
        // We choose increments that are representable at this magnitude.
        let xs = [
            1e16_f64,
            1e16_f64 + 2.0,
            1e16_f64 + 4.0,
            1e16_f64 + 6.0,
        ];
        let vp = variance_population(&xs).unwrap();
        // Variance of [0,2,4,6] is 5.0 (population).
        assert!((vp - 5.0).abs() < 1e-9, "vp={vp}");
    }
}
