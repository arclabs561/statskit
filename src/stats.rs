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

/// Compute the (population) variance of a slice.
///
/// Returns `None` for an empty slice.
pub fn variance_population(xs: &[f64]) -> Option<f64> {
    let m = mean(xs)?;
    Some(xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64)
}

/// Compute the (sample) variance of a slice (Bessel corrected).
///
/// Returns `None` if `xs.len() < 2`.
pub fn variance_sample(xs: &[f64]) -> Option<f64> {
    if xs.len() < 2 {
        return None;
    }
    let m = mean(xs)?;
    Some(xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (xs.len() as f64 - 1.0))
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
}
