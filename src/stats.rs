//! Statistical helpers: descriptive statistics, bootstrap, hypothesis tests, and effect sizes.

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;

// ---------------------------------------------------------------------------
// Descriptive statistics
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Bootstrap, hypothesis tests, multiple-comparison corrections, effect sizes
// ---------------------------------------------------------------------------

/// Configuration for bootstrap resampling.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BootstrapConfig {
    pub n_resamples: usize,
    pub alpha: f64,
    pub seed: Option<u64>,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            n_resamples: 10_000,
            alpha: 0.05,
            seed: None,
        }
    }
}

/// Result of a bootstrap confidence interval estimation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BootstrapResult {
    pub point_estimate: f64,
    pub lower: f64,
    pub upper: f64,
    pub p_value: f64,
}

/// Result of a Wilcoxon signed-rank test.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WilcoxonResult {
    pub statistic: f64,
    pub p_value: f64,
    pub significant: bool,
}

/// Result of a permutation test.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PermutationResult {
    pub observed: f64,
    pub p_value: f64,
    pub significant: bool,
}

/// BCa (bias-corrected and accelerated) bootstrap confidence interval.
///
/// `statistic` computes the test statistic from paired score vectors.
pub fn bootstrap_bca<F>(
    scores_a: &[f64],
    scores_b: &[f64],
    statistic: F,
    config: BootstrapConfig,
) -> BootstrapResult
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    assert_eq!(scores_a.len(), scores_b.len(), "length mismatch");
    let n = scores_a.len();
    assert!(n > 0, "empty input");

    let observed = statistic(scores_a, scores_b);

    let mut rng = match config.seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_os_rng(),
    };

    let indices: Vec<usize> = (0..n).collect();
    let mut boot_stats = Vec::with_capacity(config.n_resamples);

    for _ in 0..config.n_resamples {
        let sample_indices: Vec<usize> =
            (0..n).map(|_| *indices.choose(&mut rng).unwrap()).collect();
        let sa: Vec<f64> = sample_indices.iter().map(|&i| scores_a[i]).collect();
        let sb: Vec<f64> = sample_indices.iter().map(|&i| scores_b[i]).collect();
        boot_stats.push(statistic(&sa, &sb));
    }

    boot_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Bias correction: z0
    let prop_less =
        boot_stats.iter().filter(|&&s| s < observed).count() as f64 / config.n_resamples as f64;
    let z0 = normal_ppf(prop_less.clamp(1e-10, 1.0 - 1e-10));

    // Acceleration: jackknife estimate
    let mut jack_vals = Vec::with_capacity(n);
    for skip in 0..n {
        let ja: Vec<f64> = (0..n).filter(|&i| i != skip).map(|i| scores_a[i]).collect();
        let jb: Vec<f64> = (0..n).filter(|&i| i != skip).map(|i| scores_b[i]).collect();
        jack_vals.push(statistic(&ja, &jb));
    }
    let jack_mean: f64 = jack_vals.iter().sum::<f64>() / n as f64;
    let num: f64 = jack_vals.iter().map(|&v| (jack_mean - v).powi(3)).sum();
    let den: f64 = jack_vals.iter().map(|&v| (jack_mean - v).powi(2)).sum();
    let a = if den.abs() < 1e-15 {
        0.0
    } else {
        num / (6.0 * den.powf(1.5))
    };

    let z_alpha_lo = normal_ppf(config.alpha / 2.0);
    let z_alpha_hi = normal_ppf(1.0 - config.alpha / 2.0);

    let adj_lo = normal_cdf(z0 + (z0 + z_alpha_lo) / (1.0 - a * (z0 + z_alpha_lo)));
    let adj_hi = normal_cdf(z0 + (z0 + z_alpha_hi) / (1.0 - a * (z0 + z_alpha_hi)));

    let lo_idx = (adj_lo * config.n_resamples as f64).round() as usize;
    let hi_idx = (adj_hi * config.n_resamples as f64).round() as usize;
    let lo_idx = lo_idx.min(config.n_resamples - 1);
    let hi_idx = hi_idx.min(config.n_resamples - 1);

    // Two-sided p-value: proportion of bootstrap stats on the other side of zero
    let p_value = if observed >= 0.0 {
        2.0 * boot_stats.iter().filter(|&&s| s <= 0.0).count() as f64 / config.n_resamples as f64
    } else {
        2.0 * boot_stats.iter().filter(|&&s| s >= 0.0).count() as f64 / config.n_resamples as f64
    }
    .min(1.0);

    BootstrapResult {
        point_estimate: observed,
        lower: boot_stats[lo_idx],
        upper: boot_stats[hi_idx],
        p_value,
    }
}

/// Wilcoxon signed-rank test (paired, non-parametric).
///
/// Uses normal approximation for n > 20.
pub fn wilcoxon(a: &[f64], b: &[f64], alpha: f64) -> WilcoxonResult {
    assert_eq!(a.len(), b.len(), "length mismatch");

    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();
    let nonzero: Vec<f64> = diffs.iter().copied().filter(|&d| d.abs() > 1e-15).collect();
    let n = nonzero.len();

    if n == 0 {
        return WilcoxonResult {
            statistic: 0.0,
            p_value: 1.0,
            significant: false,
        };
    }

    // Rank absolute values
    let mut abs_indexed: Vec<(usize, f64)> = nonzero
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d.abs()))
        .collect();
    abs_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (abs_indexed[j].1 - abs_indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[abs_indexed[k].0] = avg_rank;
        }
        i = j;
    }

    let w_plus: f64 = nonzero
        .iter()
        .zip(ranks.iter())
        .filter(|(d, _)| **d > 0.0)
        .map(|(_, r)| *r)
        .sum();

    let w_minus: f64 = nonzero
        .iter()
        .zip(ranks.iter())
        .filter(|(d, _)| **d < 0.0)
        .map(|(_, r)| *r)
        .sum();

    let w = w_plus.min(w_minus);

    // Normal approximation
    let n_f = n as f64;
    let mean_w = n_f * (n_f + 1.0) / 4.0;
    let var_w = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0;

    let z = if var_w > 0.0 {
        (w - mean_w).abs() / var_w.sqrt()
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - normal_cdf(z));

    WilcoxonResult {
        statistic: w,
        p_value,
        significant: p_value < alpha,
    }
}

/// Permutation test for comparing two paired score vectors.
///
/// `statistic` computes the test statistic. p-value is the proportion of
/// permuted statistics with absolute value >= |observed|.
pub fn permutation_test<F>(
    a: &[f64],
    b: &[f64],
    n_permutations: usize,
    statistic: F,
    seed: Option<u64>,
) -> PermutationResult
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    assert_eq!(a.len(), b.len(), "length mismatch");
    let n = a.len();
    let observed = statistic(a, b);

    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_os_rng(),
    };

    let mut count_extreme = 0usize;

    for _ in 0..n_permutations {
        let mut pa = a.to_vec();
        let mut pb = b.to_vec();
        for i in 0..n {
            let flip: bool = *[true, false].choose(&mut rng).unwrap();
            if flip {
                std::mem::swap(&mut pa[i], &mut pb[i]);
            }
        }
        let perm_stat = statistic(&pa, &pb);
        if perm_stat.abs() >= observed.abs() {
            count_extreme += 1;
        }
    }

    let p_value = count_extreme as f64 / n_permutations as f64;
    PermutationResult {
        observed,
        p_value,
        significant: p_value < 0.05,
    }
}

/// Almost Stochastic Order (ASO) violation level.
///
/// Returns epsilon_min in [0, 1]. Values < 0.5 indicate A stochastically
/// dominates B. Based on del Barrio et al. (2018) and Dror et al. (2019).
pub fn aso(scores_a: &[f64], scores_b: &[f64], n_bootstrap: usize, seed: Option<u64>) -> f64 {
    let n_a = scores_a.len();
    let n_b = scores_b.len();
    assert!(n_a > 0 && n_b > 0, "empty input");

    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_os_rng(),
    };

    let indices_a: Vec<usize> = (0..n_a).collect();
    let indices_b: Vec<usize> = (0..n_b).collect();

    let mut eps_values = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample
        let sa: Vec<f64> = (0..n_a)
            .map(|_| scores_a[*indices_a.choose(&mut rng).unwrap()])
            .collect();
        let sb: Vec<f64> = (0..n_b)
            .map(|_| scores_b[*indices_b.choose(&mut rng).unwrap()])
            .collect();

        eps_values.push(compute_epsilon(&sa, &sb));
    }

    // Return the mean epsilon
    eps_values.iter().sum::<f64>() / n_bootstrap as f64
}

fn compute_epsilon(a: &[f64], b: &[f64]) -> f64 {
    // Compute the minimum epsilon such that F_A(x) <= F_B(x) + epsilon for all x
    // This is the maximum violation of stochastic order
    let mut all_vals: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    all_vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    all_vals.dedup();

    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    let mut max_violation = 0.0f64;

    for &x in &all_vals {
        // F_B(x) - F_A(x): if positive, A dominates at this point
        // We want max(F_A(x) - F_B(x), 0) as violation
        let fa = a.iter().filter(|&&v| v <= x).count() as f64 / n_a;
        let fb = b.iter().filter(|&&v| v <= x).count() as f64 / n_b;
        // For A to stochastically dominate B (A >= B in distribution),
        // we need F_A(x) <= F_B(x) for all x (A has more mass on higher values).
        // Violation = max(F_A(x) - F_B(x), 0)
        let violation = (fa - fb).max(0.0);
        max_violation = max_violation.max(violation);
    }

    max_violation
}

/// Benjamini-Hochberg FDR correction.
pub fn benjamini_hochberg(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();
    if n == 0 {
        return vec![];
    }

    let mut indexed: Vec<(usize, f64)> =
        p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut adjusted = vec![0.0; n];
    let mut cummin = f64::INFINITY;

    for i in (0..n).rev() {
        let rank = i + 1;
        let adj = (indexed[i].1 * n as f64 / rank as f64).min(1.0);
        cummin = cummin.min(adj);
        adjusted[indexed[i].0] = cummin;
    }

    adjusted
}

/// Bonferroni correction: `p_adj[i] = min(p[i] * n, 1.0)`.
pub fn bonferroni(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len() as f64;
    p_values.iter().map(|&p| (p * n).min(1.0)).collect()
}

/// Cohen's d effect size (pooled standard deviation).
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    if n_a < 2.0 || n_b < 2.0 {
        return 0.0;
    }

    let mean_a: f64 = a.iter().sum::<f64>() / n_a;
    let mean_b: f64 = b.iter().sum::<f64>() / n_b;

    let var_a: f64 = a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b: f64 = b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    let pooled_sd = (((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0)).sqrt();

    if pooled_sd < 1e-15 {
        return 0.0;
    }

    (mean_a - mean_b) / pooled_sd
}

/// Rank-biserial correlation (non-parametric effect size).
///
/// Based on the Wilcoxon W+ and W- sums. Range [-1, 1].
pub fn rank_biserial(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();
    let nonzero: Vec<f64> = diffs.iter().copied().filter(|&d| d.abs() > 1e-15).collect();
    let n = nonzero.len();

    if n == 0 {
        return 0.0;
    }

    let mut abs_indexed: Vec<(usize, f64)> = nonzero
        .iter()
        .enumerate()
        .map(|(i, &d)| (i, d.abs()))
        .collect();
    abs_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (abs_indexed[j].1 - abs_indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[abs_indexed[k].0] = avg_rank;
        }
        i = j;
    }

    let w_plus: f64 = nonzero
        .iter()
        .zip(ranks.iter())
        .filter(|(d, _)| **d > 0.0)
        .map(|(_, r)| *r)
        .sum();

    let w_minus: f64 = nonzero
        .iter()
        .zip(ranks.iter())
        .filter(|(d, _)| **d < 0.0)
        .map(|(_, r)| *r)
        .sum();

    let total_rank = w_plus + w_minus;
    if total_rank < 1e-15 {
        return 0.0;
    }

    (w_plus - w_minus) / total_rank
}

/// Convenience: mean difference between paired score vectors.
pub fn mean_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).sum();
    sum / a.len() as f64
}

// ---------------------------------------------------------------------------
// Normal distribution helpers (no external dependency)
// ---------------------------------------------------------------------------

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn normal_ppf(p: f64) -> f64 {
    // Rational approximation (Abramowitz & Stegun, adapted from Peter Acklam)
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

fn erf(x: f64) -> f64 {
    // Horner form of the Abramowitz & Stegun approximation (max error ~1.5e-7)
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Descriptive statistics tests (from original statskit) ---

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
        let xs = [1e16_f64, 1e16_f64 + 2.0, 1e16_f64 + 4.0, 1e16_f64 + 6.0];
        let vp = variance_population(&xs).unwrap();
        // Variance of [0,2,4,6] is 5.0 (population).
        assert!((vp - 5.0).abs() < 1e-9, "vp={vp}");
    }

    // --- Statistical tests (from evalkit) ---

    #[test]
    fn test_bootstrap_bca_known() {
        // Two clearly different distributions: CI should not contain 0
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
        let config = BootstrapConfig {
            n_resamples: 5000,
            alpha: 0.05,
            seed: Some(42),
        };
        let result = bootstrap_bca(&a, &b, mean_diff, config);
        // mean_diff = -10.0, CI should be entirely negative
        assert!(result.point_estimate < 0.0);
        assert!(result.upper < 0.0);
    }

    #[test]
    fn test_wilcoxon_identical() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = wilcoxon(&a, &b, 0.05);
        assert!(!result.significant);
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wilcoxon_different() {
        let a: Vec<f64> = (0..30).map(|i| i as f64 + 10.0).collect();
        let b: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let result = wilcoxon(&a, &b, 0.05);
        assert!(result.significant, "p={}", result.p_value);
    }

    #[test]
    fn test_permutation_test_identical() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = permutation_test(&a, &b, 1000, mean_diff, Some(42));
        assert!(!result.significant);
    }

    #[test]
    fn test_permutation_test_different() {
        let a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        let b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let result = permutation_test(&a, &b, 5000, mean_diff, Some(42));
        assert!(result.significant, "p={}", result.p_value);
    }

    #[test]
    fn test_benjamini_hochberg() {
        let pvals = [0.01, 0.04, 0.03, 0.20];
        let adj = benjamini_hochberg(&pvals);
        // Sorted: 0.01(rank1), 0.03(rank2), 0.04(rank3), 0.20(rank4)
        // Adjusted: 0.01*4/1=0.04, 0.03*4/2=0.06, 0.04*4/3=0.053, 0.20*4/4=0.20
        // With monotonicity enforcement (cummin from right)
        assert!(adj[0] < adj[3]); // smallest p stays smallest
        assert!(adj[0] <= 0.05); // 0.01 should still be significant
        for &a in &adj {
            assert!(a <= 1.0);
            assert!(a >= 0.0);
        }
    }

    #[test]
    fn test_bonferroni() {
        let pvals = [0.01, 0.03, 0.50];
        let adj = bonferroni(&pvals);
        assert!((adj[0] - 0.03).abs() < 1e-10);
        assert!((adj[1] - 0.09).abs() < 1e-10);
        assert!((adj[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cohens_d_zero() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let d = cohens_d(&a, &b);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_cohens_d_large() {
        let a = [10.0, 11.0, 12.0, 13.0, 14.0];
        let b = [0.0, 1.0, 2.0, 3.0, 4.0];
        let d = cohens_d(&a, &b);
        assert!(d > 0.8, "d = {d}");
    }

    #[test]
    fn test_rank_biserial() {
        let a = [10.0, 11.0, 12.0, 13.0, 14.0];
        let b = [0.0, 1.0, 2.0, 3.0, 4.0];
        let rb = rank_biserial(&a, &b);
        assert!(rb >= -1.0 && rb <= 1.0, "rb = {rb}");
        assert!(rb > 0.0); // a clearly greater
    }

    #[test]
    fn test_aso_dominance() {
        // A clearly better (higher scores) than B
        let a = [0.9, 0.85, 0.88, 0.92, 0.87, 0.91, 0.86, 0.89];
        let b = [0.5, 0.45, 0.48, 0.52, 0.47, 0.51, 0.46, 0.49];
        let eps = aso(&a, &b, 1000, Some(42));
        assert!(eps < 0.5, "eps = {eps}");
    }

    #[test]
    fn test_mean_diff() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let d = mean_diff(&a, &b);
        assert!((d - (-3.0)).abs() < 1e-10);
    }
}
