//! Calibration check for `bootstrap_bca`: over seeded replicates, the nominal
//! 95% interval must contain the true parameter about 95% of the time.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statskit::{BootstrapConfig, bootstrap_bca, mean_diff};

#[test]
fn bca_95_percent_interval_covers_true_mean_about_95_percent() {
    // Exponential(1) samples are skewed, so the bias and acceleration terms
    // matter. scipy.stats.bootstrap(method="BCa") covers the true mean in
    // 0.935 of 400 replicates of this design (n = 30, B = 999).
    const REPLICATES: u64 = 400;
    const N: usize = 30;
    let true_mean = 1.0;
    let zeros = [0.0; N];

    let mut covered = 0u32;
    for rep in 0..REPLICATES {
        let mut rng = StdRng::seed_from_u64(rep);
        let a: Vec<f64> = (0..N).map(|_| -(1.0 - rng.random::<f64>()).ln()).collect();
        let ci = bootstrap_bca(
            &a,
            &zeros,
            mean_diff,
            BootstrapConfig {
                n_resamples: 999,
                alpha: 0.05,
                seed: Some(10_000 + rep),
            },
        );
        if ci.lower <= true_mean && true_mean <= ci.upper {
            covered += 1;
        }
    }

    // Binomial sd at p = 0.95 and 400 replicates is about 0.011.
    let coverage = f64::from(covered) / REPLICATES as f64;
    assert!(
        (0.90..=0.98).contains(&coverage),
        "BCa 95% coverage {coverage:.3} outside [0.90, 0.98]"
    );
}
