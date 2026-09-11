use num_rational::BigRational;
use proptest::prelude::*;
use statskit::conformal::{ConformalError, Coverage, Threshold, calibrate_in_place};

fn rational_rank(scores: &[f64], coverage: BigRational) -> Option<f64> {
    let n = BigRational::from_integer((scores.len() as u64 + 1).into());
    let target = n * coverage;
    let numerator = target.numer().clone();
    let denominator = target.denom().clone();
    let rank = (numerator + denominator.clone() - 1_u8) / denominator;
    let rank: usize = rank.try_into().expect("small test rank");
    if rank > scores.len() {
        return None;
    }
    let mut ordered = scores.to_vec();
    ordered.sort_by(f64::total_cmp);
    Some(ordered[rank - 1])
}

fn oracle_for_alpha(scores: &[f64], alpha: f64) -> Option<f64> {
    let alpha = BigRational::from_float(alpha).expect("finite alpha");
    rational_rank(scores, BigRational::from_integer(1.into()) - alpha)
}

#[test]
fn exact_ratio_selects_order_statistic_and_preserves_inclusive_ties() {
    let mut scores = [-3.0, -1.0, -1.0, 4.0];
    let coverage = Coverage::from_ratio(3, 5).unwrap();
    assert_eq!(
        calibrate_in_place(&mut scores, coverage),
        Ok(Threshold::Finite(-1.0))
    );
}

#[test]
fn exact_ratio_returns_unbounded_when_rank_needs_more_scores() {
    let mut scores = [-1.0];
    let coverage = Coverage::from_ratio(9, 10).unwrap();
    assert_eq!(
        calibrate_in_place(&mut scores, coverage),
        Ok(Threshold::Unbounded)
    );
}

#[test]
fn validation_precedes_mutation() {
    let mut scores = [4.0, f64::NAN, -3.0];
    let original = scores;
    assert_eq!(
        calibrate_in_place(&mut scores, Coverage::from_ratio(1, 2).unwrap()),
        Err(ConformalError::NonFiniteScore { index: 1 })
    );
    assert!(scores[0].to_bits() == original[0].to_bits());
    assert!(scores[1].is_nan());
    assert!(scores[2].to_bits() == original[2].to_bits());
}

#[test]
fn rejects_empty_and_nonfinite_scores() {
    let mut empty = [];
    assert_eq!(
        calibrate_in_place(&mut empty, Coverage::from_ratio(1, 2).unwrap()),
        Err(ConformalError::EmptyCalibration)
    );
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut scores = [bad];
        assert_eq!(
            calibrate_in_place(&mut scores, Coverage::from_ratio(1, 2).unwrap()),
            Err(ConformalError::NonFiniteScore { index: 0 })
        );
    }
}

#[test]
fn rejects_invalid_levels() {
    for alpha in [f64::NAN, f64::NEG_INFINITY, 0.0, 1.0, f64::INFINITY] {
        assert!(matches!(
            Coverage::from_miscoverage(alpha),
            Err(ConformalError::InvalidCoverage)
        ));
    }
    for (covered, total) in [(0, 1), (1, 1), (2, 1), (1, 0)] {
        assert!(matches!(
            Coverage::from_ratio(covered, total),
            Err(ConformalError::InvalidCoverage)
        ));
    }
}

#[test]
fn float_miscoverage_matches_independent_rational_oracle_at_boundaries() {
    let scores = [-4.0, -1.0, 0.0, 1.0, 7.0];
    for alpha in [
        f64::from_bits(1),
        0.1,
        0.5,
        0.9,
        f64::from_bits(0x3fefffffffffffff),
    ] {
        let mut input = scores;
        let got =
            calibrate_in_place(&mut input, Coverage::from_miscoverage(alpha).unwrap()).unwrap();
        let want = oracle_for_alpha(&scores, alpha)
            .map(Threshold::Finite)
            .unwrap_or(Threshold::Unbounded);
        assert_eq!(got, want, "alpha={alpha:?}");
    }
}

#[test]
fn represented_alpha_avoids_rounded_one_minus_alpha_rank_change() {
    let alpha = 0.3333333333333333_f64;
    let mut scores = [-2.0, 5.0];
    // For the represented alpha, floor(3 * alpha) is zero, so the exact rank
    // is three and the finite-sample fallback applies. Computing `1.0 - alpha`
    // first rounds this case differently and can incorrectly select rank two.
    assert_eq!(
        calibrate_in_place(&mut scores, Coverage::from_miscoverage(alpha).unwrap()),
        Ok(Threshold::Unbounded)
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct("tests/conformal.proptest-regressions")
        )),
        .. ProptestConfig::default()
    })]

    #[test]
    fn represented_float_rank_matches_big_rational_oracle(
        scores in proptest::collection::vec(-1.0e6_f64..1.0e6, 1..40),
        alpha in prop_oneof![
            Just(f64::from_bits(1)),
            Just(0.1),
            Just(0.3333333333333333),
            Just(0.5),
            Just(0.9),
            Just(f64::from_bits(0x3fefffffffffffff)),
            0.0_f64..1.0_f64,
            (1_u64..0x3ff0_0000_0000_0000_u64).prop_map(f64::from_bits),
        ],
    ) {
        prop_assume!(alpha.is_finite() && alpha > 0.0 && alpha < 1.0);
        let expected = oracle_for_alpha(&scores, alpha)
            .map(Threshold::Finite)
            .unwrap_or(Threshold::Unbounded);
        let mut input = scores;
        let got = calibrate_in_place(&mut input, Coverage::from_miscoverage(alpha).unwrap()).unwrap();
        prop_assert_eq!(got, expected);
    }

    #[test]
    fn exact_ratio_rank_matches_big_rational_oracle(
        scores in proptest::collection::vec(-1.0e6_f64..1.0e6, 1..40),
        (total, covered) in (2_u64..=u64::MAX)
            .prop_flat_map(|total| (Just(total), 1..total)),
    ) {
        let expected = rational_rank(&scores, BigRational::new(covered.into(), total.into()))
            .map(Threshold::Finite)
            .unwrap_or(Threshold::Unbounded);
        let mut input = scores;
        let got = calibrate_in_place(&mut input, Coverage::from_ratio(covered, total).unwrap()).unwrap();
        prop_assert_eq!(got, expected);
    }

    #[test]
    fn leave_one_out_includes_at_least_the_requested_number_of_scores(
        scores in proptest::collection::vec(-3_i8..=3_i8, 2..40)
            .prop_map(|scores| scores.into_iter().map(f64::from).collect::<Vec<_>>()),
        (total, covered) in (2_u64..100)
            .prop_flat_map(|total| (Just(total), 1..total)),
    ) {
        let coverage = Coverage::from_ratio(covered, total).unwrap();
        let total_points = scores.len() as u128;
        let expected_included = (total_points * u128::from(covered)).div_ceil(u128::from(total));
        let mut included = 0_u128;
        for held_out in 0..scores.len() {
            let mut calibration = scores.clone();
            let observed = calibration.remove(held_out);
            let threshold = calibrate_in_place(&mut calibration, coverage).unwrap();
            let is_included = match threshold {
                // The inclusive cutoff is required for the order-statistic
                // guarantee and can only increase coverage under ties.
                Threshold::Finite(qhat) => observed <= qhat,
                Threshold::Unbounded => true,
            };
            included += if is_included { 1 } else { 0 };
        }
        prop_assert!(included >= expected_included);
    }
}
