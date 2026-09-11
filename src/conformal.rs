//! Score-only split-conformal calibration.
//!
//! This module selects a threshold from one scalar nonconformity score per
//! exchangeable calibration unit. It does not fit a score, aggregate groups,
//! or construct an interval or prediction set; callers own those semantics.
//! Scores must use the same fixed rule for calibration and future units, with
//! larger values denoting worse fit.

use thiserror::Error;

/// A target marginal coverage level for split conformal calibration.
///
/// Construct this with [`Coverage::from_ratio`] when a mathematical level is
/// known exactly, or [`Coverage::from_miscoverage`] when the intended level is
/// the exact binary value of an `f64` miscoverage `alpha`. Both constructors
/// require a level strictly between zero and one.
#[derive(Debug, Clone, Copy)]
pub struct Coverage(Representation);

#[derive(Debug, Clone, Copy)]
enum Representation {
    /// Target coverage `covered / total`.
    Ratio { covered: u64, total: u64 },
    /// Miscoverage `mantissa / 2^shift`, exactly as represented by `f64`.
    MiscoverageBinary { mantissa: u64, shift: u16 },
}

impl Coverage {
    /// Build coverage `covered / total` exactly.
    ///
    /// Returns [`ConformalError::InvalidCoverage`] unless
    /// `0 < covered < total`.
    pub fn from_ratio(covered: u64, total: u64) -> Result<Self, ConformalError> {
        if covered == 0 || covered >= total {
            return Err(ConformalError::InvalidCoverage);
        }
        Ok(Self(Representation::Ratio { covered, total }))
    }

    /// Build coverage from an `f64` miscoverage level `alpha`.
    ///
    /// The chosen rank is exact for the supplied binary floating-point value,
    /// rather than for a decimal spelling that may have produced it. Use
    /// [`Coverage::from_ratio`] when the intended level is a rational such as
    /// `9 / 10`.
    pub fn from_miscoverage(alpha: f64) -> Result<Self, ConformalError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(ConformalError::InvalidCoverage);
        }

        let bits = alpha.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as u16;
        let fraction = bits & ((1_u64 << 52) - 1);
        let (mantissa, shift) = if exponent == 0 {
            // Positive subnormal: fraction * 2^-1074.
            (fraction, 1074)
        } else {
            // Normal: (2^52 + fraction) * 2^(exponent - 1023 - 52).
            ((1_u64 << 52) | fraction, 1075 - exponent)
        };
        Ok(Self(Representation::MiscoverageBinary { mantissa, shift }))
    }

    fn rank(self, calibration_count: usize) -> u128 {
        let total_count = calibration_count as u128 + 1;
        match self.0 {
            Representation::Ratio { covered, total } => {
                let numerator = total_count * u128::from(covered);
                let denominator = u128::from(total);
                numerator.div_ceil(denominator)
            }
            Representation::MiscoverageBinary { mantissa, shift } => {
                // ceil(N * (1 - alpha)) equals N - floor(N * alpha).
                // `usize` is at most 64 bits on supported Rust platforms, so
                // this product uses at most 117 bits.
                let product = total_count * u128::from(mantissa);
                let floor_miscoverage = if shift >= 128 { 0 } else { product >> shift };
                total_count - floor_miscoverage
            }
        }
    }
}

/// The threshold selected by split conformal calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    /// A finite score threshold. Future scores at or below it are included.
    Finite(f64),
    /// The finite sample is too small for the requested coverage; include all
    /// candidates under the caller's score-to-prediction rule.
    Unbounded,
}

/// Invalid coverage levels or calibration scores.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ConformalError {
    /// Coverage must be strictly inside `(0, 1)`.
    #[error("coverage must be in (0, 1)")]
    InvalidCoverage,
    /// Calibration requires at least one score.
    #[error("calibration scores are empty")]
    EmptyCalibration,
    /// The score at this input position was NaN or infinite.
    #[error("calibration score at index {index} must be finite")]
    NonFiniteScore {
        /// Position in the original input slice.
        index: usize,
    },
}

/// Select a finite-sample split-conformal threshold in place.
///
/// For `n` calibration scores and target coverage `p`, this selects the
/// `ceil((n + 1) * p)`-th smallest score. If that rank exceeds `n`, returns
/// [`Threshold::Unbounded`], the conservative finite-sample fallback. Equal
/// scores are included because the caller should use an inclusive cutoff.
///
/// Fix the scoring rule independently of calibration. If calibration and
/// future scores are exchangeable conditional on that rule, accepting a future
/// score at or below the threshold gives marginal coverage at least `p`.
/// With [`Threshold::Unbounded`], accept every future score.
///
/// Scores may be signed but must be finite. Validation completes before any
/// mutation. A finite result reorders the slice without fully sorting it.
///
/// ```
/// use statskit::conformal::{calibrate_in_place, Coverage, Threshold};
///
/// let mut scores = [-1.0, 0.5, 2.0, 3.0];
/// let coverage = Coverage::from_ratio(3, 5)?;
/// assert_eq!(calibrate_in_place(&mut scores, coverage)?, Threshold::Finite(2.0));
/// # Ok::<(), statskit::conformal::ConformalError>(())
/// ```
pub fn calibrate_in_place(
    scores: &mut [f64],
    coverage: Coverage,
) -> Result<Threshold, ConformalError> {
    if scores.is_empty() {
        return Err(ConformalError::EmptyCalibration);
    }
    if let Some(index) = scores.iter().position(|score| !score.is_finite()) {
        return Err(ConformalError::NonFiniteScore { index });
    }

    let rank = coverage.rank(scores.len());
    if rank > scores.len() as u128 {
        return Ok(Threshold::Unbounded);
    }
    let (_, threshold, _) = scores.select_nth_unstable_by(rank as usize - 1, f64::total_cmp);
    Ok(Threshold::Finite(*threshold))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn rank_stays_wide_at_maximum_usize_count() {
        let coverage = Coverage::from_ratio(u64::MAX - 1, u64::MAX).unwrap();
        // `rank` is private because actual allocation at this length is neither
        // necessary nor possible in a unit test. It exercises the arithmetic
        // used for the largest representable `usize` count without overflowing a
        // `u64` intermediate.
        let count = usize::MAX;
        assert_eq!(coverage.rank(count), u128::from(u64::MAX));

        let tiny_alpha = Coverage::from_miscoverage(f64::from_bits(1)).unwrap();
        assert_eq!(tiny_alpha.rank(count), 1_u128 << 64);
    }
}
