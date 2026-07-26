"""Bootstrap confidence intervals and proportion intervals.

Vendored from Happynood/quant-reasoning-bench (src/quantthink/metrics/stats.py),
which came from Happynood/quant-toolcall-bench (quantcall/metrics/bootstrap.py).

Two changes against the original:

* **Vectorized.** The spec calls for 10 000 resamples on every reported delta,
  over hundreds of episodes. The pure-Python resampling loop is fine for a
  handful of numbers and far too slow once every scheme x perturbation cell
  needs one.
* **Paired resampling added.** A quantized policy and the fp32 baseline are
  evaluated from *identical* initial states under an identical RNG stream, so
  their episode returns are matched pairs, not independent samples. Resampling
  the two independently would throw away that pairing and inflate the interval
  with between-episode variance that cancels exactly in the difference. The
  paired estimator resamples episode indices once and applies them to both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float
    confidence: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0.0 or self.high < 0.0

    @property
    def width(self) -> float:
        return self.high - self.low

    def overlaps(self, other: Interval) -> bool:
        return self.low <= other.high and other.low <= self.high

    def to_json(self) -> dict[str, float]:
        return {
            "estimate": self.estimate,
            "ci_low": self.low,
            "ci_high": self.high,
            "confidence": self.confidence,
        }


def _percentiles(samples: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_mean_ci(
    data: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for the mean of `data`."""
    values = np.asarray(data, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("data must not be empty")
    estimate = float(values.mean())
    if values.size == 1:
        return Interval(estimate, estimate, estimate, confidence)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[idx].mean(axis=1)
    low, high = _percentiles(means, confidence)
    return Interval(estimate, low, high, confidence)


def paired_delta_ci(
    treatment: np.ndarray,
    baseline: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Interval for `mean(treatment - baseline)` over matched episodes.

    Episode `i` of both arrays must come from the same initial state and the
    same environment RNG stream, which is what the rollout harness guarantees
    for a fixed seed.
    """
    a = np.asarray(treatment, dtype=np.float64).ravel()
    b = np.asarray(baseline, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    return bootstrap_mean_ci(a - b, n_resamples, confidence, seed)


def paired_relative_delta_ci(
    treatment: np.ndarray,
    baseline: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = 0,
) -> Interval:
    """Interval for the *relative* change, `mean(t)/mean(b) - 1`.

    The ratio is recomputed inside each resample rather than derived from the
    interval of the absolute difference: a ratio of means is not a linear
    function of the pair, so propagating the absolute interval would misstate it.
    """
    a = np.asarray(treatment, dtype=np.float64).ravel()
    b = np.asarray(baseline, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("data must not be empty")

    def ratio(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        denominator = np.where(np.abs(y) > 0, y, np.nan)
        return x / denominator - 1.0

    estimate = float(ratio(np.array([a.mean()]), np.array([b.mean()]))[0])
    if a.size == 1:
        return Interval(estimate, estimate, estimate, confidence)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_resamples, a.size))
    ratios = ratio(a[idx].mean(axis=1), b[idx].mean(axis=1))
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return Interval(estimate, float("nan"), float("nan"), confidence)
    low, high = _percentiles(ratios, confidence)
    return Interval(estimate, low, high, confidence)


def wilson_interval(
    successes: int,
    trials: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Interval:
    """Wilson score interval for a proportion.

    Used for failure rate. The normal approximation is not usable here: failure
    rates in this benchmark are routinely exactly 0 or exactly 1, where it gives
    a zero-width interval and claims certainty from a finite sample.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} outside 0..{trials}")

    # Two-sided normal quantile, computed without pulling in scipy.
    z = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = (z / denominator) * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return Interval(
        p, float(max(0.0, center - margin)), float(min(1.0, center + margin)), confidence
    )


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF, Acklam's rational approximation.

    Accurate to about 1e-9 over the range that matters here, which is far more
    than a confidence level needs, and avoids a scipy dependency in a module
    that is otherwise pure NumPy.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]

    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return float(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > p_high:
        q = np.sqrt(-2 * np.log(1 - p))
        return float(
            -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return float(
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
