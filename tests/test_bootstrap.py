"""Confidence-interval machinery, checked against analytic values.

Every headline delta in this project carries one of these intervals, so they are
tested against what statistics guarantees rather than against recorded output.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_control_bench.stats.bootstrap import (
    Interval,
    _normal_quantile,
    bootstrap_mean_ci,
    paired_delta_ci,
    paired_relative_delta_ci,
    wilson_interval,
)

# ── normal quantile ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "p,expected",
    [(0.975, 1.959963985), (0.995, 2.575829304), (0.95, 1.644853627), (0.5, 0.0)],
)
def test_normal_quantile_matches_published_values(p: float, expected: float) -> None:
    assert abs(_normal_quantile(p) - expected) < 1e-6


def test_normal_quantile_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        _normal_quantile(0.0)


# ── bootstrap mean ────────────────────────────────────────────────────────────


def test_interval_brackets_the_point_estimate() -> None:
    data = np.random.default_rng(0).normal(10.0, 2.0, size=200)
    ci = bootstrap_mean_ci(data, n_resamples=2000, seed=1)
    assert ci.low < ci.estimate < ci.high
    assert ci.estimate == pytest.approx(float(data.mean()))


def test_interval_is_deterministic_for_a_fixed_seed() -> None:
    data = np.random.default_rng(0).normal(size=100)
    a = bootstrap_mean_ci(data, n_resamples=1000, seed=7)
    b = bootstrap_mean_ci(data, n_resamples=1000, seed=7)
    assert (a.low, a.high) == (b.low, b.high)


def test_interval_width_shrinks_roughly_as_one_over_sqrt_n() -> None:
    rng = np.random.default_rng(2)
    small = bootstrap_mean_ci(rng.normal(size=100), n_resamples=4000, seed=1)
    large = bootstrap_mean_ci(rng.normal(size=1600), n_resamples=4000, seed=1)
    # 16x the data should be about 4x tighter; allow generous slack.
    ratio = small.width / large.width
    assert 2.5 < ratio < 6.0


def test_interval_approximates_the_analytic_normal_interval() -> None:
    """For a large normal sample the bootstrap should land near mean +- 1.96 se."""
    data = np.random.default_rng(3).normal(5.0, 1.0, size=4000)
    ci = bootstrap_mean_ci(data, n_resamples=4000, seed=1)

    se = float(data.std(ddof=1) / np.sqrt(data.size))
    analytic_half_width = 1.959964 * se
    assert abs(ci.width / 2 - analytic_half_width) / analytic_half_width < 0.1


def test_coverage_is_close_to_nominal() -> None:
    """The property that makes a 95% interval a 95% interval."""
    rng = np.random.default_rng(4)
    covered = 0
    trials = 200
    for i in range(trials):
        sample = rng.normal(0.0, 1.0, size=80)
        ci = bootstrap_mean_ci(sample, n_resamples=800, seed=i)
        covered += ci.low <= 0.0 <= ci.high
    assert 0.88 < covered / trials < 1.0


def test_single_observation_gives_a_degenerate_interval() -> None:
    ci = bootstrap_mean_ci(np.array([3.0]))
    assert ci.low == ci.high == ci.estimate == 3.0


def test_empty_data_is_rejected() -> None:
    with pytest.raises(ValueError):
        bootstrap_mean_ci(np.array([]))


# ── paired deltas ─────────────────────────────────────────────────────────────


def test_paired_interval_is_tighter_than_ignoring_the_pairing() -> None:
    """The reason paired resampling exists.

    Two policies evaluated from identical initial states share almost all their
    episode-to-episode variance. Treating them as independent samples would
    inflate the interval with variance that cancels in the difference.
    """
    rng = np.random.default_rng(5)
    shared = rng.normal(0.0, 10.0, size=300)  # episode difficulty
    baseline = shared + rng.normal(0.0, 0.1, size=300)
    treatment = shared + rng.normal(0.0, 0.1, size=300) - 1.0

    paired = paired_delta_ci(treatment, baseline, n_resamples=3000, seed=1)
    unpaired_width = (
        bootstrap_mean_ci(treatment, n_resamples=3000, seed=1).width
        + bootstrap_mean_ci(baseline, n_resamples=3000, seed=1).width
    )
    assert paired.width < unpaired_width / 10
    assert paired.low < -1.0 < paired.high


def test_paired_delta_detects_a_real_difference() -> None:
    rng = np.random.default_rng(6)
    baseline = rng.normal(100.0, 5.0, size=400)
    treatment = baseline - 3.0
    ci = paired_delta_ci(treatment, baseline, n_resamples=2000, seed=1)
    assert ci.excludes_zero
    assert ci.estimate == pytest.approx(-3.0)


def test_paired_delta_reports_no_difference_when_there_is_none() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.normal(100.0, 5.0, size=400)
    treatment = baseline.copy()
    ci = paired_delta_ci(treatment, baseline, n_resamples=2000, seed=1)
    assert not ci.excludes_zero


def test_mismatched_pairs_are_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        paired_delta_ci(np.zeros(5), np.zeros(6))


def test_relative_delta_matches_the_ratio_of_means() -> None:
    baseline = np.full(200, 100.0)
    treatment = np.full(200, 90.0)
    ci = paired_relative_delta_ci(treatment, baseline, n_resamples=500, seed=1)
    assert ci.estimate == pytest.approx(-0.10)


def test_relative_delta_brackets_the_truth_on_noisy_data() -> None:
    rng = np.random.default_rng(8)
    baseline = rng.normal(100.0, 3.0, size=500)
    treatment = baseline * 0.9
    ci = paired_relative_delta_ci(treatment, baseline, n_resamples=2000, seed=1)
    assert ci.low < -0.10 < ci.high


# ── Wilson interval ───────────────────────────────────────────────────────────


def test_wilson_matches_a_published_value() -> None:
    # 0 successes in 100 trials, 95%: upper bound 0.0370 (Wilson).
    ci = wilson_interval(0, 100)
    assert ci.estimate == 0.0
    assert ci.low == 0.0
    assert ci.high == pytest.approx(0.0370, abs=5e-4)


def test_wilson_is_not_degenerate_at_the_boundaries() -> None:
    """The reason the normal approximation is not used.

    A failure rate of exactly 0 out of 100 is not proof that failures are
    impossible, and the interval must say so.
    """
    for successes, trials in [(0, 100), (100, 100), (0, 5)]:
        ci = wilson_interval(successes, trials)
        assert ci.width > 0.0


def test_wilson_stays_inside_zero_one() -> None:
    for successes in range(0, 21):
        ci = wilson_interval(successes, 20)
        assert 0.0 <= ci.low <= ci.high <= 1.0


def test_wilson_narrows_with_more_trials() -> None:
    assert wilson_interval(5, 10).width > wilson_interval(500, 1000).width


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


# ── interval helpers ──────────────────────────────────────────────────────────


def test_overlap_detection() -> None:
    a = Interval(1.0, 0.5, 1.5, 0.95)
    b = Interval(2.0, 1.4, 2.6, 0.95)
    c = Interval(5.0, 4.0, 6.0, 0.95)
    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)


def test_json_round_trip_keeps_the_confidence_level() -> None:
    payload = Interval(1.0, 0.5, 1.5, 0.95).to_json()
    assert payload["confidence"] == 0.95
    assert payload["ci_low"] < payload["estimate"] < payload["ci_high"]


# ── sweep table rendering ─────────────────────────────────────────────────────


def _table_row(scheme: str, mean_return: float, collapsed: int = 0) -> dict:
    return {
        "scheme": scheme,
        "quantization": {"mean_bits_per_weight": 4.0, "collapsed_norm_std": collapsed},
        "mean_return": mean_return,
        "relative_delta_return": {"estimate": -0.02, "ci_low": -0.03, "ci_high": -0.01},
        "failure_rate": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.01},
        "open_loop_error": {"normalized_rms": 0.02},
        "divergence_horizon": {"median": 83.0, "horizon": 1000},
        "amplification_factor": 0.9,
    }


def test_collapsed_row_is_not_rendered_as_a_missing_measurement() -> None:
    """A NaN return means the policy stopped being a policy, not that it scored badly.

    Measured on Tier 1: quantizing the normalization statistics to int4 zeroes
    entries of `norm_std`, so every observation divides by zero and the actions
    are NaN. A bare "nan" in a results column reads as a run that never happened.
    """
    from quant_control_bench.eval.sweep import format_table

    table = format_table({"schemes": [_table_row("int4-channel", float("nan"), collapsed=3)]})
    assert "collapsed: NaN actions" in table
    assert "3 of the normalization scales quantized to zero" in table
    assert "nan" not in table.lower().replace("nan actions", "")


def test_finite_row_still_reports_its_numbers() -> None:
    from quant_control_bench.eval.sweep import format_table

    table = format_table({"schemes": [_table_row("int4-channel", 30.91)]})
    assert "30.91" in table
    assert "collapsed" not in table
