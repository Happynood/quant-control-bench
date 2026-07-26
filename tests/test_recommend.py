"""The precision recommender.

Synthetic frontiers throughout: the point is the selection logic, and building the
inputs by hand is the only way to test the cases real data does not happen to
contain (an infeasible requirement, a censored axis, a decreasing axis whose
ordering inverts).
"""

from __future__ import annotations

from typing import Any

import pytest

from quant_control_bench.recommend import Requirement, recommend, render


def _frontier(
    axis: str,
    p50_by_scheme: dict[str, float],
    ci_width: float = 0.0,
    censored: bool = False,
) -> dict[str, Any]:
    return {
        "env": "TestEnv",
        "frontiers": [
            {
                "scheme": scheme,
                "axis": axis,
                "p50": p50,
                "censored": censored,
                "p50_ci": {
                    "estimate": p50,
                    "ci_low": p50 - ci_width,
                    "ci_high": p50 + ci_width,
                },
                "points": [],
            }
            for scheme, p50 in p50_by_scheme.items()
        ],
    }


def _sweep(bits: dict[str, float]) -> dict[str, Any]:
    return {
        "env": "TestEnv",
        "schemes": [
            {
                "scheme": scheme,
                "quantization": {"mean_bits_per_weight": b},
                "mean_return": 30.0,
                "relative_delta_return": {"estimate": -0.01},
            }
            for scheme, b in bits.items()
        ],
    }


BITS = _sweep({"fp32": 32.0, "int8-channel": 8.0, "int4-channel": 4.0, "ternary": 1.58})


def test_cheapest_feasible_scheme_wins() -> None:
    frontiers = [
        _frontier("obs_noise", {"fp32": 0.10, "int8-channel": 0.099, "int4-channel": 0.095})
    ]
    result = recommend(frontiers, Requirement(retain=0.9), sweep_payload=BITS)

    assert result.winner is not None
    assert result.winner.scheme == "int4-channel"
    assert not result.infeasible


def test_requirement_binds_on_the_worst_axis_not_the_average() -> None:
    """A scheme that keeps all its push margin and half its friction margin has a
    friction problem, not a 75% score."""
    frontiers = [
        _frontier("push_impulse", {"fp32": 10.0, "int4-channel": 10.0}),
        _frontier("friction_scale", {"fp32": 0.2, "int4-channel": 0.6}),
    ]
    result = recommend(frontiers, Requirement(retain=0.9), sweep_payload=BITS)

    candidate = next(c for c in result.candidates if c.scheme == "int4-channel")
    assert not candidate.feasible
    assert any("friction_scale" in r for r in candidate.rejections)
    assert not any("push_impulse" in r for r in candidate.rejections)
    # Averaging 100% and 50% would have cleared a 90% bar at 75%... and even that
    # would not: the point is that the failing axis alone decides.
    assert result.winner is not None and result.winner.scheme == "fp32"


def test_decreasing_axis_ordering_is_not_inverted() -> None:
    """On `friction_scale` a larger `P50` is worse.

    Taking the ratio on raw `P50` would score the policy that gave up at friction
    0.6 as retaining 300% of a baseline that lasted to 0.2.
    """
    frontiers = [_frontier("friction_scale", {"fp32": 0.2, "int4-channel": 0.6})]
    result = recommend(frontiers, Requirement(retain=0.9), sweep_payload=BITS)

    candidate = next(c for c in result.candidates if c.scheme == "int4-channel")
    # Tolerated reduction: 0.8 for the baseline, 0.4 for the candidate.
    assert candidate.worst_retained == pytest.approx(0.5)


def test_baseline_retains_all_of_itself_with_no_spurious_spread() -> None:
    """fp32's numerator and denominator are the same measurement.

    Propagating its own interval as if the denominator were exact invented a
    spread, and under `--conservative` it reported that fp32 fails to retain 95%
    of fp32.
    """
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int8-channel": 0.099}, ci_width=0.02)]
    result = recommend(frontiers, Requirement(retain=0.95, conservative=True), sweep_payload=BITS)

    baseline = next(c for c in result.candidates if c.scheme == "fp32")
    assert baseline.worst_retained == pytest.approx(1.0)
    assert baseline.axes[0].retained_low == pytest.approx(1.0)
    assert baseline.feasible


def test_conservative_mode_is_strictly_harder_than_the_point_estimate() -> None:
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.093}, ci_width=0.01)]

    optimistic = recommend(frontiers, Requirement(retain=0.9), sweep_payload=BITS)
    cautious = recommend(frontiers, Requirement(retain=0.9, conservative=True), sweep_payload=BITS)

    assert optimistic.winner is not None and optimistic.winner.scheme == "int4-channel"
    assert cautious.winner is not None and cautious.winner.scheme == "fp32"


def test_retention_alone_can_never_be_infeasible() -> None:
    """A property of the design, asserted so it stays a deliberate one.

    fp32 retains exactly 100% of itself and `retain` is capped at 1.0, so a
    robustness-only requirement always has at least one answer. Infeasibility is
    a latency phenomenon: it means the budget excludes even the baseline.
    """
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.001})]
    for retain in (0.5, 0.9, 1.0):
        for conservative in (False, True):
            result = recommend(
                frontiers,
                Requirement(retain=retain, conservative=conservative),
                sweep_payload=BITS,
            )
            assert not result.infeasible
            assert result.winner is not None and result.winner.scheme == "fp32"


def test_infeasible_when_the_latency_budget_excludes_even_the_baseline() -> None:
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.099})]
    result = recommend(
        frontiers,
        Requirement(retain=0.9, max_latency_ms=0.1),
        sweep_payload=BITS,
        latency_ms={"fp32": 0.5, "int4-channel": 0.4},
    )
    assert result.infeasible
    assert result.infeasible_reason is not None
    assert "0.100 ms/step" in result.infeasible_reason


def test_a_latency_budget_without_a_measurement_is_refused() -> None:
    """Silently dropping the constraint would be worse than not answering.

    Browser latency has to come from a real browser on the target machine.
    """
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.099})]
    with pytest.raises(ValueError, match="no measured browser latency"):
        recommend(frontiers, Requirement(retain=0.9, max_latency_ms=5.0), sweep_payload=BITS)


def test_latency_budget_excludes_schemes_that_miss_it() -> None:
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.099})]
    result = recommend(
        frontiers,
        Requirement(retain=0.9, max_latency_ms=1.0),
        sweep_payload=BITS,
        latency_ms={"fp32": 0.5, "int4-channel": 2.0},
    )
    assert result.winner is not None and result.winner.scheme == "fp32"


def test_a_scheme_with_no_latency_measurement_is_not_assumed_to_pass() -> None:
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.099})]
    result = recommend(
        frontiers,
        Requirement(retain=0.9, max_latency_ms=5.0),
        sweep_payload=BITS,
        latency_ms={"fp32": 0.5},
    )
    candidate = next(c for c in result.candidates if c.scheme == "int4-channel")
    assert any("not measured" in r for r in candidate.rejections)
    assert result.winner is not None and result.winner.scheme == "fp32"


def test_missing_baseline_is_an_error() -> None:
    frontiers = [_frontier("obs_noise", {"int4-channel": 0.099})]
    with pytest.raises(ValueError, match="no 'fp32' row"):
        recommend(frontiers, Requirement(retain=0.9), sweep_payload=BITS)


def test_unmeasured_axis_is_rejected_rather_than_skipped() -> None:
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.099})]
    with pytest.raises(ValueError, match="no baseline frontier measured"):
        recommend(
            frontiers, Requirement(retain=0.9, axes=("obs_noise", "mass_scale")), sweep_payload=BITS
        )


def test_a_scheme_missing_one_axis_is_not_silently_credited() -> None:
    frontiers = [
        _frontier("push_impulse", {"fp32": 10.0, "int4-channel": 9.9}),
        _frontier("obs_noise", {"fp32": 0.10}),
    ]
    result = recommend(frontiers, Requirement(retain=0.9), sweep_payload=BITS)
    candidate = next(c for c in result.candidates if c.scheme == "int4-channel")
    assert any("not measured" in r for r in candidate.rejections)


def test_retain_outside_the_unit_interval_is_rejected() -> None:
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="retain must be in"):
            Requirement(retain=bad)


def test_render_names_the_binding_axis_when_infeasible() -> None:
    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.01})]
    result = recommend(frontiers, Requirement(retain=1.0, axes=("obs_noise",)), sweep_payload=BITS)
    text = render(result)
    assert "obs_noise" in text
    assert "int4-channel" in text


def test_json_round_trips() -> None:
    import json

    frontiers = [_frontier("obs_noise", {"fp32": 0.10, "int4-channel": 0.099})]
    result = recommend(frontiers, Requirement(retain=0.9), sweep_payload=BITS)
    payload = json.loads(json.dumps(result.to_json()))

    assert payload["recommended"]["scheme"] == "int4-channel"
    assert payload["infeasible"] is False
    assert payload["requirement"]["retain"] == pytest.approx(0.9)
