"""Figure generation.

The figures are published artifacts, so the tests check the two things that
would silently produce a wrong picture: dropping a point that cannot be drawn on
a log scale, and marking a censored frontier as if it were a measured crossing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from quant_control_bench.plots.figures import (
    SCHEME_ORDER,
    amplification_points,
    load_json,
    plot_frontier,
    plot_frontier_curves,
    plot_sweep,
)

pytest.importorskip("matplotlib")


def _sweep_row(scheme: str, drop: float, ol: float) -> dict[str, Any]:
    return {
        "scheme": scheme,
        "quantization": {"mean_bits_per_weight": 8.0},
        "mean_return": 100.0 * (1.0 - drop),
        "relative_delta_return": {
            "estimate": -drop,
            "ci_low": -drop * 1.1,
            "ci_high": -drop * 0.9,
        },
        "failure_rate": {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.01},
        "open_loop_error": {"normalized_rms": ol},
        "divergence_horizon": {"median": 500.0, "horizon": 1000},
        "amplification_factor": (drop / ol) if ol else float("inf"),
    }


def _sweep_payload() -> dict[str, Any]:
    return {
        "env": "TestEnv",
        "schemes": [
            _sweep_row("fp32", 0.0, 0.0),
            _sweep_row("int8-channel", 0.01, 0.002),
            _sweep_row("ternary", 0.7, 0.04),
        ],
    }


def _frontier_payload(axis: str, censored: bool = False) -> dict[str, Any]:
    rates = [1.0, 1.0, 0.9, 0.8] if censored else [1.0, 0.9, 0.4, 0.1]
    return {
        "env": "TestEnv",
        "frontiers": [
            {
                "scheme": "fp32",
                "axis": axis,
                "p50": 3.0,
                "censored": censored,
                "p50_ci": {"estimate": 3.0, "ci_low": 2.5, "ci_high": 3.5},
                "points": [
                    {
                        "magnitude": float(i),
                        "success_rate": r,
                        "mean_return": 100.0 * r,
                        "failure_rate": 1.0 - r,
                    }
                    for i, r in enumerate(rates)
                ],
            }
        ],
    }


def test_sweep_figure_is_written(tmp_path: Path) -> None:
    out = plot_sweep(_sweep_payload(), tmp_path / "sweep.png")
    assert out.exists() and out.stat().st_size > 0


def test_amplification_panel_drops_the_unplottable_baseline() -> None:
    """fp32 has zero error and zero drop; on log axes it has no position.

    Matplotlib would place it at the clipped edge, where it reads as a measured
    point sitting on the `A = 1` diagonal. It must be absent instead.
    """
    points = amplification_points(_sweep_payload()["schemes"])
    assert [p[0] for p in points] == ["int8-channel", "ternary"]


def test_amplification_panel_drops_a_scheme_that_beat_the_baseline() -> None:
    """A negative drop has no position on a log axis either.

    Noise can put a quantized policy marginally above fp32. Clipping it to the
    axis edge would show a fabricated point; it is dropped.
    """
    rows = [_sweep_row("fp16", -0.002, 1e-5)]
    assert amplification_points(rows) == []


def test_frontier_figures_are_written(tmp_path: Path) -> None:
    payloads = [_frontier_payload("push_impulse"), _frontier_payload("obs_noise", censored=True)]
    bars = plot_frontier(payloads, tmp_path / "frontier.png")
    curves = plot_frontier_curves(payloads, tmp_path / "curves.png")
    assert bars.exists() and bars.stat().st_size > 0
    assert curves.exists() and curves.stat().st_size > 0


def test_frontier_rejects_unknown_axes(tmp_path: Path) -> None:
    payload = {"env": "TestEnv", "frontiers": [{"scheme": "fp32", "axis": "gravity", "p50": 1.0}]}
    with pytest.raises(ValueError, match="no known perturbation axes"):
        plot_frontier([payload], tmp_path / "frontier.png")


def test_scheme_order_covers_the_registry() -> None:
    """A scheme missing from the order would sort last and change colour."""
    from quant_control_bench.quantize import SCHEMES

    assert set(SCHEMES) == set(SCHEME_ORDER)


def test_load_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"env": "X"}))
    assert load_json(path)["env"] == "X"


def test_decreasing_axis_is_reoriented_so_taller_means_better() -> None:
    """Raw `P50` reads backwards on a downward-swept axis.

    `ternary` cannot walk at all and gives up at friction 1.0, so its raw `P50`
    is the largest number in the friction panel — the tallest bar, reading as
    the most robust scheme. Reorientation must put it at the bottom.
    """
    from quant_control_bench.plots.figures import bar_value

    fp32_raw, ternary_raw = 0.189, 1.0
    assert ternary_raw > fp32_raw
    assert bar_value("friction_scale", ternary_raw) < bar_value("friction_scale", fp32_raw)


def test_increasing_axes_keep_raw_p50_to_match_their_labels() -> None:
    """The panels are labelled "P50 — <axis>", so they must show `P50`.

    Reorienting these to distance-from-nominal would put 1.811 under a label
    reading "torso mass scale" where the measured crossing was 2.811. Retention
    ratios use `tolerated_magnitude` instead; the two quantities are not
    interchangeable.
    """
    from quant_control_bench.plots.figures import bar_value

    for axis in ("push_impulse", "mass_scale", "actuator_delay", "obs_noise"):
        assert bar_value(axis, 0.42) == 0.42


def test_plot_orientation_and_retention_orientation_are_distinct() -> None:
    from quant_control_bench.eval.frontier import tolerated_magnitude
    from quant_control_bench.plots.figures import bar_value

    # Same axis, same crossing, two different quantities on purpose.
    assert bar_value("mass_scale", 2.811) == pytest.approx(2.811)
    assert tolerated_magnitude("mass_scale", 2.811) == pytest.approx(1.811)
    # On the decreasing axis both agree, because nominal - P50 is the tolerance.
    assert bar_value("friction_scale", 0.189) == pytest.approx(0.811)
    assert tolerated_magnitude("friction_scale", 0.189) == pytest.approx(0.811)


def test_reorientation_keeps_the_interval_ordered() -> None:
    """Flipping the axis swaps which bound is the lower one."""
    from quant_control_bench.plots.figures import bar_value

    low, high = bar_value("friction_scale", 0.252), bar_value("friction_scale", 0.230)
    assert low < high
