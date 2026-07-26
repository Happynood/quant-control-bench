"""Figures generated from result JSON, never from re-run computation.

Every figure here reads a file that a measured run wrote. Nothing is recomputed,
smoothed or fitted, so a plot cannot drift away from the table it illustrates.

Schemes keep a fixed order and a fixed colour across every figure, so the same
precision is the same colour in the sweep panel and in the frontier panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Coarsest last: the reader should scan left-to-right from fp32 down to ternary.
SCHEME_ORDER = (
    "fp32",
    "fp16",
    "int8-tensor",
    "int8-channel",
    "int8-act",
    "mixed-head-fp16",
    "int4-channel",
    "int4-group32",
    "ternary",
)

AXIS_LABELS = {
    "push_impulse": "push impulse (N·s)",
    "mass_scale": "torso mass scale",
    "friction_scale": "ground friction scale",
    "actuator_delay": "actuator delay (control steps)",
    "obs_noise": "observation noise σ",
}

# Plotting raw `P50` for all five axes side by side inverts the reading on the one
# swept downward. `ternary` fails at friction 1.0 — it cannot walk at all — and
# its raw `P50` of 1.00 is the tallest bar in the panel, which reads as the most
# robust scheme. That is the same trap as grading a collapsed policy against its
# own return, in visual form. The bar therefore shows the distance from nominal
# the policy tolerated, so taller is better on every panel. The table in the
# README carries raw `P50` with its interval.
#
# The orientation itself is a property of the axis and lives with the axis code.
BAR_LABELS = {
    "friction_scale": "friction reduction tolerated (1 − P50)",
}


def bar_value(axis: str, p50: float) -> float:
    """`P50` reoriented so that a taller bar always means a more robust policy.

    Deliberately *not* `tolerated_magnitude`. That function returns distance from
    nominal on every axis, which is what a retention ratio needs but not what
    these panels are labelled with: `mass_scale` is labelled "P50 — torso mass
    scale" and must therefore show 2.811, not the 1.811 of tolerated increase.
    Only the downward-swept axis is reoriented, and its label says so.
    """
    from quant_control_bench.eval.frontier import DECREASING_AXES, NOMINAL

    return NOMINAL[axis] - p50 if axis in DECREASING_AXES else p50


def _decreasing_axes() -> frozenset[str]:
    from quant_control_bench.eval.frontier import DECREASING_AXES

    return DECREASING_AXES


def _figure_setup() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _scheme_index(scheme: str) -> int:
    return SCHEME_ORDER.index(scheme) if scheme in SCHEME_ORDER else len(SCHEME_ORDER)


def _colors(schemes: list[str]) -> dict[str, Any]:
    plt = _figure_setup()
    cmap = plt.get_cmap("viridis")
    n = max(len(SCHEME_ORDER) - 1, 1)
    return {s: cmap(_scheme_index(s) / n) for s in schemes}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def amplification_points(rows: list[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """`(scheme, open-loop error, relative return drop)` for the log-log panel.

    Rows with a zero on either axis are dropped rather than clipped. fp32 is the
    usual case: it sits at the origin of both axes by construction, and
    matplotlib would silently park it at the edge of a log scale, where it reads
    as a measured point lying on the `A = 1` diagonal. A scheme that came out
    *better* than the baseline has a negative drop and is dropped for the same
    reason — it has no position on a log axis, and inventing one would be a
    fabricated data point.
    """
    points = []
    for row in rows:
        ol = float(row["open_loop_error"]["normalized_rms"])
        drop = -float(row["relative_delta_return"]["estimate"])
        if ol > 0.0 and drop > 0.0:
            points.append((str(row["scheme"]), ol, drop))
    return points


def plot_sweep(payload: dict[str, Any], path: str | Path) -> Path:
    """Two panels: what each precision costs, and whether the loop amplified it.

    The right panel is the H1 picture. Open-loop action error is on the x axis
    and the closed-loop return drop on the y axis; the dashed diagonal is
    `A = 1`, where the loop passes the error through without amplifying it.
    Points above that line are the phenomenon this benchmark exists to measure.
    """
    plt = _figure_setup()

    rows = sorted(payload["schemes"], key=lambda r: _scheme_index(r["scheme"]))
    names = [r["scheme"] for r in rows]
    colors = _colors(names)

    fig, (ax_return, ax_amp) = plt.subplots(1, 2, figsize=(13, 5))

    drops = [-100.0 * r["relative_delta_return"]["estimate"] for r in rows]
    # Asymmetric error bars: the interval is on the delta, not on the mean, and
    # a symmetric +/- would misstate it whenever the bootstrap is skewed.
    lows = [-100.0 * r["relative_delta_return"]["ci_high"] for r in rows]
    highs = [-100.0 * r["relative_delta_return"]["ci_low"] for r in rows]
    err = np.abs(np.array([np.array(drops) - np.array(lows), np.array(highs) - np.array(drops)]))

    positions = np.arange(len(rows))
    ax_return.bar(
        positions,
        drops,
        yerr=err,
        capsize=4,
        color=[colors[n] for n in names],
        edgecolor="black",
        linewidth=0.5,
    )
    ax_return.axhline(0.0, color="black", linewidth=0.8)
    ax_return.set_xticks(positions)
    ax_return.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax_return.set_ylabel("return lost vs fp32 (%)")
    ax_return.set_title("Cost of precision (bootstrap 95% CI)")
    ax_return.grid(axis="y", alpha=0.3)

    # One collapsed scheme flattens every other bar to invisibility on a linear
    # axis — on Tier 0 `ternary` loses 73% while the rest lose under 0.15%, and
    # the linear plot then reads as "only ternary has any cost at all", which is
    # false. Symlog keeps the small bars legible without hiding the large one.
    finite = np.array([d for d in drops if np.isfinite(d) and d > 0.0])
    if finite.size and finite.max() / finite.min() > 100.0:
        ax_return.set_yscale("symlog", linthresh=float(finite.min()))
        ax_return.set_ylabel("return lost vs fp32 (%, symlog)")

    # Identical schemes land on identical points — `int4-group32` reduces to
    # `int4-channel` whenever a group spans the whole reduction axis — so the
    # labels are staggered rather than drawn on top of each other.
    for i, (scheme, ol, drop) in enumerate(amplification_points(rows)):
        ax_amp.scatter(
            ol, drop, s=90, color=colors[scheme], edgecolor="black", linewidth=0.5, zorder=3
        )
        ax_amp.annotate(
            scheme,
            (ol, drop),
            textcoords="offset points",
            xytext=(8, 4 if i % 2 == 0 else -11),
            fontsize=8,
        )

    ax_amp.set_xscale("log")
    ax_amp.set_yscale("log")
    limits = np.array(ax_amp.get_xlim())
    ax_amp.plot(limits, limits, "--", color="gray", linewidth=1, zorder=1, label="A = 1")
    ax_amp.set_xlim(*limits)
    ax_amp.set_xlabel("open-loop action error (normalized RMS)")
    ax_amp.set_ylabel("relative return drop")
    ax_amp.set_title("Feedback amplification: above the line, the loop made it worse")
    ax_amp.grid(alpha=0.3, which="both")
    ax_amp.legend(loc="upper left", fontsize=9)

    fig.suptitle(f"{payload['env']} — quantization sweep", fontsize=13)
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_frontier(payloads: list[dict[str, Any]], path: str | Path) -> Path:
    """`P50` per scheme, one panel per perturbation axis.

    Censored points — where the success rate never fell through 50% inside the
    swept grid — are drawn hollow with an arrow, because the true frontier lies
    somewhere beyond the last magnitude and the bar is a lower bound.
    """
    plt = _figure_setup()

    by_axis: dict[str, list[dict[str, Any]]] = {}
    env = ""
    for payload in payloads:
        env = payload.get("env", env)
        for row in payload["frontiers"]:
            by_axis.setdefault(row["axis"], []).append(row)

    axes_present = [a for a in AXIS_LABELS if a in by_axis]
    if not axes_present:
        raise ValueError("no known perturbation axes in the supplied frontier results")

    ncols = min(len(axes_present), 3)
    nrows = int(np.ceil(len(axes_present) / ncols))
    fig, grid = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    flat = [ax for row in grid for ax in row]

    names = sorted({r["scheme"] for rows in by_axis.values() for r in rows}, key=_scheme_index)
    colors = _colors(names)

    for ax, axis in zip(flat, axes_present, strict=False):
        rows = sorted(by_axis[axis], key=lambda r: _scheme_index(r["scheme"]))
        positions = np.arange(len(rows))
        raw = np.array([r["p50"] for r in rows])
        raw_low = np.array(
            [r["p50_ci"]["ci_low"] if r["p50_ci"] else v for r, v in zip(rows, raw, strict=True)]
        )
        raw_high = np.array(
            [r["p50_ci"]["ci_high"] if r["p50_ci"] else v for r, v in zip(rows, raw, strict=True)]
        )

        # Reorienting a decreasing axis flips the interval end for end, so the
        # bounds are recomputed and re-sorted rather than carried across.
        values = np.array([bar_value(axis, v) for v in raw])
        bound_a = np.array([bar_value(axis, v) for v in raw_low])
        bound_b = np.array([bar_value(axis, v) for v in raw_high])
        lows = np.minimum(bound_a, bound_b)
        highs = np.maximum(bound_a, bound_b)
        err = np.abs(np.array([values - lows, highs - values]))

        censored = np.array([bool(r["censored"]) for r in rows])
        ax.bar(
            positions,
            values,
            yerr=err,
            capsize=3,
            color=[
                "none" if c else colors[r["scheme"]] for r, c in zip(rows, censored, strict=True)
            ],
            edgecolor=[
                colors[r["scheme"]] if c else "black" for r, c in zip(rows, censored, strict=True)
            ],
            linewidth=[1.6 if c else 0.5 for c in censored],
            hatch=["//" if c else "" for c in censored],
        )
        for x, value, is_censored in zip(positions, values, censored, strict=True):
            if is_censored:
                ax.annotate(
                    "≥",
                    (x, value),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=11,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels([r["scheme"] for r in rows], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(BAR_LABELS.get(axis, f"P50 — {AXIS_LABELS[axis]}"))
        ax.set_title(axis if axis not in _decreasing_axes() else f"{axis} (reoriented)")
        ax.grid(axis="y", alpha=0.3)

    for ax in flat[len(axes_present) :]:
        ax.axis("off")

    fig.suptitle(f"{env} — robustness frontier: taller is more robust on every panel", fontsize=12)
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_frontier_curves(payloads: list[dict[str, Any]], path: str | Path) -> Path:
    """The success-rate curves `P50` is read off, one panel per axis.

    `P50` compresses each curve to a single crossing. This figure shows the
    curves themselves, so a crossing produced by a curve that is already flat
    near zero is visible as such rather than hidden behind a bar.
    """
    plt = _figure_setup()

    by_axis: dict[str, list[dict[str, Any]]] = {}
    env = ""
    for payload in payloads:
        env = payload.get("env", env)
        for row in payload["frontiers"]:
            by_axis.setdefault(row["axis"], []).append(row)

    axes_present = [a for a in AXIS_LABELS if a in by_axis]
    ncols = min(len(axes_present), 3)
    nrows = int(np.ceil(len(axes_present) / ncols))
    fig, grid = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    flat = [ax for row in grid for ax in row]

    names = sorted({r["scheme"] for rows in by_axis.values() for r in rows}, key=_scheme_index)
    colors = _colors(names)

    for ax, axis in zip(flat, axes_present, strict=False):
        rows = sorted(by_axis[axis], key=lambda r: _scheme_index(r["scheme"]))
        for row in rows:
            magnitudes = [p["magnitude"] for p in row["points"]]
            rates = [100.0 * p["success_rate"] for p in row["points"]]
            ax.plot(
                magnitudes,
                rates,
                marker="o",
                markersize=4,
                linewidth=1.4,
                color=colors[row["scheme"]],
                label=row["scheme"],
            )
        ax.axhline(50.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(AXIS_LABELS[axis])
        ax.set_ylabel("success rate (%)")
        ax.set_ylim(-3, 103)
        ax.set_title(axis)
        ax.grid(alpha=0.3)

    for ax in flat[len(axes_present) :]:
        ax.axis("off")

    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=9, ncol=2)
    fig.suptitle(f"{env} — success rate against perturbation magnitude", fontsize=13)
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
