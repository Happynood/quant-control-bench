"""Robustness frontier: the perturbation magnitude a policy survives.

`P50` is the magnitude at which the success rate crosses 50%. Reporting it needs
a success criterion, and termination alone is not one: CartpoleBalance never
terminates, so a policy that has completely stopped balancing still shows a 0%
failure rate.

Success is therefore a return threshold — and the threshold comes from the
**fp32 baseline's** unperturbed return, shared across every scheme, never from
the policy's own. Measured on Tier 0, scoring each policy against itself made
`ternary` look like the *most* robust scheme on the observation-noise axis
(P50 0.600 against fp32's 0.275). It was not robust; its unperturbed return had
already collapsed to 269 against fp32's 999, so it was being graded against a
bar less than a third as high. A policy that has already failed must not be able
to pass by failing consistently.

Success also requires surviving the episode — see :func:`episode_success` for
the measurement that forced it. A return threshold alone made the push axis
degenerate, because an impulse arriving late in an episode cannot cost much
return no matter how badly it ends.

The interval on `P50` comes from resampling episodes and re-deriving the whole
crossing, not from propagating the interval of any single magnitude: the
crossing is a nonlinear function of the whole curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_control_bench.eval.perturbations import Perturbation, perturbed_rollout
from quant_control_bench.export.bundle import PolicyBundle
from quant_control_bench.stats.bootstrap import DEFAULT_CONFIDENCE, Interval

# An episode counts as a success if it retains this fraction of the *fp32*
# unperturbed return. See the module docstring for why it is not the policy's own.
DEFAULT_SUCCESS_FRACTION = 0.5

# Magnitude grids per axis. The unperturbed value is always the first point, so
# the curve starts at (approximately) 100% success by construction.
#
# These are calibrated against the *measured* fp32 Tier 1 baseline rather than
# guessed. The first pass guessed, and three of the five axes came out unusable:
# friction bottomed out at 0.3 with the baseline still succeeding 89% of the
# time (censored, no crossing in the grid at all), while mass and observation
# noise put their entire transition inside a single grid interval — 100%->32%
# from one point to the next — so `P50` was an interpolation across one segment
# and could not separate the schemes.
#
# The ranges the design suggests (mass scale in [0.7, 1.3], friction in
# [0.5, 1.5]) do not perturb this policy at all: measured, it succeeds 100% of
# episodes at mass x1.3 and 98% at friction x0.5. MuJoCo Playground trains Go1
# with its own domain randomization, so the trained policy's margin is wider
# than the design assumed. The grids therefore run past the suggested ranges until
# the baseline actually breaks.
#
# Each grid keeps points below the fp32 crossing, because a coarser scheme fails
# earlier and its crossing has to be resolved too.
DEFAULT_GRIDS: dict[str, tuple[float, ...]] = {
    # fp32 crossing measured at ~14 N*s (88% at 8, 37% at 16).
    "push_impulse": (0.0, 2.0, 5.0, 8.0, 11.0, 14.0, 18.0),
    # fp32 crossing measured at ~2.7x (100% at 2.0, 32% at 3.0).
    "mass_scale": (1.0, 1.5, 2.0, 2.3, 2.6, 2.9, 3.2),
    # fp32 still succeeds 89% at 0.3, so this runs down to near-frictionless.
    "friction_scale": (1.0, 0.7, 0.5, 0.3, 0.2, 0.12, 0.06),
    # Integer-valued by construction: `delay = round(magnitude)` control steps,
    # so fractional points are not distinct measurements. Measured, the baseline
    # goes 100% -> 70% -> 3% over delays 0, 1, 2; everything at 3 and beyond is
    # already 0%, so the grid stops at 4 rather than spending rollouts there.
    "actuator_delay": (0.0, 1.0, 2.0, 3.0, 4.0),
    # fp32 crossing measured at ~0.15 (97% at 0.10, 0% at 0.20).
    "obs_noise": (0.0, 0.05, 0.10, 0.13, 0.16, 0.20, 0.30),
}

# A grid is only meaningful for the robot it was calibrated against, and the two
# tiers are not remotely comparable: a cart on a rail has no ground friction to
# scale and tips over under pushes three orders of magnitude smaller than the
# ones a quadruped shrugs off. Running Tier 1's grid on Cartpole would report
# every scheme as failing at the first point; running Cartpole's on Go1 is what
# produced the censored first attempt.
ENV_GRIDS: dict[str, dict[str, tuple[float, ...]]] = {
    "CartpoleBalance": {
        "push_impulse": (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
        "mass_scale": (1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0),
        "friction_scale": (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3),
        "actuator_delay": (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0),
        "obs_noise": (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6),
    },
}


# The unperturbed value of each axis. `P50` is a magnitude on the axis, but
# "how much perturbation did the policy tolerate" is the distance from nominal,
# and only that quantity is comparable across axes swept in opposite directions.
NOMINAL: dict[str, float] = {
    "push_impulse": 0.0,
    "mass_scale": 1.0,
    "friction_scale": 1.0,
    "actuator_delay": 0.0,
    "obs_noise": 0.0,
}

# Axes swept downward from nominal, where a larger `P50` means the policy gave up
# sooner. Kept as a set rather than inferred from the grid so that the direction
# is a stated property of the axis, not an accident of how a grid was written.
DECREASING_AXES = frozenset({"friction_scale"})


def tolerated_magnitude(axis: str, p50: float) -> float:
    """How far from nominal the policy got, as a non-negative distance.

    Necessary because `friction_scale` is swept downward: its raw `P50` of 1.0
    means "failed before friction changed at all", which is the worst possible
    result while also being the largest number. Any comparison across axes — a
    figure, a retained-robustness ratio, a recommendation — has to be made on
    this quantity rather than on raw `P50`.
    """
    nominal = NOMINAL[axis]
    return nominal - p50 if axis in DECREASING_AXES else p50 - nominal


def grid_for(env: str, axis: str) -> tuple[float, ...]:
    """Magnitude grid for one axis on one environment.

    Falls back to the Tier 1 calibration for environments with no entry of their
    own, which is a guess and should be checked against the baseline before its
    `P50` is quoted — see the first-attempt failure described above.
    """
    return ENV_GRIDS.get(env, DEFAULT_GRIDS)[axis]


@dataclass
class FrontierPoint:
    magnitude: float
    success_rate: float
    mean_return: float
    failure_rate: float
    nonfinite_fraction: float
    returns: np.ndarray
    success: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "magnitude": self.magnitude,
            "success_rate": self.success_rate,
            "mean_return": self.mean_return,
            "failure_rate": self.failure_rate,
            "nonfinite_fraction": self.nonfinite_fraction,
        }


def episode_success(returns: np.ndarray, terminated: np.ndarray, threshold: float) -> np.ndarray:
    """An episode succeeds if it finished upright and kept half the baseline return.

    All three conditions are needed, and the middle one was missing at first.

    **Survival.** With a return threshold alone, the push axis was degenerate:
    the impulse lands at a uniformly random step, so an episode pushed at step
    900 has already banked 90% of its return and clears a 50% bar whether or not
    it then falls over. Measured on the fp32 Go1 policy, the success rate
    flattened at ~54% and never crossed 50% — at 16 N·s, at 256 N·s, or anywhere
    between — while the fraction of episodes that actually fell rose from 63% to
    94%. `P50` was therefore undefined on that axis for reasons that had nothing
    to do with robustness. A robot lying on its back is not a success regardless
    of what it earned first.

    **Finiteness.** Extreme perturbations blow the physics up: at 512 N·s the
    mean return came back `nan`. A non-finite return is a failed episode, not a
    missing one, and must not propagate into an average.

    On `CartpoleBalance` this is exactly the old criterion — that task never
    terminates and its returns are finite — so Tier 0 results are unaffected.
    """
    finite = np.isfinite(returns)
    return finite & ~terminated.astype(bool) & (np.where(finite, returns, -np.inf) >= threshold)


@dataclass
class Frontier:
    axis: str
    success_threshold: float
    reference_return: float
    points: list[FrontierPoint]
    p50: float
    p50_ci: Interval | None
    # True when the success rate never fell below 50% within the swept grid, so
    # `p50` is a lower bound rather than a crossing.
    censored: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "success_threshold": self.success_threshold,
            "reference_return": self.reference_return,
            "p50": self.p50,
            "censored": self.censored,
            "p50_ci": self.p50_ci.to_json() if self.p50_ci is not None else None,
            "points": [p.to_json() for p in self.points],
        }


def _crossing(magnitudes: np.ndarray, rates: np.ndarray) -> float:
    """Magnitude where the success rate first drops through 50%.

    Linearly interpolated between the bracketing grid points. If the policy
    never drops below 50% the frontier lies beyond the grid, and the last
    magnitude is returned — censored, not extrapolated, because inventing a
    number past the measured range is exactly what this project forbids.
    """
    below = np.nonzero(rates < 0.5)[0]
    if below.size == 0:
        return float(magnitudes[-1])
    i = int(below[0])
    if i == 0:
        return float(magnitudes[0])

    x0, x1 = float(magnitudes[i - 1]), float(magnitudes[i])
    y0, y1 = float(rates[i - 1]), float(rates[i])
    if y0 == y1:
        return x1
    return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)


def robustness_frontier(
    bundle: PolicyBundle,
    axis: str,
    reference_return: float,
    magnitudes: tuple[float, ...] | None = None,
    success_fraction: float = DEFAULT_SUCCESS_FRACTION,
    num_episodes: int = 100,
    seed: int = 0,
    horizon: int | None = None,
    env: Any = None,
    n_resamples: int = 2000,
    confidence: float = DEFAULT_CONFIDENCE,
    bootstrap_seed: int = 0,
) -> Frontier:
    """Sweep one axis and locate the 50% success crossing.

    `reference_return` is required and must be the fp32 baseline's unperturbed
    return, the same value for every scheme being compared.
    """
    grid = np.asarray(
        magnitudes if magnitudes is not None else grid_for(bundle.env, axis), dtype=float
    )

    reference = float(reference_return)
    threshold = success_fraction * reference

    points: list[FrontierPoint] = []
    for magnitude in grid:
        result = perturbed_rollout(
            bundle,
            Perturbation(axis, float(magnitude)),
            num_episodes=num_episodes,
            seed=seed,
            horizon=horizon,
            env=env,
        )
        successes = episode_success(result.episode_return, result.terminated, threshold)
        finite = np.isfinite(result.episode_return)
        points.append(
            FrontierPoint(
                magnitude=float(magnitude),
                success_rate=float(successes.mean()),
                # Averaged over the finite episodes only. A single blown-up
                # episode would otherwise turn the whole cell into `nan` and
                # hide the ones that were measured; the count is reported
                # alongside so the average is never read without it.
                mean_return=float(result.episode_return[finite].mean())
                if finite.any()
                else float("nan"),
                failure_rate=result.failure_rate,
                nonfinite_fraction=float((~finite).mean()),
                returns=result.episode_return,
                success=successes,
            )
        )

    rates = np.array([p.success_rate for p in points])
    p50 = _crossing(grid, rates)
    censored = bool(np.all(rates >= 0.5))

    # Resample episodes jointly across magnitudes: the same episode index is a
    # different initial state at each magnitude, but resampling independently
    # per magnitude would let the curve cross 50% from sampling noise alone.
    # Resamples the per-episode success flags, not the returns: the interval has
    # to be built from the same criterion as the point estimate, or it would
    # describe a curve that was never plotted.
    rng = np.random.default_rng(bootstrap_seed)
    stacked = np.stack([p.success for p in points])
    n_episodes = stacked.shape[1]
    idx = rng.integers(0, n_episodes, size=(n_resamples, n_episodes))
    crossings = np.empty(n_resamples)
    for i in range(n_resamples):
        resampled = stacked[:, idx[i]].mean(axis=1)
        crossings[i] = _crossing(grid, resampled)

    alpha = 1.0 - confidence
    low, high = np.percentile(crossings, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ci = Interval(p50, float(low), float(high), confidence)

    return Frontier(
        axis=axis,
        success_threshold=threshold,
        reference_return=reference,
        points=points,
        p50=p50,
        p50_ci=ci,
        censored=censored,
    )


def format_frontier_table(payloads: list[dict[str, Any]], schemes: list[str] | None = None) -> str:
    """Markdown `P50` table across axes, for the README and the model card.

    One row per scheme, one column per axis, each cell `P50 [low, high]`. Axis
    direction is *not* normalized here — `friction_scale` is swept downward, so a
    larger number in that column means the policy gave up sooner. The figure
    reorients it; a table of numbers is read against its header, so the raw
    quantity the design defines is what gets published.
    """
    by_scheme: dict[str, dict[str, dict[str, Any]]] = {}
    axes: list[str] = []
    for payload in payloads:
        for row in payload["frontiers"]:
            axis = row["axis"]
            if axis not in axes:
                axes.append(axis)
            by_scheme.setdefault(row["scheme"], {})[axis] = row

    order = schemes if schemes is not None else list(by_scheme)
    lines = [
        "| scheme | " + " | ".join(f"`{a}`" for a in axes) + " |",
        "|---" * (len(axes) + 1) + "|",
    ]
    for scheme in order:
        rows = by_scheme.get(scheme)
        if rows is None:
            continue
        cells = []
        for axis in axes:
            row = rows.get(axis)
            if row is None:
                # A missing axis is a run that did not happen, not a zero.
                cells.append("n/a (not run)")
                continue
            ci = row["p50_ci"]
            text = f"{row['p50']:.3f}"
            if ci is not None:
                text += f" [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            if row["censored"]:
                text = "≥ " + text + " (censored)"
            cells.append(text)
        lines.append(f"| `{scheme}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)
