"""The precision recommender — the prescriptive deliverable.

Given a robustness requirement and a per-step latency budget, name the cheapest
precision that meets both, or say that none does.

Shape follows Happynood/quant-reasoning-bench (src/quantthink/budget/frontier.py):
a constraint object, a feasibility filter, then a single selection among the
survivors, with an explicit empty result rather than a fallback. Adapted here
because the constraint is robustness retention across several perturbation axes
rather than a single VRAM number, and because "cheapest" is bits per weight
rather than memory.

Two design decisions worth stating.

**Retention is measured on tolerated perturbation, not on raw `P50`.**
`friction_scale` is swept downward, so its raw `P50` rises as a policy gets
worse; a ratio of raw `P50` values would rank the worst scheme highest on that
axis. The ratio is taken on distance from nominal — see
:func:`quant_control_bench.eval.frontier.tolerated_magnitude`.

**The requirement binds on the worst axis, not the average.** A controller that
keeps 100% of its push robustness and 40% of its friction robustness has not
"retained 70%"; it has a friction problem. Averaging would let a scheme buy its
way past a real failure with an axis nobody was worried about.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from quant_control_bench.eval.frontier import tolerated_magnitude

BASELINE_SCHEME = "fp32"


@dataclass(frozen=True)
class Requirement:
    """What the deployment needs.

    `retain` is the fraction of the fp32 baseline's tolerated perturbation that
    must survive on *every* requested axis. `max_latency_ms` is the per-step
    browser inference budget; `None` leaves latency unconstrained.
    """

    retain: float = 0.9
    max_latency_ms: float | None = None
    axes: tuple[str, ...] | None = None
    # When true, the *lower* end of the retention interval must clear the bar, so
    # the recommendation survives sampling error rather than merely being the
    # most likely outcome. Off by default because it is a strictly stronger claim
    # than the design asks for, and it should be an explicit choice.
    conservative: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.retain <= 1.0:
            raise ValueError(f"retain must be in (0, 1], got {self.retain}")
        if self.max_latency_ms is not None and self.max_latency_ms <= 0.0:
            raise ValueError(f"max_latency_ms must be positive, got {self.max_latency_ms}")


@dataclass
class AxisRetention:
    axis: str
    p50: float
    tolerated: float
    baseline_tolerated: float
    retained: float
    retained_low: float | None
    censored: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    scheme: str
    bits_per_weight: float
    mean_return: float | None
    relative_return_delta: float | None
    latency_ms: float | None
    axes: list[AxisRetention] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.rejections

    @property
    def worst_retained(self) -> float:
        """Retention on the binding axis. See the module docstring."""
        return min((a.retained for a in self.axes), default=float("nan"))

    def to_json(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "bits_per_weight": self.bits_per_weight,
            "mean_return": self.mean_return,
            "relative_return_delta": self.relative_return_delta,
            "latency_ms": self.latency_ms,
            "worst_retained": self.worst_retained,
            "feasible": self.feasible,
            "rejections": list(self.rejections),
            "axes": [a.to_json() for a in self.axes],
        }


@dataclass
class Recommendation:
    requirement: Requirement
    winner: Candidate | None
    candidates: list[Candidate]
    infeasible_reason: str | None = None

    @property
    def infeasible(self) -> bool:
        return self.winner is None

    def to_json(self) -> dict[str, Any]:
        return {
            "requirement": asdict(self.requirement),
            "infeasible": self.infeasible,
            "infeasible_reason": self.infeasible_reason,
            "recommended": self.winner.to_json() if self.winner else None,
            "candidates": [c.to_json() for c in self.candidates],
        }


def _index_frontiers(payloads: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_scheme: dict[str, dict[str, dict[str, Any]]] = {}
    for payload in payloads:
        for row in payload["frontiers"]:
            by_scheme.setdefault(row["scheme"], {})[row["axis"]] = row
    return by_scheme


def _index_sweep(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if payload is None:
        return {}
    return {row["scheme"]: row for row in payload["schemes"]}


def recommend(
    frontier_payloads: list[dict[str, Any]],
    requirement: Requirement,
    sweep_payload: dict[str, Any] | None = None,
    latency_ms: dict[str, float] | None = None,
) -> Recommendation:
    """Cheapest scheme meeting `requirement`, or an explicit infeasible result.

    `latency_ms` maps scheme to measured per-step browser latency. It is only
    consulted when the requirement sets a budget, and a requested budget with no
    measurement available is a hard error rather than a silently dropped
    constraint — a recommendation that quietly ignored the latency budget would
    be worse than no recommendation.
    """
    frontiers = _index_frontiers(frontier_payloads)
    sweep = _index_sweep(sweep_payload)
    latency = latency_ms or {}

    if BASELINE_SCHEME not in frontiers:
        raise ValueError(
            f"the frontier results contain no {BASELINE_SCHEME!r} row; retention is "
            "measured against the baseline and cannot be computed without it"
        )

    if requirement.max_latency_ms is not None and not latency:
        raise ValueError(
            "a latency budget was requested but no measured browser latency was "
            "supplied. Latency must come from a real browser run on the target "
            "machine; it is never estimated here. Drop the budget or pass a "
            "latency measurement."
        )

    axes = list(requirement.axes) if requirement.axes else sorted(frontiers[BASELINE_SCHEME])
    missing = [a for a in axes if a not in frontiers[BASELINE_SCHEME]]
    if missing:
        raise ValueError(f"no baseline frontier measured for axes: {missing}")

    baseline_tolerated = {
        axis: tolerated_magnitude(axis, frontiers[BASELINE_SCHEME][axis]["p50"]) for axis in axes
    }

    candidates: list[Candidate] = []
    for scheme, rows in frontiers.items():
        sweep_row = sweep.get(scheme)
        quant = (sweep_row or {}).get("quantization", {})
        mean_return = (sweep_row or {}).get("mean_return")
        candidate = Candidate(
            scheme=scheme,
            bits_per_weight=float(quant.get("mean_bits_per_weight", float("nan"))),
            mean_return=mean_return,
            relative_return_delta=(sweep_row or {})
            .get("relative_delta_return", {})
            .get("estimate"),
            latency_ms=latency.get(scheme),
        )

        for axis in axes:
            row = rows.get(axis)
            if row is None:
                candidate.rejections.append(f"{axis}: not measured")
                continue

            tolerated = tolerated_magnitude(axis, row["p50"])
            base = baseline_tolerated[axis]
            retained = tolerated / base if base > 0 else float("nan")

            retained_low: float | None = None
            ci = row.get("p50_ci")
            if scheme == BASELINE_SCHEME:
                # The baseline retains exactly all of itself, with no
                # uncertainty: numerator and denominator are the same
                # measurement, so the ratio is identically 1. Feeding fp32's own
                # interval through the generic path instead treats the numerator
                # as uncertain and the denominator as fixed, which invented a
                # spurious spread — under `--conservative` it reported that fp32
                # fails to retain 95% of fp32.
                retained, retained_low = 1.0, 1.0
            elif ci is not None and base > 0:
                # Reorienting flips which interval end is the pessimistic one, so
                # both are mapped and the smaller retention is taken.
                #
                # Caveat, stated rather than hidden: this propagates only the
                # candidate's interval and treats the baseline `P50` as exact, so
                # it understates the true uncertainty of the ratio. A properly
                # paired interval would need the per-episode success flags of
                # both policies resampled together, which the stored frontier
                # results do not carry.
                ends = [
                    tolerated_magnitude(axis, ci["ci_low"]) / base,
                    tolerated_magnitude(axis, ci["ci_high"]) / base,
                ]
                retained_low = min(ends)

            candidate.axes.append(
                AxisRetention(
                    axis=axis,
                    p50=float(row["p50"]),
                    tolerated=float(tolerated),
                    baseline_tolerated=float(base),
                    retained=float(retained),
                    retained_low=retained_low,
                    censored=bool(row.get("censored", False)),
                )
            )

            bound = (
                retained_low if requirement.conservative and retained_low is not None else retained
            )
            if not (bound >= requirement.retain):
                kind = "lower bound" if requirement.conservative else "estimate"
                candidate.rejections.append(
                    f"{axis}: retains {bound:.1%} ({kind}) of baseline, "
                    f"needs {requirement.retain:.1%}"
                )

        if requirement.max_latency_ms is not None:
            if candidate.latency_ms is None:
                candidate.rejections.append("latency: not measured for this scheme")
            elif candidate.latency_ms > requirement.max_latency_ms:
                candidate.rejections.append(
                    f"latency: {candidate.latency_ms:.3f} ms/step exceeds "
                    f"{requirement.max_latency_ms:.3f} ms"
                )

        candidates.append(candidate)

    candidates.sort(key=lambda c: (c.bits_per_weight, -c.worst_retained))

    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        return Recommendation(
            requirement=requirement,
            winner=None,
            candidates=candidates,
            infeasible_reason=(
                f"no scheme retains {requirement.retain:.0%} of the fp32 frontier on every "
                f"requested axis ({', '.join(axes)})"
                + (
                    f" within {requirement.max_latency_ms:.3f} ms/step"
                    if requirement.max_latency_ms is not None
                    else ""
                )
            ),
        )

    # Cheapest first; `bits_per_weight` is the cost, retention breaks ties. fp32
    # is itself a candidate and wins when nothing cheaper qualifies, which is the
    # correct answer rather than a failure to recommend.
    return Recommendation(requirement=requirement, winner=feasible[0], candidates=candidates)


def render(recommendation: Recommendation) -> str:
    """Human-readable form, used by the CLI and mirrored in the Space panel."""
    req = recommendation.requirement
    mode = "lower CI bound" if req.conservative else "point estimate"
    lines = [
        f"Requirement: retain >= {req.retain:.0%} of the fp32 robustness frontier "
        f"on every axis ({mode})"
        + (
            f", latency <= {req.max_latency_ms:.3f} ms/step"
            if req.max_latency_ms is not None
            else ", latency unconstrained"
        )
    ]

    if recommendation.winner is None:
        lines.append(f"INFEASIBLE: {recommendation.infeasible_reason}")
    else:
        w = recommendation.winner
        lines.append(
            f"Recommended: {w.scheme} at {w.bits_per_weight:.2f} bits/weight — "
            f"worst axis retains {w.worst_retained:.1%}"
        )
        for a in w.axes:
            interval = "" if a.retained_low is None else f" (>= {a.retained_low:.1%} at 95%)"
            lines.append(f"    {a.axis:16s} P50 {a.p50:8.3f}  retains {a.retained:6.1%}{interval}")

    lines.append("")
    lines.append("Candidates, cheapest first:")
    for c in recommendation.candidates:
        mark = "OK  " if c.feasible else "no  "
        reason = "" if c.feasible else f"  <- {c.rejections[0]}"
        lines.append(
            f"  {mark}{c.scheme:16s} {c.bits_per_weight:5.2f} bits  "
            f"worst axis {c.worst_retained:6.1%}{reason}"
        )
    return "\n".join(lines)
