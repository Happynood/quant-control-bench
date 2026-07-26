"""Scheme sweep: every metric for every precision, against one fp32 baseline.

Three things are shared across all schemes on purpose, and sharing them is what
makes the columns comparable:

* **The replay buffer.** Open-loop error is measured on the states the *fp32*
  policy visits. Letting each scheme supply its own states would measure a
  different question per row.
* **The per-DOF scale** used to normalize divergence distances. Recomputing it
  per scheme would rescale the axis each time, and a policy that shakes more
  would appear to diverge less.
* **The evaluation seeds.** Identical seeds mean identical initial states, which
  makes the return comparison paired and lets the interval drop the
  episode-difficulty variance that cancels in the difference.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from quant_control_bench.config import QcbConfig
from quant_control_bench.envs import load_env
from quant_control_bench.eval.metrics import (
    amplification_factor,
    compute_dof_scale,
    divergence_horizon,
    open_loop_error,
)
from quant_control_bench.eval.rollout import RolloutResult, collect_states, rollout
from quant_control_bench.export.bundle import PolicyBundle
from quant_control_bench.manifest import collect_manifest
from quant_control_bench.quantize import apply_scheme, get_scheme
from quant_control_bench.stats.bootstrap import (
    paired_delta_ci,
    paired_relative_delta_ci,
    wilson_interval,
)

BASELINE_SCHEME = "fp32"


def _concat(results: list[RolloutResult], field: str) -> np.ndarray:
    return np.concatenate([getattr(r, field) for r in results])


def evaluate_bundle(
    bundle: PolicyBundle,
    cfg: QcbConfig,
    env: Any,
    horizon: int | None = None,
    episodes: int | None = None,
) -> list[RolloutResult]:
    n = episodes if episodes is not None else cfg.eval.episodes
    return [
        rollout(bundle, num_episodes=n, seed=seed, horizon=horizon, env=env)
        for seed in cfg.eval.seeds
    ]


def run_sweep(
    cfg: QcbConfig,
    config_path: str | Path,
    baseline_dir: str | Path,
    schemes: list[str] | None = None,
    episodes: int | None = None,
    horizon: int | None = None,
    divergence_starts: int = 100,
    buffer_size: int | None = None,
    quantize_obs_norm: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Run every scheme against the fp32 baseline and return a result payload."""
    baseline = PolicyBundle.load(baseline_dir)
    env = load_env(baseline.env)
    scheme_ids = schemes if schemes is not None else list(cfg.schemes)
    if BASELINE_SCHEME not in scheme_ids:
        scheme_ids = [BASELINE_SCHEME, *scheme_ids]

    n_states = buffer_size if buffer_size is not None else cfg.eval.replay_buffer_size
    boot = cfg.bootstrap

    if progress:
        print(f"collecting {n_states:,} states from the fp32 policy", flush=True)
    states = collect_states(baseline, num_states=n_states, env=env)
    dof_scale = compute_dof_scale(baseline, horizon=horizon, env=env)

    baseline_runs = evaluate_bundle(baseline, cfg, env, horizon, episodes)
    baseline_returns = _concat(baseline_runs, "episode_return")
    baseline_mean = float(baseline_returns.mean())
    # Taken from the baseline run rather than from the last loop iteration, so
    # the recorded horizon does not depend on which scheme happened to run last.
    resolved_horizon = baseline_runs[0].horizon

    rows: list[dict[str, Any]] = []
    for scheme_id in scheme_ids:
        scheme = get_scheme(scheme_id)
        quantized, report = apply_scheme(
            baseline,
            scheme,
            quantize_obs_norm=quantize_obs_norm,
            calibration_states=states if scheme.quantizes_activations else None,
        )

        runs = evaluate_bundle(quantized, cfg, env, horizon, episodes)
        returns = _concat(runs, "episode_return")
        terminated = _concat(runs, "terminated")
        jitter = _concat(runs, "action_jitter")

        ol = open_loop_error(quantized, baseline, states)
        div = divergence_horizon(
            quantized,
            baseline,
            dof_scale,
            num_starts=divergence_starts,
            horizon=horizon,
            env=env,
        )

        delta = paired_delta_ci(
            returns, baseline_returns, boot.resamples, boot.confidence, boot.seed
        )
        relative = paired_relative_delta_ci(
            returns, baseline_returns, boot.resamples, boot.confidence, boot.seed
        )
        failures = wilson_interval(int(terminated.sum()), int(terminated.size), boot.confidence)

        # The amplification factor uses the *drop*, so a positive A means the
        # closed loop lost more than the open-loop error alone suggests.
        amplification = amplification_factor(-relative.estimate, ol)

        rows.append(
            {
                "scheme": scheme_id,
                "quantization": report.to_json(),
                "mean_return": float(returns.mean()),
                "delta_return": delta.to_json(),
                "relative_delta_return": relative.to_json(),
                "failure_rate": failures.to_json(),
                "mean_action_jitter": float(jitter.mean()),
                "open_loop_error": ol.to_json(),
                "divergence_horizon": div.to_json(),
                "amplification_factor": amplification,
                "episodes": int(returns.size),
            }
        )

        if progress:
            print(
                f"  {scheme_id:16s} return {float(returns.mean()):9.3f} "
                f"({relative.estimate:+7.3%} [{relative.low:+.3%}, {relative.high:+.3%}])  "
                f"ol_rms {ol.normalized_rms:.3e}  T_div {div.median:6.1f}  "
                f"A {amplification:8.2f}",
                flush=True,
            )

    return {
        "env": baseline.env,
        "baseline_policy": str(baseline_dir),
        "baseline_scheme": BASELINE_SCHEME,
        "baseline_mean_return": baseline_mean,
        "episodes_per_seed": episodes if episodes is not None else cfg.eval.episodes,
        "seeds": list(cfg.eval.seeds),
        "horizon": resolved_horizon,
        "replay_buffer_size": int(states.shape[0]),
        "divergence_starts": divergence_starts,
        "divergence_epsilon": cfg.eval.divergence_eps,
        "quantized_obs_norm": quantize_obs_norm,
        "bootstrap": {
            "resamples": boot.resamples,
            "confidence": boot.confidence,
            "seed": boot.seed,
        },
        "dof_scale": dof_scale.tolist(),
        "schemes": rows,
        "manifest": asdict(collect_manifest(config_path, cfg)),
    }


def write_sweep(payload: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n")


def format_table(payload: dict[str, Any]) -> str:
    """Markdown table of the sweep, for the README and the model card."""
    header = (
        "| scheme | bits/weight | mean return | Δreturn vs fp32 (95% CI) | "
        "failure rate (Wilson 95%) | open-loop RMS | T_div median | A |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for row in payload["schemes"]:
        rel = row["relative_delta_return"]
        fail = row["failure_rate"]
        ol = row["open_loop_error"]
        div = row["divergence_horizon"]
        amp = row["amplification_factor"]
        amp_text = "n/a" if not np.isfinite(amp) else f"{amp:.1f}"
        # `A` divides the return drop by the open-loop error. When the drop's
        # own interval spans zero, the numerator is indistinguishable from zero
        # and the ratio carries no information — on Tier 1 that produced values
        # like -6.4 for fp16, which reads as a measurement and is not one. The
        # raw number stays in the JSON; the published table declines to quote it.
        if rel["ci_low"] <= 0.0 <= rel["ci_high"]:
            amp_text = "n/a (no measurable loss)"

        # A non-finite return means the policy emitted NaN actions — it did not
        # score badly, it stopped being a policy. Printing "nan" in a results
        # column reads as a run that is missing, so the row says what happened
        # and names the cause when the report identified one.
        if not np.isfinite(row["mean_return"]):
            collapsed = row["quantization"].get("collapsed_norm_std", 0)
            cause = (
                f" ({collapsed} of the normalization scales quantized to zero)" if collapsed else ""
            )
            lines.append(
                f"| `{row['scheme']}` "
                f"| {row['quantization']['mean_bits_per_weight']:.2f} "
                f"| **collapsed: NaN actions**{cause} "
                "| — | — | — | — | — |"
            )
            continue

        lines.append(
            f"| `{row['scheme']}` "
            f"| {row['quantization']['mean_bits_per_weight']:.2f} "
            f"| {row['mean_return']:.2f} "
            f"| {rel['estimate']:+.3%} [{rel['ci_low']:+.3%}, {rel['ci_high']:+.3%}] "
            f"| {fail['estimate']:.1%} [{fail['ci_low']:.1%}, {fail['ci_high']:.1%}] "
            f"| {ol['normalized_rms']:.2e} "
            f"| {div['median']:.0f} / {div['horizon']} "
            f"| {amp_text} |"
        )
    return "\n".join(lines)
