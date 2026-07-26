"""Command line interface.

CLI shape adapted from Happynood/quant-reasoning-bench (src/quantthink/cli.py):
a click group, one subcommand per pipeline stage, every run writing a result
JSON and a manifest side by side.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import click

from quant_control_bench import __version__
from quant_control_bench.config import load_config
from quant_control_bench.envs import MJX_IMPL, TIER0_ENV, TIER1_ENV
from quant_control_bench.hardware import collect_hardware

# Python<->ONNX parity threshold: ||pi_onnx - pi_jax||_inf < 1e-4.
PARITY_TOL = 1e-4


@click.group()
@click.version_option(__version__, prog_name="qcb")
def main() -> None:
    """quant-control-bench: post-training quantization of closed-loop controllers."""


@main.command("env-check")
def env_check() -> None:
    """Print the accelerator and simulator stack this machine will actually use."""
    hw = collect_hardware()
    click.echo(f"python   : {hw.python_version.split()[0]}")
    click.echo(f"platform : {hw.platform_info}")
    click.echo(f"cpu      : {hw.cpu_model} ({hw.cpu_count} threads)")
    if hw.gpu is None:
        click.echo("gpu      : none detected")
    else:
        click.echo(f"gpu      : {hw.gpu.name} ({hw.gpu.vram_total_mb} MiB)")
        click.echo(f"driver   : {hw.gpu.driver_version}  cuda {hw.gpu.cuda_version}")
        click.echo(f"jax      : {hw.gpu.jax_devices} backend={hw.gpu.jax_default_backend}")
    click.echo(f"mjx impl : {MJX_IMPL}")
    click.echo(f"tier 0   : {TIER0_ENV}")
    click.echo(f"tier 1   : {TIER1_ENV}")


@main.command("smoke")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--manifest", "manifest_path", default=None, type=click.Path())
@click.option("--steps", default=100, show_default=True)
@click.option("--num-envs", default=64, show_default=True)
def smoke(
    config_path: str,
    output_path: str,
    manifest_path: str | None,
    steps: int,
    num_envs: int,
) -> None:
    """Run the end-to-end smoke pipeline and write a result JSON."""
    from quant_control_bench.eval.smoke import run_smoke, write_result

    cfg = load_config(config_path)
    result = run_smoke(cfg, config_path, steps=steps, num_envs=num_envs, seed=cfg.train.seed)
    write_result(result, output_path)

    if manifest_path:
        from quant_control_bench.manifest import collect_manifest, write_manifest

        write_manifest(collect_manifest(config_path, cfg), manifest_path)

    if not result["onnx_parity_ok"]:
        raise click.ClickException(
            f"ONNX parity failed: {result['onnx_parity_max_abs']:.3e} >= {PARITY_TOL:.0e}"
        )
    click.echo(
        f"smoke OK: env={result['env']} return={result['mean_return']:.3f} "
        f"failure={result['failure_rate']:.1%} "
        f"onnx_parity={result['onnx_parity_max_abs']:.2e} "
        f"vram={result['vram_used_mb']} MiB"
    )


@main.command("train")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--out", "out_dir", required=True, type=click.Path())
@click.option("--timesteps", type=int, default=None, help="override the tuned budget")
@click.option("--num-envs", type=int, default=None, help="override the config's num_envs")
@click.option("--checkpoints/--no-checkpoints", default=True, show_default=True)
@click.option(
    "--restore-from",
    type=click.Path(exists=True),
    default=None,
    help="continue from a checkpoint step directory (weights only; the optimizer "
    "state and step counter are not restored)",
)
def train(
    config_path: str,
    out_dir: str,
    timesteps: int | None,
    num_envs: int | None,
    checkpoints: bool,
    restore_from: str | None,
) -> None:
    """Train a policy with Playground's tuned PPO config and export the bundle."""
    from quant_control_bench.manifest import collect_manifest, write_manifest
    from quant_control_bench.train.ppo import train_policy

    cfg = load_config(config_path)
    if timesteps is not None:
        cfg.train.num_timesteps = timesteps
    if num_envs is not None:
        cfg.train.num_envs = num_envs

    out = Path(out_dir)
    click.echo(f"training {cfg.env} seed={cfg.train.seed} num_envs={cfg.train.num_envs}")
    if restore_from:
        click.echo(f"restoring weights from {restore_from}")
    _, record = train_policy(
        cfg,
        out,
        checkpoint_dir=(out / "checkpoints") if checkpoints else None,
        restore_from=restore_from,
    )
    write_manifest(collect_manifest(config_path, cfg), out / "manifest.json")

    click.echo(
        f"trained: reward={record.final_reward:.3f} +/- {record.final_reward_std:.3f} "
        f"in {record.wall_clock_s / 60:.1f} min, peak vram {record.peak_vram_mb} MiB"
    )
    click.echo(f"extraction max |delta action| vs brax: {record.extraction_max_abs_error:.2e}")
    click.echo(
        f"timesteps: requested {record.num_timesteps_requested:,}, "
        f"executed {record.num_timesteps_actual:,}, tuned {record.num_timesteps_tuned:,}"
    )
    if record.budget_was_reduced:
        click.echo(
            f"NOTE: trained for {record.num_timesteps_actual:,} steps, below the tuned "
            f"{record.num_timesteps_tuned:,}",
            err=True,
        )


@main.command("export")
@click.option("--policy", "policy_dir", required=True, type=click.Path(exists=True))
@click.option("--out", "onnx_path", required=True, type=click.Path())
@click.option("--samples", default=1000, show_default=True)
def export(policy_dir: str, onnx_path: str, samples: int) -> None:
    """Export a policy bundle to ONNX and check parity before writing it off."""
    import numpy as np

    from quant_control_bench.export.bundle import PolicyBundle
    from quant_control_bench.export.onnx_export import OPSET, export_onnx, run_onnx

    bundle = PolicyBundle.load(policy_dir)
    path = export_onnx(bundle, onnx_path)

    rng = np.random.default_rng(0)
    obs = rng.normal(0.0, 1.0, size=(samples, bundle.obs_dim)).astype(np.float32)
    max_abs = float(np.abs(run_onnx(path, obs) - bundle.act(obs)).max())

    click.echo(f"exported opset {OPSET} -> {path} ({path.stat().st_size:,} bytes)")
    click.echo(f"parity over {samples} observations: max |delta| = {max_abs:.3e}")
    if max_abs >= PARITY_TOL:
        raise click.ClickException(f"ONNX parity failed: {max_abs:.3e} >= {PARITY_TOL:.0e}")


@main.command("quantize")
@click.option("--policy", "policy_dir", required=True, type=click.Path(exists=True))
@click.option("--scheme", "scheme_id", required=True)
@click.option("--out", "out_dir", required=True, type=click.Path())
@click.option("--quantize-obs-norm", is_flag=True, help="also quantize the normalization stats")
@click.option("--calibration-states", default=10_000, show_default=True)
@click.option("--export/--no-export", "do_export", default=True, show_default=True)
def quantize(
    policy_dir: str,
    scheme_id: str,
    out_dir: str,
    quantize_obs_norm: bool,
    calibration_states: int,
    do_export: bool,
) -> None:
    """Apply a quantization scheme to a policy and write the result."""
    import numpy as np

    from quant_control_bench.export.bundle import PolicyBundle
    from quant_control_bench.export.onnx_export import export_onnx, run_onnx
    from quant_control_bench.quantize import apply_scheme, get_scheme

    bundle = PolicyBundle.load(policy_dir)
    scheme = get_scheme(scheme_id)

    states = None
    if scheme.quantizes_activations:
        from quant_control_bench.eval.rollout import collect_states

        click.echo(f"collecting {calibration_states:,} states from the fp32 policy")
        states = collect_states(bundle, num_states=calibration_states)

    quantized, report = apply_scheme(
        bundle, scheme, quantize_obs_norm=quantize_obs_norm, calibration_states=states
    )

    out = Path(out_dir)
    quantized.save(out)
    (out / "quantization.json").write_text(json.dumps(report.to_json(), indent=2) + "\n")

    click.echo(
        f"{scheme_id}: {report.mean_bits_per_weight:.2f} bits/weight over "
        f"{report.weight_count:,} weights"
    )
    for e in report.tensor_errors:
        click.echo(
            f"  {e.name:24s} rms {e.rms_error:.3e} "
            f"({e.relative_rms_error:6.2%} of tensor rms)  zeros {e.zero_fraction:5.1%}"
        )

    if do_export:
        path = export_onnx(quantized, out / f"policy_{scheme_id}.onnx")
        obs = np.random.default_rng(0).normal(size=(1000, bundle.obs_dim)).astype(np.float32)
        max_abs = float(np.abs(run_onnx(path, obs) - quantized.act(obs)).max())
        click.echo(f"exported -> {path.name}, parity max |delta| = {max_abs:.3e}")
        if max_abs >= PARITY_TOL:
            raise click.ClickException(f"ONNX parity failed: {max_abs:.3e} >= {PARITY_TOL:.0e}")


@main.command("eval")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--policy", "policy_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--episodes", type=int, default=None, help="override the config's episode count")
@click.option("--horizon", type=int, default=None, help="override the env's episode length")
def evaluate(
    config_path: str,
    policy_dir: str,
    output_path: str,
    episodes: int | None,
    horizon: int | None,
) -> None:
    """Roll out a policy deterministically and write per-episode metrics."""
    from quant_control_bench.envs import load_env
    from quant_control_bench.eval.rollout import rollout
    from quant_control_bench.eval.smoke import write_result
    from quant_control_bench.export.bundle import PolicyBundle
    from quant_control_bench.manifest import collect_manifest

    cfg = load_config(config_path)
    bundle = PolicyBundle.load(policy_dir)
    n = episodes if episodes is not None else cfg.eval.episodes

    # One env instance is reused across seeds so the MJX model is compiled once.
    env = load_env(bundle.env)

    per_seed = []
    for seed in cfg.eval.seeds:
        result = rollout(bundle, num_episodes=n, seed=seed, horizon=horizon, env=env)
        per_seed.append(result.to_json())
        click.echo(
            f"  seed {seed}: return {result.mean_return:9.3f}  "
            f"failure {result.failure_rate:5.1%}  jitter {result.mean_jitter:.4f}"
        )

    returns = [s["mean_return"] for s in per_seed]
    payload = {
        "policy": str(policy_dir),
        "env": bundle.env,
        "scheme": "fp32",
        "episodes_per_seed": n,
        "seeds": list(cfg.eval.seeds),
        "mean_return": float(sum(returns) / len(returns)),
        "per_seed": per_seed,
        "manifest": asdict(collect_manifest(config_path, cfg)),
    }
    write_result(payload, output_path)
    click.echo(f"eval OK: mean return over {len(returns)} seeds = {payload['mean_return']:.3f}")


@main.command("sweep")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--policy", "baseline_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option(
    "--table", "table_path", default=None, type=click.Path(), help="write a markdown table"
)
@click.option("--episodes", type=int, default=None)
@click.option("--horizon", type=int, default=None)
@click.option("--divergence-starts", type=int, default=100, show_default=True)
@click.option("--buffer-size", type=int, default=None)
@click.option("--quantize-obs-norm", is_flag=True)
def sweep(
    config_path: str,
    baseline_dir: str,
    output_path: str,
    table_path: str | None,
    episodes: int | None,
    horizon: int | None,
    divergence_starts: int,
    buffer_size: int | None,
    quantize_obs_norm: bool,
) -> None:
    """Every scheme against the fp32 baseline, with bootstrap intervals."""
    from quant_control_bench.eval.sweep import format_table, run_sweep, write_sweep

    cfg = load_config(config_path)
    payload = run_sweep(
        cfg,
        config_path,
        baseline_dir,
        episodes=episodes,
        horizon=horizon,
        divergence_starts=divergence_starts,
        buffer_size=buffer_size,
        quantize_obs_norm=quantize_obs_norm,
    )
    write_sweep(payload, output_path)

    table = format_table(payload)
    if table_path:
        Path(table_path).parent.mkdir(parents=True, exist_ok=True)
        Path(table_path).write_text(table + "\n")
    click.echo("")
    click.echo(table)
    click.echo(f"\nwrote {output_path}")


@main.command("frontier")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--policy", "baseline_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_path", required=True, type=click.Path())
@click.option("--axis", "axes", multiple=True, help="repeatable; defaults to the config's list")
@click.option("--episodes", type=int, default=100, show_default=True)
@click.option("--horizon", type=int, default=None)
def frontier(
    config_path: str,
    baseline_dir: str,
    output_path: str,
    axes: tuple[str, ...],
    episodes: int,
    horizon: int | None,
) -> None:
    """Robustness frontier (P50) per scheme, per perturbation axis."""
    from quant_control_bench.envs import load_env
    from quant_control_bench.eval.frontier import robustness_frontier
    from quant_control_bench.eval.perturbations import AXES, NONE, perturbed_rollout
    from quant_control_bench.eval.sweep import write_sweep
    from quant_control_bench.export.bundle import PolicyBundle
    from quant_control_bench.manifest import collect_manifest
    from quant_control_bench.quantize import apply_scheme, get_scheme

    cfg = load_config(config_path)
    baseline = PolicyBundle.load(baseline_dir)
    env = load_env(baseline.env)
    selected = list(axes) if axes else (list(cfg.perturbations) or list(AXES))

    # One reference return for every scheme and axis. Scoring a policy against
    # its own unperturbed return lets a collapsed policy pass by failing
    # consistently — measured on Tier 0, that ranked `ternary` most robust.
    reference = perturbed_rollout(
        baseline, NONE, num_episodes=episodes, horizon=horizon, env=env
    ).mean_return
    click.echo(f"fp32 reference return: {reference:.3f}")

    # Collected once, only if a selected scheme quantizes activations.
    calibration = None
    if any(get_scheme(s).quantizes_activations for s in cfg.schemes):
        from quant_control_bench.eval.rollout import collect_states

        click.echo(f"collecting {cfg.eval.replay_buffer_size:,} calibration states")
        calibration = collect_states(baseline, num_states=cfg.eval.replay_buffer_size, env=env)

    rows = []
    for scheme_id in cfg.schemes:
        quantized, _ = apply_scheme(baseline, scheme_id, calibration_states=calibration)
        for axis in selected:
            result = robustness_frontier(
                quantized,
                axis,
                reference_return=reference,
                num_episodes=episodes,
                horizon=horizon,
                env=env,
            )
            rows.append({"scheme": scheme_id, **result.to_json()})
            ci = result.p50_ci
            mark = " (censored)" if result.censored else ""
            click.echo(
                f"  {scheme_id:16s} {axis:15s} P50={result.p50:8.3f} "
                f"[{ci.low:.3f}, {ci.high:.3f}]{mark}"
                if ci
                else f"  {scheme_id:16s} {axis:15s} P50={result.p50:8.3f}{mark}"
            )

    write_sweep(
        {
            "env": baseline.env,
            "baseline_policy": str(baseline_dir),
            "reference_return": reference,
            "episodes": episodes,
            "axes": selected,
            "frontiers": rows,
            "manifest": asdict(collect_manifest(config_path, cfg)),
        },
        output_path,
    )
    click.echo(f"wrote {output_path}")


@main.command("publish-bundle")
@click.option("--out", "out_dir", default="publish", show_default=True, type=click.Path())
def publish_bundle(out_dir: str) -> None:
    """Assemble the HF model repo and Space directories from measured artifacts."""
    from quant_control_bench.publish import (
        MODEL_REPO,
        SPACE_REPO,
        build_model_repo,
        build_space,
    )

    root = Path.cwd()
    out = Path(out_dir)
    manifest = build_model_repo(root, out / "model")
    build_space(root, out / "space")

    click.echo(f"model repo -> {out / 'model'}  ({MODEL_REPO})")
    click.echo(f"  training: {manifest['training']['num_timesteps_executed']:,} steps, ")
    click.echo(f"  variants: {len(manifest['artifacts']['onnx'])}")
    click.echo(f"space      -> {out / 'space'}  ({SPACE_REPO})")
    click.echo("")
    click.echo("Upload with:")
    click.echo(
        f"  unset ALL_PROXY all_proxy && hf upload {MODEL_REPO} {out / 'model'} . --repo-type model"
    )
    click.echo(
        f"  unset ALL_PROXY all_proxy && hf upload {SPACE_REPO} {out / 'space'} . --repo-type space"
    )


@main.command("export-scene")
@click.option("--env", "env_name", default=TIER1_ENV, show_default=True)
@click.option("--out", "out_dir", required=True, type=click.Path())
def export_scene(env_name: str, out_dir: str) -> None:
    """Export the scene XML, meshes and post-parse overrides for the browser."""
    from quant_control_bench.export.scene import bundle_scene

    bundle = bundle_scene(env_name, out_dir)
    click.echo(f"{bundle.env}: {len(bundle.files)} files, {bundle.total_bytes:,} bytes")
    click.echo(f"  top xml     : {bundle.top_xml}")
    click.echo(f"  nq/nv/nu    : {bundle.nq}/{bundle.nv}/{bundle.nu}")
    click.echo(
        f"  timestep    : {bundle.timestep} s, ccd_iterations {bundle.ccd_iterations}, "
        f"{bundle.n_substeps} physics steps per control step"
    )
    click.echo("  overrides   : dof_damping, actuator gain/bias (not present in the XML)")


@main.command("recommend")
@click.option(
    "--frontier",
    "frontier_paths",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="repeatable; one frontier result file per perturbation axis",
)
@click.option("--sweep", "sweep_path", default=None, type=click.Path(exists=True))
@click.option(
    "--latency",
    "latency_path",
    default=None,
    type=click.Path(exists=True),
    help="measured browser latency JSON: {scheme: ms_per_step}. Required if "
    "--max-latency-ms is given; never estimated.",
)
@click.option(
    "--retain",
    default=0.9,
    show_default=True,
    help="fraction of the fp32 robustness frontier that must survive on every axis",
)
@click.option("--max-latency-ms", type=float, default=None)
@click.option("--axis", "axes", multiple=True, help="restrict to these axes")
@click.option(
    "--conservative/--point-estimate",
    default=False,
    show_default=True,
    help="require the lower end of the retention interval to clear the bar",
)
@click.option("--output", "output_path", default=None, type=click.Path())
def recommend_cmd(
    frontier_paths: tuple[str, ...],
    sweep_path: str | None,
    latency_path: str | None,
    retain: float,
    max_latency_ms: float | None,
    axes: tuple[str, ...],
    conservative: bool,
    output_path: str | None,
) -> None:
    """Cheapest precision meeting a robustness and latency budget."""
    from quant_control_bench.eval.sweep import write_sweep
    from quant_control_bench.plots.figures import load_json
    from quant_control_bench.recommend import Requirement, recommend, render

    requirement = Requirement(
        retain=retain,
        max_latency_ms=max_latency_ms,
        axes=tuple(axes) if axes else None,
        conservative=conservative,
    )

    latency = None
    if latency_path:
        payload = load_json(latency_path)
        latency = {k: float(v) for k, v in payload.get("ms_per_step", payload).items()}

    try:
        result = recommend(
            [load_json(p) for p in frontier_paths],
            requirement,
            sweep_payload=load_json(sweep_path) if sweep_path else None,
            latency_ms=latency,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None

    click.echo(render(result))
    if output_path:
        write_sweep(result.to_json(), output_path)
        click.echo(f"\nwrote {output_path}")

    # An infeasible requirement is a valid answer, not a crash. It still exits
    # non-zero so a pipeline that asked for a config it cannot have notices.
    if result.infeasible:
        raise SystemExit(2)


@main.command("plot")
@click.option("--sweep", "sweep_path", default=None, type=click.Path(exists=True))
@click.option(
    "--frontier",
    "frontier_paths",
    multiple=True,
    type=click.Path(exists=True),
    help="repeatable; one result file per perturbation axis",
)
@click.option("--out", "out_dir", default="plots", show_default=True, type=click.Path())
@click.option("--prefix", default="tier1", show_default=True)
@click.option(
    "--frontier-table",
    "frontier_table_path",
    default=None,
    type=click.Path(),
    help="also write the P50 markdown table across axes",
)
def plot(
    sweep_path: str | None,
    frontier_paths: tuple[str, ...],
    out_dir: str,
    prefix: str,
    frontier_table_path: str | None,
) -> None:
    """Render figures from result JSON. Nothing is recomputed or re-run."""
    from quant_control_bench.plots.figures import (
        load_json,
        plot_frontier,
        plot_frontier_curves,
        plot_sweep,
    )

    if not sweep_path and not frontier_paths:
        raise click.ClickException("nothing to plot: pass --sweep and/or --frontier")

    out = Path(out_dir)
    if sweep_path:
        written = plot_sweep(load_json(sweep_path), out / f"{prefix}_sweep.png")
        click.echo(f"wrote {written}")
    if frontier_paths:
        payloads = [load_json(p) for p in frontier_paths]
        click.echo(f"wrote {plot_frontier(payloads, out / f'{prefix}_frontier.png')}")
        click.echo(f"wrote {plot_frontier_curves(payloads, out / f'{prefix}_frontier_curves.png')}")

        if frontier_table_path:
            from quant_control_bench.eval.frontier import format_frontier_table
            from quant_control_bench.plots.figures import SCHEME_ORDER

            table = format_frontier_table(payloads, schemes=list(SCHEME_ORDER))
            path = Path(frontier_table_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(table + "\n")
            click.echo(f"wrote {path}")


@main.command("manifest")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def manifest_cmd(config_path: str) -> None:
    """Print the run manifest for a config without running anything."""
    from quant_control_bench.manifest import collect_manifest

    cfg = load_config(config_path)
    json.dump(asdict(collect_manifest(config_path, cfg)), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
