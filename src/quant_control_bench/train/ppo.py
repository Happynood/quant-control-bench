"""PPO training entrypoint.

Brax PPO with MuJoCo Playground's tuned hyperparameters. This module wires them
together, applies the two sanctioned overrides (``num_envs`` for the VRAM
protocol, ``num_timesteps`` for the wall-clock budget), and writes a
:class:`PolicyBundle` plus a training record that states exactly what was run.

Any reduction against the tuned defaults is written into the record. A reduced
budget is allowed; a *silently* reduced budget is not.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quant_control_bench.config import QcbConfig
from quant_control_bench.envs import load_env, policy_obs_key
from quant_control_bench.envs.registry import policy_obs_dim
from quant_control_bench.export.bundle import PolicyBundle
from quant_control_bench.export.from_brax import bundle_from_brax
from quant_control_bench.hardware import gpu_memory_used_mb
from quant_control_bench.train.params import network_kwargs, tuned_ppo_config

# Brax's default policy activation for PPO. Recorded explicitly because the
# ONNX exporter has to reproduce it.
ACTIVATION = "swish"


@dataclass
class EvalPoint:
    step: int
    reward: float
    reward_std: float
    wall_clock_s: float


@dataclass
class TrainingRecord:
    env: str
    seed: int
    num_timesteps_requested: int
    num_timesteps_actual: int
    num_timesteps_tuned: int
    num_envs: int
    num_envs_tuned: int
    episode_length: int
    obs_dim: int
    action_dim: int
    obs_key: str | None
    activation: str
    hidden_sizes: list[int]
    wall_clock_s: float
    peak_vram_mb: int | None
    final_reward: float | None
    final_reward_std: float | None
    extraction_max_abs_error: float
    # Set when the run continued from an earlier checkpoint. Brax restores the
    # normalizer and the network weights but *not* the optimizer state or the
    # step counter, so a resumed run is not equivalent to an uninterrupted one
    # of the same total length, and the provenance has to say so.
    restored_from: str | None = None
    progress: list[EvalPoint] = field(default_factory=list)

    @property
    def budget_was_reduced(self) -> bool:
        """Did this run actually train for less than the tuned budget?

        Compared against what was *executed*, not what was asked for. Brax
        rounds the request up to a whole number of epochs, so the two routinely
        differ by a large factor — see :func:`minimum_trainable_timesteps`.
        """
        return self.num_timesteps_actual < self.num_timesteps_tuned

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["budget_was_reduced"] = self.budget_was_reduced
        return d


# Brax runs on the GPU, the bundle on the CPU, both in float32 once
# `enforce_fp32_matmul` is in effect. Reassociation then accounts for a few ULP;
# anything above this is a structural bug, not numerics.
#
# Without that call the gap is ~1e-2, because JAX's default matmul precision on
# this GPU truncates to a bfloat16 mantissa. See quant_control_bench.precision.
EXTRACTION_TOL = 1e-5


def minimum_trainable_timesteps(ppo_config: Any) -> int:
    """Smallest number of environment steps Brax will actually run.

    Brax does not treat ``num_timesteps`` as a budget to stop at. It divides the
    request into whole epochs and rounds *up*:

        env_step_per_training_step = batch_size * unroll_length
                                     * num_minibatches * action_repeat
        steps_per_epoch = ceil(num_timesteps
                               / (num_evals_after_init
                                  * env_step_per_training_step
                                  * max(num_resets_per_eval, 1)))

    so the floor is one training step per epoch. For the tuned CartpoleBalance
    config that floor is 88,473,600 steps — asking for 10M runs 8.8x more than
    requested, silently. Reporting the request as if it were the run is exactly
    the kind of unmeasured number this project is not allowed to publish, so the
    floor is computed explicitly and the executed total is recorded separately.
    """
    env_step_per_training_step = (
        int(ppo_config.batch_size)
        * int(ppo_config.unroll_length)
        * int(ppo_config.num_minibatches)
        * int(ppo_config.get("action_repeat", 1))
    )
    num_evals_after_init = max(int(ppo_config.num_evals) - 1, 1)
    resets = max(int(ppo_config.get("num_resets_per_eval", 0)), 1)
    return num_evals_after_init * env_step_per_training_step * resets


def verify_bundle_matches_brax(
    make_inference_fn: Any,
    params: Any,
    bundle: PolicyBundle,
    obs_key: str | None,
    samples: int = 1000,
    seed: int = 0,
) -> float:
    """Compare the extracted bundle against Brax's own deterministic policy.

    Returns the maximum absolute action difference; raises if it exceeds
    :data:`EXTRACTION_TOL`.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from quant_control_bench.precision import enforce_fp32_matmul

    enforce_fp32_matmul()

    policy = make_inference_fn(params, deterministic=True)

    rng = np.random.default_rng(seed)
    obs = rng.normal(0.0, 1.0, size=(samples, bundle.obs_dim)).astype(np.float32)
    obs_jax: Any = jnp.asarray(obs)
    if obs_key is not None:
        obs_jax = {obs_key: obs_jax}

    brax_action, _ = policy(obs_jax, jax.random.PRNGKey(seed))
    max_abs = float(np.abs(np.asarray(brax_action) - bundle.act(obs)).max())
    if not max_abs < EXTRACTION_TOL:
        raise ValueError(
            f"extracted policy disagrees with Brax by {max_abs:.3e} "
            f"(tolerance {EXTRACTION_TOL:.0e}); the bundle does not reproduce the "
            "trained network"
        )
    return max_abs


def train_policy(
    cfg: QcbConfig,
    out_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    restore_from: str | Path | None = None,
    progress: bool = True,
) -> tuple[PolicyBundle, TrainingRecord]:
    import functools

    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    # Orbax rejects a relative checkpoint path, and it does so at the *first
    # save*, i.e. after the training that produced it. Resolve here so no caller
    # can lose a run to it.
    out = Path(out_dir).resolve()
    checkpoints = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
    restore = Path(restore_from).resolve() if restore_from is not None else None
    if restore is not None and not restore.exists():
        raise FileNotFoundError(f"checkpoint to restore does not exist: {restore}")

    ppo_config = tuned_ppo_config(cfg.env)
    tuned_timesteps = int(ppo_config.num_timesteps)
    tuned_envs = int(ppo_config.num_envs)

    train_kwargs = dict(ppo_config)
    train_kwargs.pop("network_factory", None)
    net_kwargs = network_kwargs(ppo_config)

    train_kwargs["num_envs"] = int(cfg.train.num_envs)
    if cfg.train.num_timesteps is not None:
        train_kwargs["num_timesteps"] = int(cfg.train.num_timesteps)

    env = load_env(cfg.env)
    eval_env = load_env(cfg.env)
    obs_key = policy_obs_key(env.observation_size)

    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_kwargs)

    evals: list[EvalPoint] = []
    t0 = time.perf_counter()
    peak_vram = gpu_memory_used_mb()

    def progress_fn(step: int, metrics: Mapping[str, Any]) -> None:
        nonlocal peak_vram
        now = gpu_memory_used_mb()
        if now is not None and (peak_vram is None or now > peak_vram):
            peak_vram = now
        point = EvalPoint(
            step=int(step),
            reward=float(metrics.get("eval/episode_reward", float("nan"))),
            reward_std=float(metrics.get("eval/episode_reward_std", float("nan"))),
            wall_clock_s=time.perf_counter() - t0,
        )
        evals.append(point)
        if progress:
            print(
                f"  step {point.step:>12,}  reward {point.reward:9.3f}"
                f" +/- {point.reward_std:7.3f}  {point.wall_clock_s:7.1f}s"
                f"  vram {now} MiB",
                flush=True,
            )

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        seed=cfg.train.seed,
        progress_fn=progress_fn,
        # Evaluation during training uses the deterministic policy, matching
        # how every reported number in this project is produced.
        deterministic_eval=True,
        save_checkpoint_path=str(checkpoints) if checkpoints else None,
        restore_checkpoint_path=str(restore) if restore else None,
        **train_kwargs,
    )
    wall_clock = time.perf_counter() - t0

    bundle = bundle_from_brax(
        params,
        env_name=cfg.env,
        action_dim=int(env.action_size),
        obs_key=obs_key,
        activation=ACTIVATION,
    )

    # Persist before verifying. A run that trains for hours and then fails the
    # extraction check must leave the weights on disk to be debugged, not throw
    # them away.
    bundle.save(out)

    # The bundle is a hand-written re-implementation of Brax's forward pass, so
    # it is checked against the real one before anything downstream trusts it.
    # A transposed kernel or a reordered trunk still produces plausible-looking
    # actions; only a direct comparison catches that.
    max_extraction_error = verify_bundle_matches_brax(make_inference_fn, params, bundle, obs_key)

    record = TrainingRecord(
        env=cfg.env,
        seed=cfg.train.seed,
        num_timesteps_requested=int(train_kwargs["num_timesteps"]),
        num_timesteps_actual=evals[-1].step if evals else 0,
        num_timesteps_tuned=tuned_timesteps,
        num_envs=int(train_kwargs["num_envs"]),
        num_envs_tuned=tuned_envs,
        episode_length=int(ppo_config.episode_length),
        obs_dim=policy_obs_dim(env.observation_size),
        action_dim=int(env.action_size),
        obs_key=obs_key,
        activation=ACTIVATION,
        hidden_sizes=bundle.hidden_sizes,
        wall_clock_s=wall_clock,
        peak_vram_mb=peak_vram,
        final_reward=evals[-1].reward if evals else None,
        final_reward_std=evals[-1].reward_std if evals else None,
        extraction_max_abs_error=max_extraction_error,
        restored_from=str(restore) if restore else None,
        progress=evals,
    )

    (out / "training.json").write_text(json.dumps(record.to_json(), indent=2) + "\n")
    return bundle, record
