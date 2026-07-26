"""Perturbation axes for the robustness frontier.

The claim under test (H2) is that the minimum viable precision is not a
constant: a policy that survives int4 on flat ground fails at int4 once the
world stops matching training. Each axis below is one way of making it stop
matching.

Two kinds of axis, applied at different points:

* **Model perturbations** (mass, friction) change the simulated robot, so they
  are baked into a modified copy of the environment before the rollout starts.
* **Loop perturbations** (push, actuator delay, observation noise) change what
  happens between the policy and the plant, so they live inside the rollout.

Perturbation randomness uses its own PRNG stream, seeded separately from the
environment. Drawing it from the environment's key would shift the initial
states as the magnitude changes, and the frontier would then be measured
against a moving baseline.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_control_bench.envs import episode_length, load_env
from quant_control_bench.eval.rollout import RolloutResult, make_jax_policy
from quant_control_bench.export.bundle import PolicyBundle

AXES = ("push_impulse", "mass_scale", "friction_scale", "actuator_delay", "obs_noise")

# MuJoCo joint type codes.
_JOINT_FREE = 0
_JOINT_SLIDE = 2

# Body index 1 is the first non-world body: the cart on CartpoleBalance, the
# trunk on Go1. That is what the design means by "torso".
TORSO_BODY = 1


@dataclass(frozen=True)
class Perturbation:
    """One point on one axis. `magnitude` is in that axis's own units."""

    axis: str
    magnitude: float

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"unknown perturbation axis {self.axis!r}; known: {list(AXES)}")

    @property
    def is_model_level(self) -> bool:
        return self.axis in ("mass_scale", "friction_scale")

    def to_json(self) -> dict[str, Any]:
        return {"axis": self.axis, "magnitude": self.magnitude}


NONE = Perturbation("obs_noise", 0.0)


def perturbed_env(env: Any, perturbation: Perturbation) -> Any:
    """Environment with a modified physics model, or the original one.

    The env object is shallow-copied and its MJX model replaced. Mutating the
    original in place would silently contaminate every later rollout in the same
    process, including the unperturbed baseline.
    """
    if not perturbation.is_model_level or perturbation.magnitude == 1.0:
        return env

    model = env.mjx_model
    if perturbation.axis == "mass_scale":
        mass = model.body_mass.at[TORSO_BODY].multiply(perturbation.magnitude)
        model = model.replace(body_mass=mass)
    else:
        # Column 0 of geom_friction is the sliding coefficient; the other two are
        # torsional and rolling, which this axis leaves alone.
        friction = model.geom_friction.at[:, 0].multiply(perturbation.magnitude)
        model = model.replace(geom_friction=friction)

    clone = copy.copy(env)
    clone._mjx_model = model
    return clone


def root_translational_dofs(env: Any) -> int:
    """How many leading velocity DOFs a push should act on.

    A free-jointed robot takes the impulse on its three linear velocities; a
    cart on a slide joint takes it on one. Anything else has no well-defined
    torso to push, and saying so is better than pushing an arbitrary DOF.
    """
    joint_type = int(np.asarray(env.mj_model.jnt_type[0]).ravel()[0])
    if joint_type == _JOINT_FREE:
        return 3
    if joint_type == _JOINT_SLIDE:
        return 1
    raise ValueError(f"first joint has type {joint_type}, which has no translational DOFs to push")


def perturbed_rollout(
    bundle: PolicyBundle,
    perturbation: Perturbation = NONE,
    num_episodes: int = 100,
    seed: int = 0,
    horizon: int | None = None,
    env: Any = None,
    perturbation_seed: int = 12345,
) -> RolloutResult:
    """Deterministic rollout under one perturbation."""
    import jax
    import jax.numpy as jnp

    base_env = env if env is not None else load_env(bundle.env)
    run_env = perturbed_env(base_env, perturbation)
    steps = int(horizon if horizon is not None else episode_length(base_env))
    policy = make_jax_policy(bundle)

    axis = perturbation.axis
    magnitude = float(perturbation.magnitude)
    delay = int(round(magnitude)) if axis == "actuator_delay" else 0
    noise_sigma = magnitude if axis == "obs_noise" else 0.0

    reset = jax.jit(jax.vmap(run_env.reset))
    step = jax.jit(jax.vmap(run_env.step))

    state = reset(jax.random.split(jax.random.PRNGKey(seed), num_episodes))
    action_size = int(run_env.action_size)

    push_key, noise_key = jax.random.split(jax.random.PRNGKey(perturbation_seed))
    if axis == "push_impulse" and magnitude != 0.0:
        n_dofs = root_translational_dofs(base_env)
        torso_mass = float(np.asarray(base_env.mjx_model.body_mass)[TORSO_BODY])
        # Impulse (N*s) into a velocity change: J = m * dv.
        delta_v = magnitude / max(torso_mass, 1e-9)
        push_step = jax.random.randint(push_key, (num_episodes,), 1, max(steps, 2))
        direction = jax.random.normal(push_key, (num_episodes, n_dofs))
        direction = direction / jnp.maximum(
            jnp.linalg.norm(direction, axis=-1, keepdims=True), 1e-9
        )
        push = direction * delta_v
    else:
        n_dofs = 0
        push_step = jnp.full((num_episodes,), -1)
        push = jnp.zeros((num_episodes, 1))

    zeros = jnp.zeros((num_episodes,))
    # The delay line holds the actions a real actuator has not applied yet.
    queue = jnp.zeros((max(delay, 1), num_episodes, action_size))

    def body(carry, t):
        state, alive, total, survived, prev_action, jitter, jitter_n, queue = carry

        obs = state.obs
        if noise_sigma > 0.0:
            key = jax.random.fold_in(noise_key, t)
            if bundle.obs_key is not None:
                noisy = obs[bundle.obs_key] + noise_sigma * jax.random.normal(
                    key, obs[bundle.obs_key].shape
                )
                obs = {**obs, bundle.obs_key: noisy}
            else:
                obs = obs + noise_sigma * jax.random.normal(key, obs.shape)

        commanded = policy(obs)
        if delay > 0:
            applied = queue[0]
            queue = jnp.concatenate([queue[1:], commanded[None]], axis=0)
        else:
            applied = commanded

        counts = jnp.where(t == 0, 0.0, alive)
        jitter = jitter + jnp.linalg.norm(applied - prev_action, axis=-1) * counts
        jitter_n = jitter_n + counts

        nstate = step(state, applied)

        if n_dofs > 0:
            kick = jnp.where((t == push_step)[:, None], push, 0.0)
            qvel = nstate.data.qvel.at[:, :n_dofs].add(kick)
            nstate = nstate.replace(data=nstate.data.replace(qvel=qvel))

        total = total + nstate.reward * alive
        survived = survived + alive
        alive = alive * (1.0 - nstate.done)
        return (nstate, alive, total, survived, applied, jitter, jitter_n, queue), None

    init = (
        state,
        jnp.ones((num_episodes,)),
        zeros,
        zeros,
        jnp.zeros((num_episodes, action_size)),
        zeros,
        zeros,
        queue,
    )
    (_, alive, total, survived, _, jitter, jitter_n, _), _ = jax.lax.scan(
        body, init, jnp.arange(steps)
    )

    return RolloutResult(
        env=bundle.env,
        horizon=steps,
        num_episodes=num_episodes,
        seed=seed,
        episode_return=np.asarray(total),
        steps_survived=np.asarray(survived),
        terminated=np.asarray(alive) < 0.5,
        action_jitter=np.asarray(jitter) / np.maximum(np.asarray(jitter_n), 1.0),
    )
