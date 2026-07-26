"""Deterministic closed-loop rollout harness.

Every number this project reports comes from here. The contract:

* **Deterministic policy.** The action is the mode of the distribution, never a
  sample. This is the control analogue of `temperature=0`.
* **Fixed keys.** Episodes are seeded from a fixed PRNG key, so the same
  configuration produces the same trajectories on the same machine. Two policies
  evaluated with the same `seed` start from *identical* initial states, which is
  what makes a paired comparison between precisions meaningful.
* **No auto-reset.** Episodes are not restarted when they terminate. A policy
  that falls at step 40 gets the return it earned in 40 steps, not the return of
  a fresh episode stitched onto it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_control_bench.envs import episode_length, load_env
from quant_control_bench.export.bundle import PolicyBundle


@dataclass
class RolloutResult:
    """Per-episode outcomes. Arrays are all shaped `(num_episodes,)`."""

    env: str
    horizon: int
    num_episodes: int
    seed: int
    episode_return: np.ndarray
    steps_survived: np.ndarray
    terminated: np.ndarray
    action_jitter: np.ndarray

    @property
    def mean_return(self) -> float:
        return float(self.episode_return.mean())

    @property
    def failure_rate(self) -> float:
        """Fraction of episodes that hit a termination condition early."""
        return float(self.terminated.mean())

    @property
    def mean_jitter(self) -> float:
        return float(self.action_jitter.mean())

    def to_json(self) -> dict[str, Any]:
        return {
            "env": self.env,
            "horizon": self.horizon,
            "num_episodes": self.num_episodes,
            "seed": self.seed,
            "mean_return": self.mean_return,
            "std_return": float(self.episode_return.std()),
            "failure_rate": self.failure_rate,
            "mean_steps_survived": float(self.steps_survived.mean()),
            "mean_action_jitter": self.mean_jitter,
            "episode_return": self.episode_return.tolist(),
            "steps_survived": self.steps_survived.tolist(),
            "terminated": self.terminated.astype(int).tolist(),
            "action_jitter": self.action_jitter.tolist(),
        }


def make_jax_policy(bundle: PolicyBundle):
    """JAX version of :meth:`PolicyBundle.act`, for on-device rollouts.

    Kept deliberately close to the NumPy implementation; the two are checked
    against each other in the tests, because a rollout that silently uses a
    different forward pass than the exported ONNX graph would invalidate every
    comparison in the project.
    """
    import jax.numpy as jnp

    from quant_control_bench.precision import enforce_fp32_matmul

    enforce_fp32_matmul()

    kernels = [jnp.asarray(k) for k in bundle.kernels]
    biases = [jnp.asarray(b) for b in bundle.biases]
    mean = jnp.asarray(bundle.norm_mean)
    std = jnp.asarray(bundle.norm_std)
    activation = bundle.activation
    action_dim = bundle.action_dim
    obs_key = bundle.obs_key
    last = len(kernels) - 1
    # Activation quantization has to be reproduced here too. Omitting it does
    # not fail loudly: the rollout simply measures the weight-only policy while
    # the results are labelled as activation-quantized.
    act_scales = (
        None
        if bundle.activation_scales is None
        else [jnp.asarray(s) for s in bundle.activation_scales]
    )
    act_qmax = float(bundle.activation_qmax)

    def act(obs: Any):
        import jax

        x = obs[obs_key] if obs_key is not None else obs
        x = (x - mean) / std
        for i, (w, b) in enumerate(zip(kernels, biases, strict=True)):
            if act_scales is not None:
                x = jnp.clip(jnp.round(x / act_scales[i]), -act_qmax, act_qmax) * act_scales[i]
            x = x @ w + b
            if i < last:
                if activation == "swish":
                    x = x * jax.nn.sigmoid(x)
                elif activation == "relu":
                    x = jnp.maximum(x, 0.0)
                else:
                    x = jnp.tanh(x)
        return jnp.tanh(x[..., :action_dim])

    return act


def rollout(
    bundle: PolicyBundle,
    num_episodes: int = 100,
    seed: int = 0,
    horizon: int | None = None,
    env: Any = None,
) -> RolloutResult:
    """Roll `num_episodes` deterministic episodes in parallel on the GPU."""
    import jax
    import jax.numpy as jnp

    env = env if env is not None else load_env(bundle.env)
    steps = int(horizon if horizon is not None else episode_length(env))
    policy = make_jax_policy(bundle)

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    state = reset(keys)

    zeros = jnp.zeros((num_episodes,))
    carry = (
        state,
        jnp.ones((num_episodes,)),  # alive mask
        zeros,  # accumulated return
        zeros,  # accumulated steps
        jnp.zeros((num_episodes, env.action_size)),  # previous action
        zeros,  # accumulated |a_t - a_{t-1}|
        zeros,  # number of jitter samples
    )

    def body(carry, t):
        state, alive, total, survived, prev_action, jitter, jitter_n = carry
        action = policy(state.obs)

        # Step 0 has no predecessor, so it contributes no jitter sample.
        counts = jnp.where(t == 0, 0.0, alive)
        delta = jnp.linalg.norm(action - prev_action, axis=-1)
        jitter = jitter + delta * counts
        jitter_n = jitter_n + counts

        nstate = step(state, action)
        total = total + nstate.reward * alive
        survived = survived + alive
        alive = alive * (1.0 - nstate.done)
        return (nstate, alive, total, survived, action, jitter, jitter_n), None

    (_, alive, total, survived, _, jitter, jitter_n), _ = jax.lax.scan(
        body, carry, jnp.arange(steps)
    )

    survived_np = np.asarray(survived)
    return RolloutResult(
        env=bundle.env,
        horizon=steps,
        num_episodes=num_episodes,
        seed=seed,
        episode_return=np.asarray(total),
        steps_survived=survived_np,
        terminated=np.asarray(alive) < 0.5,
        action_jitter=np.asarray(jitter) / np.maximum(np.asarray(jitter_n), 1.0),
    )


def collect_states(
    bundle: PolicyBundle,
    num_states: int = 10_000,
    seed: int = 0,
    num_episodes: int = 64,
    env: Any = None,
) -> np.ndarray:
    """Replay buffer of observations visited by `bundle`, shaped `(n, obs_dim)`.

    Used for two things that both require the *fp32* policy's own state
    distribution: calibrating activation quantization, and the open-loop action
    error, which compares policies on a fixed set of states rather than on the
    states each one happens to visit.

    Episodes run in parallel and states are taken from every step of every
    episode until the buffer is full, so the buffer follows the visitation
    distribution rather than over-weighting the first steps of an episode.
    """
    import jax
    import jax.numpy as jnp

    env = env if env is not None else load_env(bundle.env)
    policy = make_jax_policy(bundle)
    horizon = int(np.ceil(num_states / num_episodes))
    horizon = min(horizon, episode_length(env))

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    state = reset(jax.random.split(jax.random.PRNGKey(seed), num_episodes))

    def body(carry, _):
        state, alive = carry
        obs = state.obs[bundle.obs_key] if bundle.obs_key is not None else state.obs
        action = policy(state.obs)
        nstate = step(state, action)
        return (nstate, alive * (1.0 - nstate.done)), (obs, alive)

    (_, _), (observations, alive) = jax.lax.scan(
        body, (state, jnp.ones((num_episodes,))), None, length=horizon
    )

    # Drop states from episodes that had already terminated: they are frozen
    # post-failure states the deployed policy would never act on.
    observations = np.asarray(observations).reshape(-1, bundle.obs_dim)
    keep = np.asarray(alive).reshape(-1) > 0.5
    observations = observations[keep]

    if observations.shape[0] < num_states:
        raise ValueError(
            f"collected only {observations.shape[0]} live states, needed {num_states}; "
            "raise num_episodes or the horizon"
        )
    return observations[:num_states]
