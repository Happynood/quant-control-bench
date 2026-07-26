"""The metrics that separate open-loop error from closed-loop consequence.

The project's central claim is that these two come apart. Open-loop action error
is measured with the feedback removed; the divergence horizon and the return
drop are measured with it in place. The ratio between them is the headline
number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quant_control_bench.envs import episode_length, load_env
from quant_control_bench.eval.rollout import make_jax_policy
from quant_control_bench.export.bundle import PolicyBundle

# A tanh-squashed policy emits actions in [-1, 1], so the full action range is 2.
# Errors are reported against this rather than against the observed spread,
# which would change per scheme and make columns incomparable.
ACTION_RANGE = 2.0

# Divergence threshold: trajectories count as separated once the
# per-DOF-normalized state distance exceeds this.
DEFAULT_DIVERGENCE_EPS = 0.1

# Additional thresholds evaluated in the same pass. The the design's single epsilon
# saturates on a stabilization task, where the baseline barely moves and the
# per-DOF spread it is normalized by is correspondingly tiny; the ladder keeps
# the metric readable without changing the headline definition.
DIVERGENCE_LADDER = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class OpenLoopError:
    """Action error on a fixed buffer of states, with the feedback removed.

    This is deliberately *not* a rollout. Both policies see exactly the same
    states — those the fp32 policy visits — so nothing here can compound.
    """

    num_states: int
    mse: float
    normalized_mse: float
    rms: float
    normalized_rms: float
    max_abs: float
    cosine_similarity: float

    def to_json(self) -> dict[str, Any]:
        return {
            "num_states": self.num_states,
            "mse": self.mse,
            "normalized_mse": self.normalized_mse,
            "rms": self.rms,
            "normalized_rms": self.normalized_rms,
            "max_abs": self.max_abs,
            "cosine_similarity": self.cosine_similarity,
            "action_range": ACTION_RANGE,
        }


@dataclass(frozen=True)
class DivergenceHorizon:
    """How long two policies stay on the same trajectory."""

    horizon: int
    epsilon: float
    num_starts: int
    steps: np.ndarray
    censored_fraction: float
    # First-crossing steps at other thresholds, from the same rollout. A single
    # epsilon saturates: on a stabilization task the baseline barely moves, so
    # the per-DOF spread is tiny and almost any error crosses 0.1 immediately.
    # The ladder shows whether a horizon of 1 means "slightly off" or "gone".
    ladder: dict[float, np.ndarray] | None = None

    @property
    def median(self) -> float:
        return float(np.median(self.steps))

    @property
    def iqr(self) -> tuple[float, float]:
        q1, q3 = np.percentile(self.steps, [25, 75])
        return float(q1), float(q3)

    def to_json(self) -> dict[str, Any]:
        q1, q3 = self.iqr
        payload: dict[str, Any] = {
            "horizon": self.horizon,
            "epsilon": self.epsilon,
            "num_starts": self.num_starts,
            "median": self.median,
            "iqr_low": q1,
            "iqr_high": q3,
            "censored_fraction": self.censored_fraction,
            "steps": self.steps.tolist(),
        }
        if self.ladder is not None:
            payload["ladder"] = {
                f"{eps:g}": {
                    "median": float(np.median(steps)),
                    "censored_fraction": float(np.mean(steps >= self.horizon)),
                }
                for eps, steps in self.ladder.items()
            }
        return payload


def open_loop_error(
    quantized: PolicyBundle,
    baseline: PolicyBundle,
    states: np.ndarray,
) -> OpenLoopError:
    """Action disagreement between two policies on identical inputs."""
    a = np.asarray(quantized.act(states), dtype=np.float64)
    b = np.asarray(baseline.act(states), dtype=np.float64)
    delta = a - b

    # Per-state squared error summed over actuators, then averaged over states:
    # this is ||pi_q(s) - pi_0(s)||^2 as the design writes it, not a per-actuator
    # mean, so a 12-actuator robot is not flattered relative to a 1-actuator one.
    per_state = np.sum(delta**2, axis=-1)
    mse = float(per_state.mean())
    rms = float(np.sqrt(np.mean(delta**2)))

    numerator = float(np.sum(a * b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = numerator / denominator if denominator > 0 else float("nan")

    return OpenLoopError(
        num_states=int(states.shape[0]),
        mse=mse,
        normalized_mse=mse / ACTION_RANGE**2,
        rms=rms,
        normalized_rms=rms / ACTION_RANGE,
        max_abs=float(np.abs(delta).max()),
        cosine_similarity=cosine,
    )


def state_vector(state: Any) -> Any:
    """Physics state used for the divergence comparison: positions and velocities.

    The observation is deliberately not used. It is what the policy already
    conditions on, is normalized, and on Go1 omits privileged quantities; the
    question `T_div` asks is whether the *robot* is in the same place, not
    whether it perceives the same thing.
    """
    import jax.numpy as jnp

    data = state.data
    return jnp.concatenate([data.qpos, data.qvel], axis=-1)


def compute_dof_scale(
    baseline: PolicyBundle,
    num_episodes: int = 32,
    seed: int = 0,
    horizon: int | None = None,
    env: Any = None,
    floor: float = 1e-6,
) -> np.ndarray:
    """Per-DOF spread of the baseline trajectory, used to normalize distances.

    Computed once from the fp32 policy and reused for every scheme. Deriving it
    separately per scheme would rescale the axis each time and make the
    resulting horizons incomparable — a policy that shakes more would look like
    it diverges less.

    DOFs that never move get the floor instead of a zero scale, which would
    otherwise turn any float noise in them into infinite distance.
    """
    import jax
    import jax.numpy as jnp

    env = env if env is not None else load_env(baseline.env)
    steps = int(horizon if horizon is not None else episode_length(env))
    policy = make_jax_policy(baseline)

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    state = reset(jax.random.split(jax.random.PRNGKey(seed), num_episodes))

    def body(carry, _):
        state = carry
        nstate = step(state, policy(state.obs))
        return nstate, state_vector(nstate)

    _, trajectory = jax.lax.scan(body, state, None, length=steps)
    spread = np.asarray(jnp.std(trajectory.reshape(-1, trajectory.shape[-1]), axis=0))
    return np.maximum(spread, floor)


def divergence_horizon(
    quantized: PolicyBundle,
    baseline: PolicyBundle,
    dof_scale: np.ndarray,
    num_starts: int = 100,
    seed: int = 0,
    epsilon: float = DEFAULT_DIVERGENCE_EPS,
    horizon: int | None = None,
    env: Any = None,
    ladder: tuple[float, ...] = DIVERGENCE_LADDER,
) -> DivergenceHorizon:
    """First step at which the two policies' trajectories separate.

    Both rollouts start from the same initial state and step the same
    environment, so the environment's RNG stream advances identically for both:
    Playground envs split their key per step regardless of the action taken.
    Any difference in the trajectory therefore comes from the policies.

    Episodes that never separate are recorded at the full horizon and counted in
    `censored_fraction`, so a median of `horizon` can be read as "most pairs
    never diverged" rather than mistaken for a measurement.
    """
    import jax
    import jax.numpy as jnp

    env = env if env is not None else load_env(baseline.env)
    steps = int(horizon if horizon is not None else episode_length(env))

    policy_q = make_jax_policy(quantized)
    policy_0 = make_jax_policy(baseline)
    scale = jnp.asarray(dof_scale)

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(seed), num_starts)
    start = reset(keys)

    # Every threshold is evaluated from the same trajectory pair, so the ladder
    # costs one extra comparison per step rather than a rollout per epsilon.
    thresholds = jnp.asarray(sorted({epsilon, *ladder}))

    def body(carry, t):
        state_q, state_0, first_crossing = carry

        next_q = step(state_q, policy_q(state_q.obs))
        next_0 = step(state_0, policy_0(state_0.obs))

        distance = jnp.linalg.norm((state_vector(next_q) - state_vector(next_0)) / scale, axis=-1)
        # A non-finite distance is maximal divergence, not the absence of it.
        # `nan > threshold` is False, so without this a policy whose trajectory
        # has gone to NaN records no crossing at any threshold, is treated as
        # censored, and reports the **full horizon** — the same value as a
        # perfect match. Measured: `int4-channel` with quantized normalization
        # statistics produces NaN actions and scored `T_div = 1000 / 1000`.
        diverged = ~jnp.isfinite(distance)
        crossed = (distance[None, :] > thresholds[:, None]) | diverged[None, :]
        # Record the first crossing only; later ones must not overwrite it.
        first_crossing = jnp.where((first_crossing < 0) & crossed, t + 1, first_crossing)
        return (next_q, next_0, first_crossing), None

    init = (start, start, -jnp.ones((thresholds.shape[0], num_starts)))
    (_, _, first_crossing), _ = jax.lax.scan(body, init, jnp.arange(steps))

    crossings = np.asarray(first_crossing)
    censored = crossings < 0
    crossings = np.where(censored, steps, crossings).astype(np.float64)

    order = sorted({epsilon, *ladder})
    headline = order.index(epsilon)

    return DivergenceHorizon(
        horizon=steps,
        epsilon=epsilon,
        num_starts=num_starts,
        steps=crossings[headline],
        censored_fraction=float(censored[headline].mean()),
        ladder={eps: crossings[i] for i, eps in enumerate(order)},
    )


def amplification_factor(
    relative_return_drop: float,
    open_loop: OpenLoopError,
) -> float:
    """`A = (relative return drop) / (relative open-loop action error)`.

    The headline number: `A >> 1` says a small per-step action error becomes a
    large loss of task performance once the plant feeds it back.

    The denominator is the RMS action error as a fraction of the action range.
    RMS rather than MSE, so numerator and denominator are both first-order in
    the perturbation and the ratio is dimensionless in the way the claim needs;
    using MSE would make `A` scale with the error rather than characterize it.

    Returns NaN when the policies are numerically identical (fp32 against
    itself), where the ratio is 0/0 and any finite answer would be invented.
    """
    denominator = open_loop.normalized_rms
    if denominator <= 0.0:
        return float("nan")
    return relative_return_drop / denominator
