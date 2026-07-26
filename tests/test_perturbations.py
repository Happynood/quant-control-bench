"""Perturbation axes and the robustness frontier."""

from __future__ import annotations

import numpy as np
import pytest

from quant_control_bench.eval.frontier import _crossing, episode_success

pytest.importorskip("mujoco_playground", reason="sim extra not installed")

from quant_control_bench.envs import TIER0_ENV, load_env  # noqa: E402
from quant_control_bench.eval.frontier import robustness_frontier  # noqa: E402
from quant_control_bench.eval.perturbations import (  # noqa: E402
    AXES,
    NONE,
    TORSO_BODY,
    Perturbation,
    perturbed_env,
    perturbed_rollout,
    root_translational_dofs,
)
from quant_control_bench.quantize import apply_scheme  # noqa: E402

EPISODES = 24
HORIZON = 200


# ── crossing interpolation (no simulator) ─────────────────────────────────────


def test_crossing_interpolates_between_grid_points() -> None:
    magnitudes = np.array([0.0, 1.0, 2.0])
    rates = np.array([1.0, 0.75, 0.25])
    # Between 1.0 and 2.0 the rate falls 0.75 -> 0.25; 0.5 sits halfway.
    assert _crossing(magnitudes, rates) == pytest.approx(1.5)


def test_crossing_censors_instead_of_extrapolating() -> None:
    """Never invent a magnitude past the swept range."""
    magnitudes = np.array([0.0, 1.0, 2.0])
    rates = np.array([1.0, 0.9, 0.8])
    assert _crossing(magnitudes, rates) == 2.0


def test_crossing_handles_immediate_failure() -> None:
    magnitudes = np.array([0.0, 1.0])
    assert _crossing(magnitudes, np.array([0.2, 0.1])) == 0.0


# ── axis plumbing ─────────────────────────────────────────────────────────────


def test_unknown_axis_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown perturbation axis"):
        Perturbation("gravity", 1.0)


@pytest.fixture(scope="module")
def env():
    return load_env(TIER0_ENV)


@pytest.fixture(scope="module")
def trained():
    from pathlib import Path

    from quant_control_bench.export.bundle import PolicyBundle

    return PolicyBundle.load(
        Path(__file__).resolve().parents[1] / "data" / "smoke" / "policy-cartpole"
    )


@pytest.mark.gpu
def test_model_perturbation_does_not_mutate_the_original_env(env) -> None:
    """A mutated env would silently contaminate every later rollout, including
    the unperturbed baseline."""
    before = float(np.asarray(env.mjx_model.body_mass)[TORSO_BODY])
    clone = perturbed_env(env, Perturbation("mass_scale", 2.0))

    assert float(np.asarray(clone.mjx_model.body_mass)[TORSO_BODY]) == pytest.approx(2 * before)
    assert float(np.asarray(env.mjx_model.body_mass)[TORSO_BODY]) == pytest.approx(before)


@pytest.mark.gpu
def test_identity_perturbation_returns_the_same_env(env) -> None:
    assert perturbed_env(env, Perturbation("mass_scale", 1.0)) is env


@pytest.mark.gpu
def test_friction_scaling_touches_only_the_sliding_coefficient(env) -> None:
    clone = perturbed_env(env, Perturbation("friction_scale", 0.5))
    before = np.asarray(env.mjx_model.geom_friction)
    after = np.asarray(clone.mjx_model.geom_friction)

    assert np.allclose(after[:, 0], before[:, 0] * 0.5)
    assert np.allclose(after[:, 1:], before[:, 1:])


@pytest.mark.gpu
def test_cartpole_has_one_translational_root_dof(env) -> None:
    assert root_translational_dofs(env) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("axis", AXES)
def test_every_axis_runs_and_stays_finite(trained, env, axis: str) -> None:
    magnitude = {"mass_scale": 1.3, "friction_scale": 0.5}.get(axis, 1.0)
    result = perturbed_rollout(
        trained,
        Perturbation(axis, magnitude),
        num_episodes=EPISODES,
        horizon=HORIZON,
        env=env,
    )
    assert np.isfinite(result.episode_return).all()
    assert result.num_episodes == EPISODES


@pytest.mark.gpu
def test_zero_magnitude_matches_the_unperturbed_rollout(trained, env) -> None:
    from quant_control_bench.eval.rollout import rollout

    plain = rollout(trained, num_episodes=EPISODES, seed=0, horizon=HORIZON, env=env)
    zero = perturbed_rollout(trained, NONE, num_episodes=EPISODES, seed=0, horizon=HORIZON, env=env)
    assert np.allclose(plain.episode_return, zero.episode_return)


@pytest.mark.gpu
def test_observation_noise_degrades_return(trained, env) -> None:
    quiet = perturbed_rollout(trained, NONE, num_episodes=EPISODES, horizon=HORIZON, env=env)
    loud = perturbed_rollout(
        trained,
        Perturbation("obs_noise", 0.5),
        num_episodes=EPISODES,
        horizon=HORIZON,
        env=env,
    )
    assert loud.mean_return < quiet.mean_return


@pytest.mark.gpu
def test_actuator_delay_shifts_the_applied_action(trained, env) -> None:
    """A delayed policy must not produce the same trajectory as an undelayed one."""
    none = perturbed_rollout(
        trained, Perturbation("actuator_delay", 0), num_episodes=EPISODES, horizon=HORIZON, env=env
    )
    delayed = perturbed_rollout(
        trained, Perturbation("actuator_delay", 3), num_episodes=EPISODES, horizon=HORIZON, env=env
    )
    assert not np.allclose(none.episode_return, delayed.episode_return)


@pytest.mark.gpu
def test_perturbed_rollout_is_reproducible(trained, env) -> None:
    kwargs = dict(num_episodes=EPISODES, seed=2, horizon=HORIZON, env=env)
    a = perturbed_rollout(trained, Perturbation("push_impulse", 2.0), **kwargs)
    b = perturbed_rollout(trained, Perturbation("push_impulse", 2.0), **kwargs)
    assert np.array_equal(a.episode_return, b.episode_return)


# ── the success criterion ─────────────────────────────────────────────────────


def test_a_fallen_episode_is_not_a_success_however_much_it_banked() -> None:
    """The defect that made the push axis unmeasurable.

    A push at a uniformly random step lands late half the time, by which point
    the return is already earned. Grading on return alone let those episodes
    pass: the fp32 policy's success rate flattened at ~54% from 16 N*s to
    256 N*s while its fall rate climbed from 63% to 94%.
    """
    returns = np.array([100.0, 100.0])
    terminated = np.array([False, True])
    assert list(episode_success(returns, terminated, threshold=50.0)) == [True, False]


def test_a_blown_up_episode_is_a_failure_not_a_gap() -> None:
    """At 512 N*s the physics diverged and the mean came back `nan`."""
    returns = np.array([100.0, np.nan, np.inf])
    terminated = np.array([False, False, False])
    assert list(episode_success(returns, terminated, threshold=50.0)) == [True, False, False]


def test_criterion_is_unchanged_where_nothing_terminates() -> None:
    """CartpoleBalance never terminates, so Tier 0 results must be untouched."""
    rng = np.random.default_rng(0)
    returns = rng.uniform(0.0, 1000.0, size=200)
    terminated = np.zeros(200, dtype=bool)
    assert np.array_equal(episode_success(returns, terminated, threshold=500.0), returns >= 500.0)


# ── the frontier ──────────────────────────────────────────────────────────────


@pytest.mark.gpu
def test_a_collapsed_policy_cannot_look_robust(trained, env) -> None:
    """The bug this guards against, found on Tier 0.

    Scoring each policy against its own unperturbed return made `ternary` the
    most robust scheme on the observation-noise axis, because its own return had
    already collapsed and the bar moved down with it. Against a shared fp32
    reference it must rank last.
    """
    reference = perturbed_rollout(
        trained, NONE, num_episodes=EPISODES, horizon=HORIZON, env=env
    ).mean_return

    ternary, _ = apply_scheme(trained, "ternary")

    def p50_for(bundle) -> float:
        return robustness_frontier(
            bundle,
            "obs_noise",
            reference_return=reference,
            num_episodes=EPISODES,
            horizon=HORIZON,
            env=env,
            n_resamples=200,
        ).p50

    assert p50_for(ternary) < p50_for(trained)


@pytest.mark.gpu
def test_frontier_records_censoring(trained, env) -> None:
    reference = perturbed_rollout(
        trained, NONE, num_episodes=EPISODES, horizon=HORIZON, env=env
    ).mean_return
    frontier = robustness_frontier(
        trained,
        "obs_noise",
        reference_return=reference,
        magnitudes=(0.0, 0.01),
        num_episodes=EPISODES,
        horizon=HORIZON,
        env=env,
        n_resamples=100,
    )
    assert frontier.censored is True
    assert frontier.p50 == 0.01


def test_grid_selection_is_per_environment() -> None:
    """Tier 0 and Tier 1 need different grids and must not share one.

    Go1's calibration pushes up to 18 N*s; a cart on a rail is long gone by
    then, and every scheme would report a crossing at the first grid point.
    """
    from quant_control_bench.eval.frontier import DEFAULT_GRIDS, grid_for

    cartpole = grid_for("CartpoleBalance", "push_impulse")
    go1 = grid_for("Go1JoystickFlatTerrain", "push_impulse")
    assert cartpole != go1
    assert go1 == DEFAULT_GRIDS["push_impulse"]


def test_unknown_environment_falls_back_to_the_tier1_grid() -> None:
    from quant_control_bench.eval.frontier import DEFAULT_GRIDS, grid_for

    assert grid_for("SomethingElse", "obs_noise") == DEFAULT_GRIDS["obs_noise"]


def test_every_axis_has_a_grid_in_every_calibration() -> None:
    """A missing axis would raise only once the sweep reached it, hours in."""
    from quant_control_bench.eval.frontier import DEFAULT_GRIDS, ENV_GRIDS

    assert set(DEFAULT_GRIDS) == set(AXES)
    for env, grids in ENV_GRIDS.items():
        assert set(grids) == set(AXES), env


def test_every_grid_starts_at_its_unperturbed_value() -> None:
    """The curve must start at ~100% success, or `P50` has no upper anchor."""
    from quant_control_bench.eval.frontier import DEFAULT_GRIDS, ENV_GRIDS

    unperturbed = {
        "push_impulse": 0.0,
        "mass_scale": 1.0,
        "friction_scale": 1.0,
        "actuator_delay": 0.0,
        "obs_noise": 0.0,
    }
    for grids in [DEFAULT_GRIDS, *ENV_GRIDS.values()]:
        for axis, grid in grids.items():
            assert grid[0] == unperturbed[axis], f"{axis}: {grid}"
