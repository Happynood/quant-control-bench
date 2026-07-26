"""Open-loop error, divergence horizon and the amplification factor.

The self-consistency check the design demands lives here: rolling the baseline
against itself must never diverge. If it does, the harness has a nondeterminism
bug and every `T_div` in the project is noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_control_bench.eval.metrics import (
    ACTION_RANGE,
    amplification_factor,
    open_loop_error,
)

pytest.importorskip("mujoco_playground", reason="sim extra not installed")

from quant_control_bench.envs import TIER0_ENV, load_env  # noqa: E402
from quant_control_bench.eval.metrics import (  # noqa: E402
    compute_dof_scale,
    divergence_horizon,
)
from quant_control_bench.quantize import apply_scheme  # noqa: E402

from .conftest import build_bundle  # noqa: E402

STARTS = 24
HORIZON = 150


# ── open-loop error (no simulator needed) ─────────────────────────────────────


def test_identical_policies_have_zero_open_loop_error(make_bundle) -> None:
    bundle = make_bundle(seed=1)
    states = np.random.default_rng(0).normal(size=(256, bundle.obs_dim)).astype(np.float32)

    err = open_loop_error(bundle, bundle, states)
    assert err.mse == 0.0
    assert err.rms == 0.0
    assert err.cosine_similarity == pytest.approx(1.0)


def test_open_loop_error_grows_as_precision_drops(make_bundle) -> None:
    bundle = make_bundle(seed=2)
    states = np.random.default_rng(1).normal(size=(512, bundle.obs_dim)).astype(np.float32)

    def rms_for(scheme: str) -> float:
        quantized, _ = apply_scheme(bundle, scheme)
        return open_loop_error(quantized, bundle, states).rms

    assert rms_for("fp16") < rms_for("int8-channel") < rms_for("int4-channel") < rms_for("ternary")


def test_open_loop_mse_sums_over_actuators(make_bundle) -> None:
    """The spec writes ||pi_q(s) - pi_0(s)||^2, so a 12-actuator robot must not
    be flattered by averaging the error across its actuators."""
    bundle = make_bundle(obs_dim=8, action_dim=4, seed=3)
    states = np.random.default_rng(2).normal(size=(64, 8)).astype(np.float32)

    quantized, _ = apply_scheme(bundle, "int4-channel")
    err = open_loop_error(quantized, bundle, states)

    delta = quantized.act(states) - bundle.act(states)
    assert err.mse == pytest.approx(float(np.sum(delta**2, axis=-1).mean()), rel=1e-6)


def test_normalization_is_against_the_full_action_range(make_bundle) -> None:
    bundle = make_bundle(seed=4)
    states = np.random.default_rng(3).normal(size=(128, bundle.obs_dim)).astype(np.float32)
    quantized, _ = apply_scheme(bundle, "int4-channel")

    err = open_loop_error(quantized, bundle, states)
    assert ACTION_RANGE == 2.0
    assert err.normalized_rms == pytest.approx(err.rms / 2.0)


def test_cosine_similarity_falls_when_actions_disagree(make_bundle) -> None:
    bundle = make_bundle(seed=5)
    states = np.random.default_rng(4).normal(size=(256, bundle.obs_dim)).astype(np.float32)

    fine, _ = apply_scheme(bundle, "int8-channel")
    coarse, _ = apply_scheme(bundle, "ternary")

    assert (
        open_loop_error(coarse, bundle, states).cosine_similarity
        < open_loop_error(fine, bundle, states).cosine_similarity
    )


# ── amplification factor ──────────────────────────────────────────────────────


def test_amplification_is_the_ratio_of_relative_quantities(make_bundle) -> None:
    bundle = make_bundle(seed=6)
    states = np.random.default_rng(5).normal(size=(128, bundle.obs_dim)).astype(np.float32)
    quantized, _ = apply_scheme(bundle, "int4-channel")
    err = open_loop_error(quantized, bundle, states)

    assert amplification_factor(0.5, err) == pytest.approx(0.5 / err.normalized_rms)


def test_amplification_is_undefined_when_the_policies_match(make_bundle) -> None:
    """fp32 against itself is 0/0. Reporting any finite number there would be
    inventing the headline metric."""
    bundle = make_bundle(seed=7)
    states = np.random.default_rng(6).normal(size=(64, bundle.obs_dim)).astype(np.float32)
    err = open_loop_error(bundle, bundle, states)
    assert np.isnan(amplification_factor(0.0, err))


# ── divergence horizon (needs the simulator) ──────────────────────────────────


@pytest.fixture(scope="module")
def env():
    return load_env(TIER0_ENV)


@pytest.fixture(scope="module")
def trained():
    from pathlib import Path

    from quant_control_bench.export.bundle import PolicyBundle

    fixture = Path(__file__).resolve().parents[1] / "data" / "smoke" / "policy-cartpole"
    return PolicyBundle.load(fixture)


@pytest.fixture(scope="module")
def dof_scale(trained, env):
    return compute_dof_scale(trained, num_episodes=16, horizon=HORIZON, env=env)


@pytest.mark.gpu
def test_baseline_never_diverges_from_itself(trained, dof_scale, env) -> None:
    """The self-consistency assertion.

    Deterministic policy, identical initial state, identical environment RNG:
    the two rollouts must stay bit-identical for the whole horizon.
    """
    result = divergence_horizon(
        trained, trained, dof_scale, num_starts=STARTS, horizon=HORIZON, env=env
    )
    assert result.censored_fraction == 1.0
    assert result.median == HORIZON
    assert np.all(result.steps == HORIZON)


@pytest.mark.gpu
@pytest.mark.slow
def test_baseline_never_diverges_from_itself_on_tier1() -> None:
    """The same assertion where it has teeth.

    Tier 0 satisfies this even with non-deterministic GPU kernels, so passing it
    proves nothing about a contact-rich environment. Measured on Go1 before the
    deterministic-ops flag, the baseline diverged from itself at step 6 of 1000
    while this file's Tier 0 version stayed green.
    """
    from pathlib import Path

    from quant_control_bench.envs import TIER1_ENV
    from quant_control_bench.export.bundle import PolicyBundle

    policy_dir = Path("artifacts/tier1-go1")
    if not policy_dir.exists():
        pytest.skip("Tier 1 policy not trained on this machine")

    trained = PolicyBundle.load(policy_dir)
    go1 = load_env(TIER1_ENV)
    scale = compute_dof_scale(trained, num_episodes=8, horizon=200, env=go1)
    result = divergence_horizon(trained, trained, scale, num_starts=16, horizon=200, env=go1)
    assert result.median == 200
    assert result.censored_fraction == 1.0


@pytest.mark.gpu
def test_a_broken_policy_diverges_quickly(trained, dof_scale, env) -> None:
    ternary, _ = apply_scheme(trained, "ternary")
    result = divergence_horizon(
        ternary, trained, dof_scale, num_starts=STARTS, horizon=HORIZON, env=env
    )
    assert result.median < HORIZON
    assert result.censored_fraction < 1.0


@pytest.mark.gpu
def test_coarser_precision_diverges_no_later(trained, dof_scale, env) -> None:
    """Ordering, not exact values: more quantization error should not buy a
    policy a longer agreement with the baseline."""

    def median_for(scheme: str) -> float:
        quantized, _ = apply_scheme(trained, scheme)
        return divergence_horizon(
            quantized, trained, dof_scale, num_starts=STARTS, horizon=HORIZON, env=env
        ).median

    assert median_for("ternary") <= median_for("int4-channel") <= median_for("fp16")


@pytest.mark.gpu
def test_dof_scale_has_no_zero_entries(dof_scale) -> None:
    """A motionless DOF would otherwise turn float noise into infinite distance."""
    assert np.all(dof_scale > 0)
    assert dof_scale.shape[0] > 0


@pytest.mark.gpu
def test_divergence_is_reproducible(trained, dof_scale, env) -> None:
    quantized, _ = apply_scheme(trained, "int4-channel")
    a = divergence_horizon(
        quantized, trained, dof_scale, num_starts=STARTS, horizon=HORIZON, seed=3, env=env
    )
    b = divergence_horizon(
        quantized, trained, dof_scale, num_starts=STARTS, horizon=HORIZON, seed=3, env=env
    )
    assert np.array_equal(a.steps, b.steps)


@pytest.mark.gpu
def test_synthetic_untrained_policy_diverges(trained, dof_scale, env) -> None:
    """Guards against a harness that reports no divergence for anything."""
    other = build_bundle(
        obs_dim=trained.obs_dim, action_dim=trained.action_dim, env=TIER0_ENV, seed=99
    )
    result = divergence_horizon(
        other, trained, dof_scale, num_starts=STARTS, horizon=HORIZON, env=env
    )
    assert result.median < HORIZON


@pytest.mark.gpu
def test_a_nan_policy_diverges_immediately_rather_than_never(trained, dof_scale, env) -> None:
    """`nan > threshold` is False, so NaN read as "never diverged".

    Measured on Tier 1: `int4-channel` with quantized normalization statistics
    divides by a zeroed scale, emits NaN actions, and scored `T_div = 1000/1000`
    — the full horizon, identical to a policy that matched the baseline exactly.
    The worst possible policy was reported with the best possible number.
    """
    broken = trained.copy()
    broken.norm_std = np.zeros_like(broken.norm_std)

    result = divergence_horizon(broken, trained, dof_scale, num_starts=8, horizon=HORIZON, env=env)
    assert result.median < HORIZON
    assert result.censored_fraction == 0.0
