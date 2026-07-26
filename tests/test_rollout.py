"""Rollout harness contract: determinism, horizon accounting, forward-pass agreement.

These use a synthetic (untrained) policy on the Tier 0 environment. The point is
the harness, not the policy: an untrained policy exercises both the survive and
terminate paths.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco_playground", reason="sim extra not installed")

from quant_control_bench.envs import TIER0_ENV, load_env  # noqa: E402
from quant_control_bench.eval.rollout import make_jax_policy, rollout  # noqa: E402

from .conftest import build_bundle  # noqa: E402

pytestmark = pytest.mark.gpu

EPISODES = 8
HORIZON = 50


@pytest.fixture(scope="module")
def env():
    return load_env(TIER0_ENV)


@pytest.fixture
def bundle():
    return build_bundle(obs_dim=5, action_dim=1, env=TIER0_ENV, seed=42)


def test_same_seed_gives_identical_trajectories(bundle, env) -> None:
    a = rollout(bundle, num_episodes=EPISODES, seed=0, horizon=HORIZON, env=env)
    b = rollout(bundle, num_episodes=EPISODES, seed=0, horizon=HORIZON, env=env)
    assert np.array_equal(a.episode_return, b.episode_return)
    assert np.array_equal(a.steps_survived, b.steps_survived)
    assert np.array_equal(a.action_jitter, b.action_jitter)


@pytest.mark.slow
def test_contact_rich_rollouts_are_reproducible() -> None:
    """The same check on Tier 1, where it actually had something to catch.

    Tier 0 is reproducible even with non-deterministic GPU kernels, because
    balancing a pole accumulates no contacts. Go1 is not: measured before the
    deterministic-ops flag was set, two identical rollouts disagreed on all 100
    of 100 episodes by up to 1.03 of return against a mean of 31.5. The Tier 0
    version of this test passed throughout.

    Marked slow — it needs the trained Tier 1 policy and two full rollouts.
    """
    from pathlib import Path

    from quant_control_bench.envs import TIER1_ENV
    from quant_control_bench.export.bundle import PolicyBundle

    policy_dir = Path("artifacts/tier1-go1")
    if not policy_dir.exists():
        pytest.skip("Tier 1 policy not trained on this machine")

    trained = PolicyBundle.load(policy_dir)
    go1 = load_env(TIER1_ENV)
    a = rollout(trained, num_episodes=16, seed=0, horizon=200, env=go1)
    b = rollout(trained, num_episodes=16, seed=0, horizon=200, env=go1)
    assert np.array_equal(a.episode_return, b.episode_return)


def test_different_seeds_give_different_trajectories(bundle, env) -> None:
    a = rollout(bundle, num_episodes=EPISODES, seed=0, horizon=HORIZON, env=env)
    b = rollout(bundle, num_episodes=EPISODES, seed=1, horizon=HORIZON, env=env)
    assert not np.array_equal(a.episode_return, b.episode_return)


def test_steps_survived_never_exceeds_the_horizon(bundle, env) -> None:
    r = rollout(bundle, num_episodes=EPISODES, seed=0, horizon=HORIZON, env=env)
    assert r.horizon == HORIZON
    assert np.all(r.steps_survived <= HORIZON)
    assert np.all(r.steps_survived >= 1)


def test_returns_are_finite(bundle, env) -> None:
    r = rollout(bundle, num_episodes=EPISODES, seed=0, horizon=HORIZON, env=env)
    assert np.isfinite(r.episode_return).all()
    assert np.isfinite(r.action_jitter).all()


def test_jax_policy_matches_the_numpy_bundle(bundle) -> None:
    """The rollout and the ONNX export must run the same function.

    The rollout uses a JAX forward pass for speed and the export uses the NumPy
    one; if they diverge, closed-loop results would describe a policy that is
    not the one shipped to the browser.
    """
    import jax.numpy as jnp

    policy = make_jax_policy(bundle)
    obs = np.random.default_rng(0).normal(size=(256, bundle.obs_dim)).astype(np.float32)
    got = np.asarray(policy(jnp.asarray(obs)))
    assert np.abs(got - bundle.act(obs)).max() < 1e-5


def test_result_json_is_serializable(bundle, env) -> None:
    import json

    r = rollout(bundle, num_episodes=EPISODES, seed=0, horizon=HORIZON, env=env)
    payload = json.loads(json.dumps(r.to_json()))
    assert payload["num_episodes"] == EPISODES
    assert len(payload["episode_return"]) == EPISODES
    assert 0.0 <= payload["failure_rate"] <= 1.0


# ── the JAX rollout path must run the same function as everything else ────────
#
# The parity tests compare NumPy against ONNX. The rollout uses a third
# implementation in JAX, and nothing compared it to the other two: an
# activation-quantized policy rolled out with un-quantized activations for a
# whole phase, silently, because the returns still looked reasonable.


@pytest.mark.parametrize(
    "scheme_id",
    ["fp32", "fp16", "int8-tensor", "int8-channel", "int4-channel", "ternary", "mixed-head-fp16"],
)
def test_jax_policy_matches_numpy_for_every_weight_scheme(bundle, scheme_id: str) -> None:
    import jax.numpy as jnp

    from quant_control_bench.quantize import apply_scheme

    quantized, _ = apply_scheme(bundle, scheme_id)
    obs = np.random.default_rng(0).normal(size=(256, bundle.obs_dim)).astype(np.float32)

    got = np.asarray(make_jax_policy(quantized)(jnp.asarray(obs)))
    assert np.abs(got - quantized.act(obs)).max() < 1e-5


def test_jax_policy_applies_activation_quantization(bundle) -> None:
    """int8-act must actually quantize activations inside the rollout."""
    import jax.numpy as jnp

    from quant_control_bench.quantize import apply_scheme

    states = np.random.default_rng(1).normal(size=(512, bundle.obs_dim)).astype(np.float32)
    quantized, _ = apply_scheme(bundle, "int8-act", calibration_states=states)
    weights_only, _ = apply_scheme(bundle, "int8-channel")

    obs = np.random.default_rng(2).normal(size=(256, bundle.obs_dim)).astype(np.float32)
    got = np.asarray(make_jax_policy(quantized)(jnp.asarray(obs)))

    assert np.abs(got - quantized.act(obs)).max() < 1e-5
    # And it must differ from the weight-only rollout, or activations were ignored.
    other = np.asarray(make_jax_policy(weights_only)(jnp.asarray(obs)))
    assert np.abs(got - other).max() > 1e-6
