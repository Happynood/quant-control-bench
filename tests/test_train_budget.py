"""Brax does not stop at `num_timesteps`; it rounds up to whole epochs.

Asking for fewer steps than one epoch trains the whole epoch anyway. Recording
the request as if it were the run would put an unmeasured number into every
training record, so the floor is computed explicitly and the executed total is
read back from the eval callback.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco_playground", reason="sim extra not installed")

from quant_control_bench.envs import TIER0_ENV, TIER1_ENV  # noqa: E402
from quant_control_bench.train.params import network_kwargs, tuned_ppo_config  # noqa: E402
from quant_control_bench.train.ppo import minimum_trainable_timesteps  # noqa: E402


def test_tier0_floor_is_far_above_a_small_request() -> None:
    cfg = tuned_ppo_config(TIER0_ENV)
    floor = minimum_trainable_timesteps(cfg)

    # batch_size 1024 * unroll_length 30 * num_minibatches 32
    #   * num_resets_per_eval 10 * (num_evals 10 - 1)
    assert floor == 88_473_600
    assert floor > 10_000_000, "a 10M request would silently run 8.8x longer"


def test_tuned_tier0_budget_is_itself_rounded_up() -> None:
    """Even the shipped 60M budget executes more than 60M."""
    cfg = tuned_ppo_config(TIER0_ENV)
    assert int(cfg.num_timesteps) == 60_000_000
    assert minimum_trainable_timesteps(cfg) > int(cfg.num_timesteps)


def test_tier1_floor_is_computable() -> None:
    cfg = tuned_ppo_config(TIER1_ENV)
    assert minimum_trainable_timesteps(cfg) > 0


def test_tier1_network_shape_comes_from_the_tuned_config() -> None:
    """Go1 overrides the network shape; the exporter must not guess it."""
    kwargs = network_kwargs(tuned_ppo_config(TIER1_ENV))
    assert kwargs["policy_hidden_layer_sizes"] == [512, 256, 128]
    assert kwargs["policy_obs_key"] == "state"
    assert kwargs["value_obs_key"] == "privileged_state"


def test_tier0_declares_no_network_override() -> None:
    assert network_kwargs(tuned_ppo_config(TIER0_ENV)) == {}


def test_unknown_env_is_rejected() -> None:
    with pytest.raises(KeyError):
        tuned_ppo_config("NotAnEnv")
