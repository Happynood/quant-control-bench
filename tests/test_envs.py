import pytest

from quant_control_bench.envs import MJX_IMPL, TIER0_ENV, TIER1_ENV
from quant_control_bench.envs.registry import policy_obs_dim, policy_obs_key

playground = pytest.importorskip("mujoco_playground", reason="sim extra not installed")


def test_mjx_impl_is_locked_to_jax() -> None:
    # The Warp backend crashes with the pinned mujoco-mjx and is not
    # bit-reproducible for T_div. Nothing may flip this silently.
    assert MJX_IMPL == "jax"


def test_flat_observation_has_no_policy_key() -> None:
    assert policy_obs_key(5) is None
    assert policy_obs_dim(5) == 5


def test_dict_observation_resolves_to_the_policy_half() -> None:
    obs = {"state": (48,), "privileged_state": (123,)}
    assert policy_obs_key(obs) == "state"
    assert policy_obs_dim(obs) == 48


def test_dict_observation_without_state_key_raises() -> None:
    with pytest.raises(KeyError):
        policy_obs_key({"privileged_state": (123,)})


def test_tier_envs_exist_in_the_playground_registry() -> None:
    from mujoco_playground import registry

    assert TIER0_ENV in registry.ALL_ENVS
    assert TIER1_ENV in registry.ALL_ENVS
