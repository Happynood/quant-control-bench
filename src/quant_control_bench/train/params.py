"""Tuned PPO hyperparameters, taken from MuJoCo Playground rather than invented.

The spec forbids hand-rolling PPO: Playground ships per-environment tuned
configs and those are the ones used here. This module only dispatches to the
right config table and applies the two overrides this project is allowed to
make — ``num_envs`` (the VRAM protocol) and ``num_timesteps`` (the wall-clock
budget) — recording both so a reduction can never be silent.
"""

from __future__ import annotations

from typing import Any


def tuned_ppo_config(env_name: str) -> Any:
    """Playground's tuned Brax PPO config for `env_name`, unmodified."""
    from mujoco_playground import registry
    from mujoco_playground.config import (
        dm_control_suite_params,
        locomotion_params,
        manipulation_params,
    )

    if env_name in registry.dm_control_suite.ALL_ENVS:
        return dm_control_suite_params.brax_ppo_config(env_name)
    if env_name in registry.locomotion.ALL_ENVS:
        return locomotion_params.brax_ppo_config(env_name)
    if env_name in registry.manipulation.ALL_ENVS:
        return manipulation_params.brax_ppo_config(env_name)
    raise KeyError(f"no tuned PPO config for env {env_name!r}")


def network_kwargs(ppo_config: Any) -> dict[str, Any]:
    """Network-shape overrides the tuned config carries, if any.

    Brax's defaults apply when the config declares none. Returned as a plain
    dict so it can be serialized into the policy metadata: the ONNX exporter has
    to rebuild the exact same architecture, and guessing it is how an exported
    policy silently stops matching the trained one.
    """
    if "network_factory" not in ppo_config:
        return {}
    return {k: _plain(v) for k, v in ppo_config.network_factory.items()}


def _plain(value: Any) -> Any:
    if isinstance(value, tuple | list):
        return [int(v) if isinstance(v, int) else v for v in value]
    return value
