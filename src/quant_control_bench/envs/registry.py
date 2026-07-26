"""Single point of entry for loading MuJoCo Playground environments.

Why this module exists rather than calling ``registry.load`` directly:

1. **MJX implementation.** Playground 0.2.0 with mujoco-mjx 3.10.0 resolves the
   default ``impl`` to the Warp backend, which crashes when the ``mujoco_warp``
   package is absent (``AttributeError: type object 'int' has no attribute
   'WARP'``). More importantly, spec  specifies the MJX/JAX backend and
    requires bit-reproducible rollouts for the divergence horizon
   ``T_div``. Forcing ``impl="jax"`` here means no call site can silently pick a
   different physics implementation and quietly change every number.

2. **Observation layout.** Some envs (Go1) expose a dict observation for an
   asymmetric actor-critic: the policy sees ``state``, the critic additionally
   sees ``privileged_state``. Only the policy half is ever exported to ONNX, so
   the key is resolved once, here.
"""

from __future__ import annotations

from typing import Any

# spec  locks the physics backend to MJX. See module docstring.
MJX_IMPL = "jax"

TIER0_ENV = "CartpoleBalance"  # smoke tier, `make verify`
TIER1_ENV = "Go1JoystickFlatTerrain"  # primary tier, headline results

# The observation key a policy consumes when the env returns a dict observation.
POLICY_OBS_KEY = "state"


def load_env(env_name: str, config_overrides: dict[str, Any] | None = None) -> Any:
    """Load a Playground env with the project's locked MJX implementation."""
    from mujoco_playground import registry

    from quant_control_bench.precision import enforce_fp32_matmul

    enforce_fp32_matmul()

    overrides: dict[str, Any] = {"impl": MJX_IMPL}
    if config_overrides:
        overrides.update(config_overrides)
    return registry.load(env_name, config_overrides=overrides)


def policy_obs_key(observation_size: Any) -> str | None:
    """Return the dict key the policy reads, or ``None`` for a flat observation.

    ``env.observation_size`` is an ``int`` for flat observations and a mapping of
    ``name -> shape`` for asymmetric actor-critic envs.
    """
    if isinstance(observation_size, dict):
        if POLICY_OBS_KEY not in observation_size:
            raise KeyError(
                f"dict observation has no {POLICY_OBS_KEY!r} key; got {sorted(observation_size)}"
            )
        return POLICY_OBS_KEY
    return None


def policy_obs_dim(observation_size: Any) -> int:
    """Width of the vector the policy network actually consumes."""
    key = policy_obs_key(observation_size)
    if key is None:
        return int(observation_size)
    shape = observation_size[key]
    return int(shape[0]) if isinstance(shape, tuple | list) else int(shape)


def episode_length(env: Any) -> int:
    """Horizon of a full episode, in control steps.

    Playground keeps this on the locked env config rather than exposing it on
    the env object, so the attribute access is wrapped here instead of being
    repeated at every call site.
    """
    config = getattr(env, "_config", None)
    length = getattr(config, "episode_length", None)
    if length is None:
        raise AttributeError(f"env {type(env).__name__} exposes no episode_length")
    return int(length)
