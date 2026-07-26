"""Environment registry and config overrides.

Every env in this project is loaded through :func:`load_env` so that the MJX
implementation is chosen in exactly one place. See :mod:`registry` for why that
matters.
"""

from quant_control_bench.envs.registry import (
    MJX_IMPL,
    TIER0_ENV,
    TIER1_ENV,
    episode_length,
    load_env,
    policy_obs_key,
)

__all__ = [
    "MJX_IMPL",
    "TIER0_ENV",
    "TIER1_ENV",
    "episode_length",
    "load_env",
    "policy_obs_key",
]
