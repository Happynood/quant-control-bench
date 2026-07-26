"""Convert Brax PPO parameters into a :class:`PolicyBundle`.

Brax returns ``(normalizer_params, policy_params, value_params)``. Only the
first two matter here: the value network is a training artifact and never runs
on a robot.

The normalization statistics are pulled *into* the bundle rather than left
outside it, so that the exported policy is a single function from raw
observation to action. Applying normalization outside the graph is the classic
way to get a policy that looks correct in Python and behaves like noise in the
browser.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_control_bench.export.bundle import PolicyBundle


def bundle_from_brax(
    params: Any,
    env_name: str,
    action_dim: int,
    obs_key: str | None,
    activation: str = "swish",
) -> PolicyBundle:
    normalizer_params, policy_params = params[0], params[1]

    mean, std = _normalizer_arrays(normalizer_params, obs_key)
    kernels, biases = _dense_layers(policy_params)

    return PolicyBundle(
        env=env_name,
        obs_dim=int(mean.shape[0]),
        action_dim=int(action_dim),
        activation=activation,
        obs_key=obs_key,
        norm_mean=mean,
        norm_std=std,
        kernels=kernels,
        biases=biases,
    )


def _normalizer_arrays(
    normalizer_params: Any, obs_key: str | None
) -> tuple[np.ndarray, np.ndarray]:
    mean = normalizer_params.mean
    std = normalizer_params.std
    if isinstance(mean, dict):
        if obs_key is None:
            raise ValueError("normalizer holds dict statistics but no observation key was given")
        mean, std = mean[obs_key], std[obs_key]
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _dense_layers(policy_params: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract Dense kernels/biases in forward order.

    Brax names them ``hidden_0 … hidden_n`` inside a ``params`` collection.
    Ordering is taken from the numeric suffix, not from dict iteration order,
    because a reordered trunk would still run and still produce plausible
    actions — a failure that no shape check would catch.
    """
    layers = policy_params["params"] if "params" in policy_params else policy_params

    names = [k for k in layers if k.startswith("hidden_")]
    if not names:
        raise ValueError(f"no hidden_* layers in policy params; got {sorted(layers)}")
    if len(names) != len(layers):
        raise ValueError(
            f"unexpected non-Dense entries in policy params: {sorted(set(layers) - set(names))}"
        )

    names.sort(key=lambda n: int(n.split("_")[-1]))
    expected = [f"hidden_{i}" for i in range(len(names))]
    if names != expected:
        raise ValueError(f"non-contiguous layer names: {names}")

    kernels = [np.asarray(layers[n]["kernel"], dtype=np.float32) for n in names]
    biases = [np.asarray(layers[n]["bias"], dtype=np.float32) for n in names]
    return kernels, biases
