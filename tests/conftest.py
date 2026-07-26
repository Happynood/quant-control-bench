"""Shared test fixtures.

The synthetic-policy factory lives here rather than in a test module so that
both the export tests and the rollout tests can use it without importing each
other.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from quant_control_bench.export.bundle import PolicyBundle


def build_bundle(
    obs_dim: int = 5,
    action_dim: int = 1,
    hidden: tuple[int, ...] = (32, 32, 32, 32),
    activation: str = "swish",
    env: str = "Synthetic",
    seed: int = 0,
) -> PolicyBundle:
    """An untrained policy with the same architecture a trained one would have.

    Weights are drawn at the scale Brax's lecun-uniform init would produce, so
    the forward pass exercises a realistic dynamic range rather than saturating
    every activation.
    """
    rng = np.random.default_rng(seed)
    sizes = [obs_dim, *hidden, 2 * action_dim]
    kernels = [
        rng.normal(0.0, 1.0 / np.sqrt(sizes[i]), size=(sizes[i], sizes[i + 1])).astype(np.float32)
        for i in range(len(sizes) - 1)
    ]
    biases = [
        rng.normal(0.0, 0.1, size=sizes[i + 1]).astype(np.float32) for i in range(len(sizes) - 1)
    ]
    return PolicyBundle(
        env=env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        activation=activation,
        obs_key=None,
        norm_mean=rng.normal(0.0, 1.0, size=obs_dim).astype(np.float32),
        norm_std=(1.0 + rng.random(obs_dim)).astype(np.float32),
        kernels=kernels,
        biases=biases,
    )


@pytest.fixture
def make_bundle() -> Callable[..., PolicyBundle]:
    return build_bundle
