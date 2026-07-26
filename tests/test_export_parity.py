"""Python <-> ONNX parity.

The the design's threshold is ||pi_onnx(s) - pi_jax(s)||_inf < 1e-4 over 1000 random
observations for fp32. These tests run it on synthetic policies so the guard is
live before any checkpoint exists; the same check runs against the trained
checkpoint in the export CLI.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_control_bench.export.bundle import GROUP_HEAD, GROUP_TRUNK, PolicyBundle
from quant_control_bench.export.onnx_export import export_onnx, run_onnx

PARITY_TOL = 1e-4


@pytest.mark.parametrize("activation", ["swish", "relu", "tanh"])
def test_onnx_matches_numpy_within_tolerance(tmp_path, make_bundle, activation: str) -> None:
    bundle = make_bundle(activation=activation)
    path = export_onnx(bundle, tmp_path / "policy.onnx")

    rng = np.random.default_rng(1234)
    obs = rng.normal(0.0, 2.0, size=(1000, bundle.obs_dim)).astype(np.float32)

    got = run_onnx(path, obs)
    want = bundle.act(obs)
    assert got.shape == want.shape == (1000, bundle.action_dim)
    assert np.abs(got - want).max() < PARITY_TOL


def test_onnx_matches_numpy_for_a_go1_shaped_policy(tmp_path, make_bundle) -> None:
    # 48-dim policy observation, 12 actuators, Playground's tuned Go1 trunk.
    bundle = make_bundle(obs_dim=48, action_dim=12, hidden=(512, 256, 128), seed=7)
    path = export_onnx(bundle, tmp_path / "policy.onnx")
    obs = np.random.default_rng(99).normal(size=(1000, 48)).astype(np.float32)
    assert np.abs(run_onnx(path, obs) - bundle.act(obs)).max() < PARITY_TOL


def test_normalization_is_inside_the_graph(tmp_path, make_bundle) -> None:
    """A caller that forgets to normalize must still get the right action.

    This is the failure mode the design calls out: normalization applied outside
    the graph looks fine in Python and behaves like noise in the browser.
    """
    bundle = make_bundle(seed=3)
    path = export_onnx(bundle, tmp_path / "policy.onnx")
    obs = np.random.default_rng(5).normal(size=(64, bundle.obs_dim)).astype(np.float32)

    from_graph = run_onnx(path, obs)
    pre_normalized = run_onnx(path, bundle.normalize(obs))

    assert np.abs(from_graph - bundle.act(obs)).max() < PARITY_TOL
    # Feeding already-normalized observations must give a *different* answer,
    # otherwise normalization is not actually in the graph.
    assert np.abs(pre_normalized - from_graph).max() > 1e-3


def test_action_stays_in_range_under_extreme_observations(tmp_path, make_bundle) -> None:
    """Saturated inputs must give finite, bounded actions.

    onnxruntime's fp32 Tanh returns up to 1.0000001, so the bound is checked
    with a float32-epsilon slack rather than exactly 1.0. Anything larger, or
    any non-finite value, means the graph is wrong.
    """
    bundle = make_bundle(seed=11)
    path = export_onnx(bundle, tmp_path / "policy.onnx")
    obs = np.random.default_rng(6).normal(0, 50, size=(256, bundle.obs_dim)).astype(np.float32)

    action = run_onnx(path, obs)
    assert np.isfinite(action).all()
    assert np.abs(action).max() <= 1.0 + 1e-5
    assert np.abs(action - bundle.act(obs)).max() < PARITY_TOL


def test_bundle_round_trips_through_disk(tmp_path, make_bundle) -> None:
    bundle = make_bundle(seed=2)
    bundle.save(tmp_path / "policy")
    loaded = PolicyBundle.load(tmp_path / "policy")

    obs = np.random.default_rng(8).normal(size=(128, bundle.obs_dim)).astype(np.float32)
    assert np.array_equal(loaded.act(obs), bundle.act(obs))
    assert loaded.hidden_sizes == bundle.hidden_sizes
    assert loaded.activation == bundle.activation


def test_layer_groups_split_head_from_trunk(make_bundle) -> None:
    bundle = make_bundle()
    groups = [bundle.layer_group(i) for i in range(bundle.num_layers)]
    assert groups[-1] == GROUP_HEAD
    assert set(groups[:-1]) == {GROUP_TRUNK}


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda b: b.__setattr__("action_dim", 3), "head emits"),
        (lambda b: b.__setattr__("obs_dim", 9), "norm_mean has shape"),
        (lambda b: b.__setattr__("activation", "gelu"), "unsupported activation"),
    ],
)
def test_bundle_rejects_inconsistent_shapes(mutate, message: str, make_bundle) -> None:
    bundle = make_bundle()
    mutate(bundle)
    with pytest.raises(ValueError, match=message):
        bundle.__post_init__()


# ── quantized policies must export identically too ────────────────────────────


@pytest.mark.parametrize(
    "scheme_id",
    [
        "fp16",
        "int8-tensor",
        "int8-channel",
        "int4-channel",
        "int4-group32",
        "ternary",
        "mixed-head-fp16",
    ],
)
def test_quantized_policies_keep_onnx_parity(tmp_path, make_bundle, scheme_id: str) -> None:
    """A quantized policy is still exported as an fp32 graph; the weights simply
    sit on a coarser grid. Parity must hold exactly as it does for fp32."""
    from quant_control_bench.quantize import apply_scheme

    quantized, _ = apply_scheme(make_bundle(seed=21), scheme_id)
    path = export_onnx(quantized, tmp_path / "policy.onnx")

    obs = np.random.default_rng(3).normal(size=(1000, quantized.obs_dim)).astype(np.float32)
    assert np.abs(run_onnx(path, obs) - quantized.act(obs)).max() < PARITY_TOL


def test_activation_quantization_is_reproduced_in_the_graph(tmp_path, make_bundle) -> None:
    """int8-act rounds the tensor entering every layer. If that stayed in Python
    only, the browser would run an un-quantized policy while the tables claimed
    otherwise."""
    from quant_control_bench.quantize import apply_scheme

    bundle = make_bundle(seed=22)
    states = np.random.default_rng(4).normal(size=(512, bundle.obs_dim)).astype(np.float32)
    quantized, _ = apply_scheme(bundle, "int8-act", calibration_states=states)

    path = export_onnx(quantized, tmp_path / "policy.onnx")
    obs = np.random.default_rng(5).normal(size=(1000, bundle.obs_dim)).astype(np.float32)

    assert np.abs(run_onnx(path, obs) - quantized.act(obs)).max() < PARITY_TOL

    # And it must differ from the weight-only export, or nothing was quantized.
    weights_only, _ = apply_scheme(bundle, "int8-channel")
    assert np.abs(quantized.act(obs) - weights_only.act(obs)).max() > 1e-6
