"""Quantizer unit tests on synthetic weights.

Round-trip error bounds, scale correctness and ternary sparsity are checked
against what the arithmetic guarantees, not against recorded outputs, so a
scheme that silently changes behaviour fails here rather than in a benchmark
result three phases later.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_control_bench.export.bundle import GROUP_HEAD, GROUP_TRUNK
from quant_control_bench.quantize import (
    SCHEMES,
    Float16Quantizer,
    GroupIntegerQuantizer,
    IntegerQuantizer,
    NoOpQuantizer,
    TernaryQuantizer,
    apply_scheme,
    calibrate_activations,
    get_scheme,
)

SPEC_SCHEMES = [
    "fp32",
    "fp16",
    "int8-tensor",
    "int8-channel",
    "int4-channel",
    "int4-group32",
    "ternary",
    "mixed-head-fp16",
    "int8-act",
]


@pytest.fixture
def weights() -> np.ndarray:
    return np.random.default_rng(0).normal(0.0, 0.5, size=(64, 32)).astype(np.float32)


# ── registry ──────────────────────────────────────────────────────────────────


def test_every_scheme_in_the_spec_is_registered() -> None:
    assert sorted(SCHEMES) == sorted(SPEC_SCHEMES)


def test_unknown_scheme_names_the_known_ones() -> None:
    with pytest.raises(KeyError, match="int4-channel"):
        get_scheme("int3-magic")


# ── round-trip error bounds ───────────────────────────────────────────────────


@pytest.mark.parametrize("bits", [8, 4, 2])
@pytest.mark.parametrize("per_channel", [False, True])
def test_integer_round_trip_stays_within_half_a_step(
    weights: np.ndarray, bits: int, per_channel: bool
) -> None:
    """Rounding to the nearest grid point cannot move a weight by more than
    half a step. This is the property that makes the scheme a quantizer rather
    than an arbitrary transform."""
    q = IntegerQuantizer(bits, per_channel=per_channel)
    out = q.quantize(weights)

    qmax = 2 ** (bits - 1) - 1
    axis = 0 if per_channel else None
    step = np.max(np.abs(weights), axis=axis, keepdims=axis is not None) / qmax

    assert np.all(np.abs(out - weights) <= step / 2 + 1e-6)


@pytest.mark.parametrize("bits", [8, 4])
def test_integer_grid_has_the_expected_number_of_levels(bits: int) -> None:
    w = np.linspace(-1.0, 1.0, 4096, dtype=np.float32).reshape(-1, 1)
    out = IntegerQuantizer(bits, per_channel=True).quantize(w)
    # Symmetric signed range -qmax..qmax, so 2*qmax + 1 levels including zero.
    assert len(np.unique(out)) == 2 * (2 ** (bits - 1) - 1) + 1


def test_largest_weight_maps_onto_the_top_of_the_grid() -> None:
    """The scale is defined by max|w|, so that weight must survive exactly."""
    w = np.array([[0.1, -0.4], [0.9, 0.2]], dtype=np.float32)
    out = IntegerQuantizer(8, per_channel=False).quantize(w)
    assert np.isclose(out.max(), 0.9, atol=1e-6)


def test_per_channel_beats_per_tensor_when_channel_scales_differ(
    weights: np.ndarray,
) -> None:
    """The reason per-channel exists: one loud channel otherwise sets the step
    size for every quiet one.

    With one channel scaled up by 1000 and only 4 bits, the per-tensor step is
    wider than any quiet weight, so every quiet weight rounds to zero and the
    whole quiet sub-tensor is destroyed. Per-channel keeps each column on its
    own grid, so the error there stays within half of that column's step.
    """
    skewed = weights.copy()
    skewed[:, 0] *= 1000.0
    quiet = slice(1, None)

    per_tensor_out = IntegerQuantizer(4, per_channel=False).quantize(skewed)
    per_channel_out = IntegerQuantizer(4, per_channel=True).quantize(skewed)

    assert np.all(per_tensor_out[:, quiet] == 0.0), "per-tensor should erase the quiet columns"
    assert np.count_nonzero(per_channel_out[:, quiet]) > 0

    qmax = 7
    step = np.max(np.abs(skewed[:, quiet]), axis=0, keepdims=True) / qmax
    channel_err = np.abs(per_channel_out - skewed)[:, quiet]
    assert np.all(channel_err <= step / 2 + 1e-6)


def test_error_grows_as_precision_drops(weights: np.ndarray) -> None:
    def rms(q) -> float:
        return float(np.sqrt(np.mean((q.quantize(weights) - weights) ** 2)))

    assert (
        rms(NoOpQuantizer())
        < rms(Float16Quantizer())
        < rms(IntegerQuantizer(8, per_channel=True))
        < rms(IntegerQuantizer(4, per_channel=True))
        < rms(TernaryQuantizer())
    )


# ── individual schemes ────────────────────────────────────────────────────────


def test_fp32_is_the_identity(weights: np.ndarray) -> None:
    out = NoOpQuantizer().quantize(weights)
    assert np.array_equal(out, weights)
    assert out is not weights, "must not alias the input"


def test_fp16_is_exact_for_representable_values() -> None:
    w = np.array([[0.5, -0.25, 1.0, 0.0]], dtype=np.float32)
    assert np.array_equal(Float16Quantizer().quantize(w), w)


def test_fp16_loses_only_mantissa_bits(weights: np.ndarray) -> None:
    out = Float16Quantizer().quantize(weights)
    # fp16 keeps 10 mantissa bits, so relative error is bounded by 2^-11.
    assert np.all(np.abs(out - weights) <= np.abs(weights) * 2**-11 + 1e-7)


def test_ternary_has_three_levels_per_channel(weights: np.ndarray) -> None:
    out = TernaryQuantizer().quantize(weights)
    for channel in range(out.shape[1]):
        levels = np.unique(out[:, channel])
        assert len(levels) <= 3
        nonzero = levels[levels != 0]
        assert np.allclose(np.abs(nonzero), np.abs(nonzero)[0])


def test_ternary_sparsity_matches_the_threshold(weights: np.ndarray) -> None:
    """Threshold is 0.7*E|w|. For Gaussian weights E|w| = sigma*sqrt(2/pi), so
    the threshold sits at 0.559 sigma and about 42% of weights fall below it."""
    out = TernaryQuantizer(threshold_ratio=0.7).quantize(weights)
    zero_fraction = float(np.mean(out == 0.0))
    assert 0.35 < zero_fraction < 0.50


def test_ternary_sparsity_rises_with_the_threshold(weights: np.ndarray) -> None:
    low = float(np.mean(TernaryQuantizer(0.3).quantize(weights) == 0.0))
    high = float(np.mean(TernaryQuantizer(1.2).quantize(weights) == 0.0))
    assert low < high


def test_ternary_scale_uses_only_the_surviving_weights() -> None:
    """Averaging over all weights instead would shrink the scale as sparsity
    rises and systematically underdrive the actuators."""
    w = np.array([[0.0], [0.0], [0.0], [1.0], [1.2]], dtype=np.float32)
    out = TernaryQuantizer(threshold_ratio=0.7).quantize(w)
    surviving = out[out != 0]
    assert np.allclose(surviving, 1.1, atol=1e-6)


def test_group_quantizer_handles_a_layer_narrower_than_its_group() -> None:
    """The Tier 0 policy's first layer is 5x32 — one partial group."""
    w = np.random.default_rng(1).normal(size=(5, 32)).astype(np.float32)
    out = GroupIntegerQuantizer(4, group_size=32).quantize(w)
    assert out.shape == w.shape
    assert np.isfinite(out).all()


def test_group_quantizer_beats_per_channel_on_a_split_distribution() -> None:
    """Groups exist to isolate outliers along the reduction axis."""
    rng = np.random.default_rng(2)
    w = rng.normal(0.0, 0.01, size=(64, 8)).astype(np.float32)
    w[:32] *= 100.0  # first group is loud, second is quiet

    grouped = GroupIntegerQuantizer(4, group_size=32).quantize(w)
    channelwise = IntegerQuantizer(4, per_channel=True).quantize(w)

    quiet = slice(32, None)
    assert np.abs(grouped - w)[quiet].max() < np.abs(channelwise - w)[quiet].max()


@pytest.mark.parametrize("scheme_id", SPEC_SCHEMES)
def test_all_zero_tensor_survives_every_scheme(scheme_id: str) -> None:
    """A zero scale would divide by zero and poison the whole policy."""
    scheme = get_scheme(scheme_id)
    zeros = np.zeros((16, 8), dtype=np.float32)
    for quantizer in (scheme.trunk, scheme.head):
        out = quantizer.quantize(zeros)
        assert np.isfinite(out).all()
        assert np.array_equal(out, zeros)


@pytest.mark.parametrize("scheme_id", SPEC_SCHEMES)
def test_quantizers_preserve_shape_and_dtype(weights: np.ndarray, scheme_id: str) -> None:
    out = get_scheme(scheme_id).trunk.quantize(weights)
    assert out.shape == weights.shape
    assert out.dtype == np.float32


# ── applying a scheme to a policy ─────────────────────────────────────────────


def test_fp32_scheme_leaves_the_policy_untouched(make_bundle) -> None:
    bundle = make_bundle(seed=5)
    out, report = apply_scheme(bundle, "fp32")
    for a, b in zip(out.kernels, bundle.kernels, strict=True):
        assert np.array_equal(a, b)
    assert report.mean_bits_per_weight == 32.0


def test_applying_a_scheme_does_not_mutate_the_original(make_bundle) -> None:
    bundle = make_bundle(seed=6)
    before = [k.copy() for k in bundle.kernels]
    apply_scheme(bundle, "int4-channel")
    for a, b in zip(bundle.kernels, before, strict=True):
        assert np.array_equal(a, b)


def test_biases_are_never_quantized(make_bundle) -> None:
    bundle = make_bundle(seed=7)
    out, _ = apply_scheme(bundle, "ternary")
    for a, b in zip(out.biases, bundle.biases, strict=True):
        assert np.array_equal(a, b)


def test_mixed_head_holds_the_head_at_fp16(make_bundle) -> None:
    """The H3 experiment: trunk at int4, head untouched by it."""
    bundle = make_bundle(seed=8)
    mixed, report = apply_scheme(bundle, "mixed-head-fp16")
    int4, _ = apply_scheme(bundle, "int4-channel")

    assert np.array_equal(mixed.kernels[0], int4.kernels[0]), "trunk should match int4"
    assert not np.array_equal(mixed.kernels[-1], int4.kernels[-1]), "head should differ"

    head_error = np.abs(mixed.kernels[-1] - bundle.kernels[-1]).max()
    int4_head_error = np.abs(int4.kernels[-1] - bundle.kernels[-1]).max()
    assert head_error < int4_head_error


def test_mean_bits_are_weighted_by_parameter_count(make_bundle) -> None:
    """A plain average over groups would badly misreport a mixed scheme."""
    bundle = make_bundle(obs_dim=48, action_dim=12, hidden=(512, 256, 128), seed=9)
    _, report = apply_scheme(bundle, "mixed-head-fp16")

    assert report.bits_per_weight_trunk == 4.0
    assert report.bits_per_weight_head == 16.0
    assert report.trunk_weight_count > report.head_weight_count
    # Far closer to the trunk's 4 bits than to the midpoint of 10.
    assert 4.0 < report.mean_bits_per_weight < 5.0


def test_observation_normalization_is_left_alone_by_default(make_bundle) -> None:
    bundle = make_bundle(seed=10)
    out, report = apply_scheme(bundle, "int4-channel")
    assert np.array_equal(out.norm_mean, bundle.norm_mean)
    assert np.array_equal(out.norm_std, bundle.norm_std)
    assert report.quantized_obs_norm is False


def test_observation_normalization_can_be_quantized_on_request(make_bundle) -> None:
    """H4 needs the pair: same scheme, normalization in and out."""
    bundle = make_bundle(seed=11)
    out, report = apply_scheme(bundle, "int4-channel", quantize_obs_norm=True)
    assert not np.array_equal(out.norm_std, bundle.norm_std)
    assert report.quantized_obs_norm is True
    assert any("obs_norm" in e.name for e in report.tensor_errors)


def test_report_records_an_error_entry_per_layer(make_bundle) -> None:
    bundle = make_bundle(seed=12)
    _, report = apply_scheme(bundle, "int8-channel")
    assert len(report.tensor_errors) == bundle.num_layers
    assert report.tensor_errors[-1].name.endswith(f"[{GROUP_HEAD}]")
    assert report.tensor_errors[0].name.endswith(f"[{GROUP_TRUNK}]")


# ── activation quantization ───────────────────────────────────────────────────


def test_activation_scheme_refuses_to_guess_its_calibration(make_bundle) -> None:
    with pytest.raises(ValueError, match="calibration"):
        apply_scheme(make_bundle(seed=13), "int8-act")


def test_activation_quantization_changes_the_forward_pass(make_bundle) -> None:
    bundle = make_bundle(seed=14)
    states = np.random.default_rng(0).normal(size=(512, bundle.obs_dim)).astype(np.float32)

    out, report = apply_scheme(bundle, "int8-act", calibration_states=states)
    weights_only, _ = apply_scheme(bundle, "int8-channel")

    assert out.quantizes_activations
    assert report.activation_bits == 8
    assert not np.array_equal(out.act(states), weights_only.act(states))


def test_calibrated_scales_cover_the_observed_range(make_bundle) -> None:
    bundle = make_bundle(seed=15)
    states = np.random.default_rng(1).normal(size=(512, bundle.obs_dim)).astype(np.float32)
    scales = calibrate_activations(bundle, states, bits=8)

    assert len(scales) == bundle.num_layers
    assert all(float(s) > 0 for s in scales)

    # The first scale covers the normalized observation exactly.
    expected = float(np.abs(bundle.normalize(states)).max()) / 127
    assert np.isclose(float(scales[0]), expected, rtol=1e-6)


def test_activation_scales_survive_a_round_trip_through_disk(tmp_path, make_bundle) -> None:
    bundle = make_bundle(seed=16)
    states = np.random.default_rng(2).normal(size=(256, bundle.obs_dim)).astype(np.float32)
    out, _ = apply_scheme(bundle, "int8-act", calibration_states=states)

    out.save(tmp_path / "policy")
    from quant_control_bench.export.bundle import PolicyBundle

    loaded = PolicyBundle.load(tmp_path / "policy")

    assert loaded.quantizes_activations
    assert loaded.activation_qmax == out.activation_qmax
    assert np.array_equal(loaded.act(states), out.act(states))


def test_quantizing_normalization_stats_records_a_collapsed_scale(make_bundle) -> None:
    """`norm_std` is a strictly positive divisor and the weight schemes are signed.

    Small entries round to zero, and a zero divisor makes every observation inf
    and then NaN. Measured on Tier 1, `int4-channel` does exactly this. The
    report has to name the cause rather than leave an unexplained empty cell.
    """
    import numpy as np

    from quant_control_bench.quantize import apply_scheme

    bundle = make_bundle(obs_dim=8, action_dim=2)
    # One channel two orders of magnitude below the rest: int4 has 7 positive
    # levels, so this lands on zero.
    bundle.norm_std = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.01], dtype=np.float32)

    _, report = apply_scheme(bundle, "int4-channel", quantize_obs_norm=True)
    assert report.collapsed_norm_std >= 1
    assert report.to_json()["collapsed_norm_std"] == report.collapsed_norm_std


def test_weight_only_quantization_never_collapses_a_scale(make_bundle) -> None:
    from quant_control_bench.quantize import apply_scheme

    for scheme in ("int4-channel", "ternary", "int8-channel"):
        _, report = apply_scheme(make_bundle(), scheme)
        assert report.collapsed_norm_std == 0, scheme
