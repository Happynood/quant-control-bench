"""Applying a scheme to a whole policy.

A scheme is not one quantizer. The hypotheses under test are about *which*
tensors get quantized, so a scheme names a quantizer per tensor group:

* `trunk` — every layer but the last
* `head` — the final layer, whose outputs map directly to actuator commands
* `obs_norm` — the observation normalization statistics

`mixed-head-fp16` exists precisely to hold the head at fp16 while the trunk goes
to int4, which is the H3 experiment. Observation-normalization quantization is a
separate flag rather than a scheme of its own, so any scheme can be run with and
without it and the H4 comparison is a controlled pair.

Biases are left in float32 throughout. They are a rounding error's worth of the
parameter count (a few hundred values against tens of thousands) and quantizing
them would confound a scheme's measured cost with a change nobody deploys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quant_control_bench.export.bundle import (
    GROUP_HEAD,
    GROUP_OBS_NORM,
    GROUP_TRUNK,
    PolicyBundle,
    activate,
)
from quant_control_bench.quantize.base import Quantizer, TensorError
from quant_control_bench.quantize.schemes import (
    Float16Quantizer,
    GroupIntegerQuantizer,
    IntegerQuantizer,
    NoOpQuantizer,
    TernaryQuantizer,
)


@dataclass(frozen=True)
class PolicyScheme:
    """One row of the scheme table: what happens to each tensor group."""

    id: str
    trunk: Quantizer
    head: Quantizer
    activation_bits: int | None = None

    @property
    def quantizes_activations(self) -> bool:
        return self.activation_bits is not None

    def quantizer_for(self, group: str) -> Quantizer:
        if group == GROUP_TRUNK:
            return self.trunk
        if group == GROUP_HEAD:
            return self.head
        raise KeyError(f"scheme {self.id!r} has no quantizer for group {group!r}")


@dataclass
class QuantizationReport:
    """What a scheme actually did, per tensor. Written into result manifests."""

    scheme: str
    quantized_obs_norm: bool
    # Entries of `norm_std` that quantization drove to zero or below. Any value
    # above zero means the policy divides by zero somewhere and its actions are
    # NaN — a collapse with a named cause, not an unexplained empty cell.
    collapsed_norm_std: int
    activation_bits: int | None
    bits_per_weight_trunk: float
    bits_per_weight_head: float
    trunk_weight_count: int
    head_weight_count: int
    tensor_errors: list[TensorError] = field(default_factory=list)

    @property
    def weight_count(self) -> int:
        return self.trunk_weight_count + self.head_weight_count

    @property
    def mean_bits_per_weight(self) -> float:
        """Weighted by parameter count.

        A plain average over the two groups would misreport every mixed scheme:
        the head is a small fraction of the parameters, so holding it at fp16
        costs far less than "half the layers at 16 bits" would suggest.
        """
        if self.weight_count == 0:
            return 0.0
        return (
            self.bits_per_weight_trunk * self.trunk_weight_count
            + self.bits_per_weight_head * self.head_weight_count
        ) / self.weight_count

    def to_json(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "quantized_obs_norm": self.quantized_obs_norm,
            "collapsed_norm_std": self.collapsed_norm_std,
            "activation_bits": self.activation_bits,
            "bits_per_weight_trunk": self.bits_per_weight_trunk,
            "bits_per_weight_head": self.bits_per_weight_head,
            "weight_count": self.weight_count,
            "trunk_weight_count": self.trunk_weight_count,
            "head_weight_count": self.head_weight_count,
            "mean_bits_per_weight": self.mean_bits_per_weight,
            "tensor_errors": [
                {
                    "name": e.name,
                    "max_abs_error": e.max_abs_error,
                    "rms_error": e.rms_error,
                    "relative_rms_error": e.relative_rms_error,
                    "zero_fraction": e.zero_fraction,
                }
                for e in self.tensor_errors
            ],
        }


def build_registry() -> dict[str, PolicyScheme]:
    """The scheme table from the design, built once."""
    fp32 = NoOpQuantizer()
    fp16 = Float16Quantizer()
    int8_tensor = IntegerQuantizer(8, per_channel=False)
    int8_channel = IntegerQuantizer(8, per_channel=True)
    int4_channel = IntegerQuantizer(4, per_channel=True)
    int4_group32 = GroupIntegerQuantizer(4, group_size=32)
    ternary = TernaryQuantizer(threshold_ratio=0.7)

    schemes = [
        PolicyScheme("fp32", fp32, fp32),
        PolicyScheme("fp16", fp16, fp16),
        PolicyScheme("int8-tensor", int8_tensor, int8_tensor),
        PolicyScheme("int8-channel", int8_channel, int8_channel),
        PolicyScheme("int4-channel", int4_channel, int4_channel),
        PolicyScheme("int4-group32", int4_group32, int4_group32),
        PolicyScheme("ternary", ternary, ternary),
        # H3: does holding the action head at fp16 recover what int4 costs?
        PolicyScheme("mixed-head-fp16", int4_channel, fp16),
        # Does activation quantization matter at all at this network size?
        PolicyScheme("int8-act", int8_channel, int8_channel, activation_bits=8),
    ]
    return {s.id: s for s in schemes}


SCHEMES = build_registry()


def get_scheme(scheme_id: str) -> PolicyScheme:
    try:
        return SCHEMES[scheme_id]
    except KeyError:
        raise KeyError(f"unknown scheme {scheme_id!r}; known: {sorted(SCHEMES)}") from None


def calibrate_activations(
    bundle: PolicyBundle,
    states: np.ndarray,
    bits: int = 8,
) -> list[np.ndarray]:
    """Per-layer input scales, measured on a replay buffer of real states.

    The range is taken from the observed maximum absolute activation rather than
    a percentile. A control policy's rare large activations are exactly the ones
    that fire during a recovery from a push; clipping them would make the
    quantized policy look better on flat ground and worse under the
    perturbations this benchmark cares about.
    """
    qmax = 2 ** (bits - 1) - 1
    x = bundle.normalize(np.asarray(states, dtype=np.float32))

    scales: list[np.ndarray] = []
    for i, (w, b) in enumerate(zip(bundle.kernels, bundle.biases, strict=True)):
        amax = float(np.abs(x).max())
        scales.append(np.asarray(amax / qmax if amax > 0 else 1.0, dtype=np.float32))
        x = x @ w + b
        if i < bundle.num_layers - 1:
            x = activate(x, bundle.activation)
    return scales


def apply_scheme(
    bundle: PolicyBundle,
    scheme: PolicyScheme | str,
    quantize_obs_norm: bool = False,
    calibration_states: np.ndarray | None = None,
) -> tuple[PolicyBundle, QuantizationReport]:
    """Return a quantized copy of `bundle` plus a record of what changed."""
    scheme = get_scheme(scheme) if isinstance(scheme, str) else scheme
    out = bundle.copy()
    errors: list[TensorError] = []

    for i in range(out.num_layers):
        group = out.layer_group(i)
        quantizer = scheme.quantizer_for(group)
        original = out.kernels[i]
        out.kernels[i] = quantizer.quantize(original)
        errors.append(quantizer.error(f"kernel_{i}[{group}]", original, out.kernels[i]))

    if quantize_obs_norm:
        # Tested as its own group: it is a handful of values against tens of
        # thousands of weights, and H4 is the claim that its influence is
        # disproportionate to that count.
        norm_quantizer = scheme.quantizer_for(GROUP_TRUNK)
        for name, attr in (("norm_mean", "norm_mean"), ("norm_std", "norm_std")):
            original = getattr(out, attr)
            quantized = norm_quantizer.quantize(original)
            setattr(out, attr, quantized)
            errors.append(norm_quantizer.error(f"{name}[{GROUP_OBS_NORM}]", original, quantized))

        # `norm_std` is a divisor and is strictly positive by construction. The
        # weight schemes are symmetric and signed, which is the wrong shape for
        # it twice over: half the range goes to values it never takes, and small
        # entries round to zero. A zero divisor is not graceful degradation —
        # every downstream observation becomes inf and then NaN. Measured on
        # Tier 1, `int4-channel` zeroes entries of a 48-element vector and the
        # policy emits NaN actions for every input.
        #
        # This is recorded rather than repaired. Choosing an unsigned or
        # log-domain quantizer for this one tensor would be the sensible
        # engineering fix, but it would also be a different experiment: H4 asks
        # what happens when the normalization statistics are quantized the same
        # way as everything else, and this is the answer.
        collapsed = int((np.asarray(out.norm_std) <= 0.0).sum())
    else:
        collapsed = 0

    if scheme.quantizes_activations:
        if calibration_states is None:
            raise ValueError(
                f"scheme {scheme.id!r} quantizes activations and needs a calibration "
                "buffer of states collected from the fp32 policy"
            )
        # Calibrated on the *unquantized* bundle: the ranges describe the
        # activations the deployed policy actually sees, and taking them from
        # the already-quantized weights would fold weight error into the
        # activation scale and understate it.
        out.activation_scales = calibrate_activations(
            bundle, calibration_states, bits=scheme.activation_bits or 8
        )
        out.activation_qmax = 2 ** ((scheme.activation_bits or 8) - 1) - 1

    out.__post_init__()

    return out, QuantizationReport(
        scheme=scheme.id,
        quantized_obs_norm=quantize_obs_norm,
        collapsed_norm_std=collapsed,
        activation_bits=scheme.activation_bits,
        bits_per_weight_trunk=scheme.trunk.bits_per_weight,
        bits_per_weight_head=scheme.head.bits_per_weight,
        trunk_weight_count=int(sum(k.size for k in out.kernels[:-1])),
        head_weight_count=int(out.kernels[-1].size),
        tensor_errors=errors,
    )
