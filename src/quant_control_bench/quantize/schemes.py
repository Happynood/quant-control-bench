"""Weight quantization schemes.

Kernels are stored `(in_features, out_features)`, so an *output channel* is a
column, axis 1. That is the axis the action head's rows map to actuators
through, which is why per-channel schemes group along it.
"""

from __future__ import annotations

import numpy as np

from quant_control_bench.quantize.base import Quantizer, fake_quantize, symmetric_scale

# Output channels are columns of a (in_features, out_features) kernel.
CHANNEL_AXIS = 0  # reduce over inputs, keeping one scale per output channel


class NoOpQuantizer(Quantizer):
    """fp32 baseline. Every delta in the project is measured against this."""

    @property
    def id(self) -> str:
        return "fp32"

    @property
    def bits_per_weight(self) -> float:
        return 32.0

    def quantize(self, w: np.ndarray) -> np.ndarray:
        return w.astype(np.float32, copy=True)


class Float16Quantizer(Quantizer):
    """IEEE half. Not a grid: 10-bit mantissa with a shared exponent."""

    @property
    def id(self) -> str:
        return "fp16"

    @property
    def bits_per_weight(self) -> float:
        return 16.0

    def quantize(self, w: np.ndarray) -> np.ndarray:
        return w.astype(np.float16).astype(np.float32)


class IntegerQuantizer(Quantizer):
    """Symmetric integer quantization, per tensor or per output channel."""

    def __init__(self, bits: int, per_channel: bool) -> None:
        if bits < 2:
            raise ValueError(f"bits must be at least 2, got {bits}")
        self._bits = bits
        self._per_channel = per_channel
        # Symmetric signed range: int8 uses -127..127, not -128..127, so that
        # zero stays representable and the grid is symmetric around it.
        self._qmax = 2 ** (bits - 1) - 1

    @property
    def id(self) -> str:
        return f"int{self._bits}-{'channel' if self._per_channel else 'tensor'}"

    @property
    def bits_per_weight(self) -> float:
        return float(self._bits)

    def quantize(self, w: np.ndarray) -> np.ndarray:
        axis = CHANNEL_AXIS if (self._per_channel and w.ndim > 1) else None
        scale = symmetric_scale(w, self._qmax, axis)
        return fake_quantize(w, scale, self._qmax)


class GroupIntegerQuantizer(Quantizer):
    """Symmetric integer quantization with one scale per group of inputs.

    Groups run along the input (reduction) axis, the standard layout for
    weight-only quantization: every group is a contiguous slice of the dot
    product, so a group's scale can be folded into the accumulation.

    A trailing partial group is kept as its own group rather than padded. The
    Tier 0 policy's first layer is 5x32 — narrower than the group size — and
    padding it with zeros would drag the group's scale toward zero and quantize
    the real weights more coarsely than the scheme promises.
    """

    def __init__(self, bits: int, group_size: int) -> None:
        if group_size < 1:
            raise ValueError(f"group_size must be positive, got {group_size}")
        self._bits = bits
        self._group_size = group_size
        self._qmax = 2 ** (bits - 1) - 1

    @property
    def id(self) -> str:
        return f"int{self._bits}-group{self._group_size}"

    @property
    def bits_per_weight(self) -> float:
        return float(self._bits)

    def quantize(self, w: np.ndarray) -> np.ndarray:
        if w.ndim == 1:
            return IntegerQuantizer(self._bits, per_channel=False).quantize(w)

        out = np.empty_like(w, dtype=np.float32)
        for start in range(0, w.shape[0], self._group_size):
            block = w[start : start + self._group_size]
            scale = symmetric_scale(block, self._qmax, axis=0)
            out[start : start + self._group_size] = fake_quantize(block, scale, self._qmax)
        return out


class TernaryQuantizer(Quantizer):
    """{-1, 0, +1} with a per-output-channel scale.

    Threshold and scale follow Ternary Weight Networks: weights below
    `threshold_ratio * E|w|` collapse to zero, and the surviving magnitude is
    the mean absolute value of the weights that did survive. Taking the scale
    over all weights instead would shrink it toward zero as sparsity rises and
    systematically underdrive the actuators.
    """

    def __init__(self, threshold_ratio: float = 0.7) -> None:
        self._threshold_ratio = threshold_ratio

    @property
    def id(self) -> str:
        return "ternary"

    @property
    def bits_per_weight(self) -> float:
        # Three states: log2(3). The stored width depends on the packing, which
        # is a deployment detail, so the information-theoretic value is used.
        return float(np.log2(3))

    @property
    def threshold_ratio(self) -> float:
        return self._threshold_ratio

    def quantize(self, w: np.ndarray) -> np.ndarray:
        axis = CHANNEL_AXIS if w.ndim > 1 else None
        keepdims = axis is not None

        absolute = np.abs(w)
        threshold = self._threshold_ratio * np.mean(absolute, axis=axis, keepdims=keepdims)
        mask = absolute > threshold

        kept = np.sum(np.where(mask, absolute, 0.0), axis=axis, keepdims=keepdims)
        count = np.sum(mask, axis=axis, keepdims=keepdims)
        scale = np.where(count > 0, kept / np.maximum(count, 1), 0.0)

        return (np.sign(w) * mask * scale).astype(np.float32)
