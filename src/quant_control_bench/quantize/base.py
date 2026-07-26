"""Quantizer ABC.

Pattern vendored from Happynood/llm-inference-benchmark
(src/llm_inference_benchmark/backends/base.py): a small abstract base plus a
name-keyed registry, so schemes are pluggable and a config can name one as a
string. Adapted from generation backends to weight quantizers.

**These are fake quantizers.** Each one rounds a float32 tensor onto the grid
its precision allows and immediately maps it back to float32, leaving dtype and
shape untouched. That is deliberate:

* The exported ONNX graph runs in `onnxruntime-web`, which computes in float32.
  A policy stored as int4 would still be dequantized before the matmul, so the
  numerical effect on the action is exactly what fake quantization reproduces.
* What this project measures is how quantization error propagates through a
  feedback loop, not how many bytes the weights occupy. Memory is a separate,
  exactly computable quantity, reported as `bits_per_weight` rather than
  inferred from a tensor's dtype.

Every scheme here is symmetric and zero-point free. Control policies have
roughly zero-centred weights, and an asymmetric scheme would put a zero point in
the graph for no measurable benefit at these widths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorError:
    """How far one tensor moved when it was quantized."""

    name: str
    max_abs_error: float
    rms_error: float
    relative_rms_error: float
    zero_fraction: float


class Quantizer(ABC):
    """Maps a float32 tensor onto a reduced-precision grid and back."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Scheme identifier, matching the ids the configs use."""

    @property
    @abstractmethod
    def bits_per_weight(self) -> float:
        """Stored width of a single weight, ignoring scale overhead."""

    @abstractmethod
    def quantize(self, w: np.ndarray) -> np.ndarray:
        """Round-trip `w` through this scheme. Same shape and dtype."""

    def error(self, name: str, original: np.ndarray, quantized: np.ndarray) -> TensorError:
        delta = quantized.astype(np.float64) - original.astype(np.float64)
        rms = float(np.sqrt(np.mean(delta**2)))
        scale = float(np.sqrt(np.mean(original.astype(np.float64) ** 2)))
        return TensorError(
            name=name,
            max_abs_error=float(np.abs(delta).max()),
            rms_error=rms,
            relative_rms_error=rms / scale if scale > 0 else 0.0,
            zero_fraction=float(np.mean(quantized == 0.0)),
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, bits={self.bits_per_weight})"


def symmetric_scale(w: np.ndarray, qmax: int, axis: int | None) -> np.ndarray:
    """Scale that maps `max|w|` onto `qmax`.

    A tensor (or channel) that is entirely zero would give a zero scale and then
    a division by zero; those entries get a scale of 1, which leaves the zeros
    exactly where they are.
    """
    amax = np.max(np.abs(w), axis=axis, keepdims=axis is not None)
    scale = amax / qmax
    return np.where(scale > 0, scale, 1.0)


def fake_quantize(w: np.ndarray, scale: np.ndarray, qmax: int) -> np.ndarray:
    """Round to the integer grid defined by `scale`, then map back to float.

    Clipped to the symmetric range `[-qmax, qmax]`. Half-way values round to
    even, following NumPy — the alternative (round-half-away-from-zero) biases
    magnitudes upward, which for an action head means a systematic overshoot in
    torque rather than symmetric noise.
    """
    q = np.clip(np.round(w / scale), -qmax, qmax)
    return (q * scale).astype(np.float32)
