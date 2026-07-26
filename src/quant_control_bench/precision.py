"""Force true float32 matmuls and bit-reproducible GPU kernels.

**This module exists because both defaults are wrong for this project.**

## Deterministic kernels

Measured on the reference machine: two `Go1JoystickFlatTerrain` rollouts of the
same policy, same seed, same environment object, in one process, disagreed on
**all 100 of 100 episodes**, by up to 1.03 of return against a mean of 31.5 —
about 3%. `CartpoleBalance` was bit-identical under the same test.

The cause is the GPU reduction order, not the policy: the two bundles compared
identical weight-for-weight, and their open-loop action error over the replay
buffer was exactly zero. Contact-rich MJX dynamics accumulate through
non-deterministic atomics, and a locomotion task amplifies the resulting few-ULP
difference within a handful of steps. Cartpole has no contacts to accumulate.

This breaks three things the design requires outright:

1. **`T_div` self-consistency.** Rolling the baseline against itself must give
   the full horizon. Measured without the flag it gave 6 steps out of 1000.
2. **Paired comparison.** Two precisions are supposed to differ only by their
   weights. If the simulator itself is a coin flip, the pairing is broken and
   the bootstrap interval is measuring simulator noise.
3. **A meaningful resolution floor.** fp32 compared against itself scored
   +0.109% [-0.039%, +0.377%], so any delta below roughly 0.4% would have been
   unmeasurable and reported anyway.

`--xla_gpu_deterministic_ops=true` removes it completely: 0 of 100 episodes
differ, max delta exactly zero. It costs about 38% wall clock on Tier 1 (45.6 s
against 33.0 s per 100-episode rollout), which is the correct trade for a
benchmark whose entire output is differences between near-identical policies.

The flag is read when the XLA backend initializes, so it has to be set *before*
JAX allocates a device. :func:`enforce_deterministic_ops` is therefore called at
package import, not lazily like the matmul setting.

## Matmul precision

On an Ampere GPU, JAX's default matmul precision for float32 inputs is not
float32. Measured on the reference machine, a 256x512 @ 512x256 product:

    default    max relative error vs float64: 3.075e-04
    bfloat16   max relative error vs float64: 3.075e-04
    highest    max relative error vs float64: 4.913e-07
    numpy      max relative error vs float64: 4.913e-07

The default is bit-identical to asking for bfloat16 — the tensor cores truncate
the mantissa. That is fatal here for two reasons:

1. **The fp32 baseline would not be fp32.** Every delta in this benchmark is
   measured against it. A baseline whose matmuls carry an 8-bit mantissa is
   already quantized, and the reported cost of int8 would be measured from the
   wrong origin.
2. **The browser disagrees with the trainer.** `onnxruntime-web` computes in
   real float32, so a policy evaluated in Python at the default precision is
   not the policy that runs in the demo.

The induced action error is around 3e-4 — the same order as the error that
weight quantization is supposed to introduce, so leaving it in place would
confound the measurement it is meant to detect.

Every JAX entry point in this package calls :func:`enforce_fp32_matmul` before
touching a device.
"""

from __future__ import annotations

import os

# What JAX must be told to get an actual float32 matmul.
MATMUL_PRECISION = "highest"

# What XLA must be told to make GPU reductions bit-reproducible.
DETERMINISM_FLAG = "--xla_gpu_deterministic_ops=true"

_applied = False


def enforce_deterministic_ops() -> None:
    """Add the deterministic-kernel flag to ``XLA_FLAGS``. Idempotent.

    Must run before the XLA backend initializes; calling it afterwards has no
    effect, which is why this is invoked at package import. Any flags the caller
    already set are preserved rather than overwritten.
    """
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_deterministic_ops" in flags:
        return
    os.environ["XLA_FLAGS"] = f"{flags} {DETERMINISM_FLAG}".strip()


def deterministic_ops_enabled() -> bool:
    return "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", "")


def enforce_fp32_matmul() -> None:
    """Set JAX's default matmul precision to true float32. Idempotent."""
    global _applied
    if _applied:
        return
    import jax

    jax.config.update("jax_default_matmul_precision", MATMUL_PRECISION)
    _applied = True


def current_matmul_precision() -> str | None:
    import jax

    # Read through the values table: the attribute shortcut is not in JAX's
    # type stubs, and the setting is what matters, not how it is reached.
    value = jax.config.values.get("jax_default_matmul_precision")
    return str(value) if value is not None else None
