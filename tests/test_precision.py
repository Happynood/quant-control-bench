"""Guard the float32 matmul enforcement.

The default JAX matmul precision on this GPU truncates float32 inputs to a
bfloat16 mantissa. If that ever leaks back in, the fp32 baseline stops being
fp32 and every reported quantization delta is measured from the wrong origin —
silently, because the policy still walks.
"""

from __future__ import annotations

import numpy as np
import pytest

from quant_control_bench.precision import (
    DETERMINISM_FLAG,
    MATMUL_PRECISION,
    current_matmul_precision,
    deterministic_ops_enabled,
    enforce_deterministic_ops,
    enforce_fp32_matmul,
)

pytest.importorskip("jax", reason="sim extra not installed")

pytestmark = pytest.mark.gpu


def test_importing_the_package_enables_deterministic_ops() -> None:
    """The flag is read at backend init, so it cannot be applied lazily.

    Without it, two identical Go1 rollouts disagreed on all 100 of 100 episodes.
    """
    import quant_control_bench  # noqa: F401

    assert deterministic_ops_enabled()


def test_enforcement_preserves_existing_xla_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Appending, not overwriting: the user may have set flags of their own."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/x")
    enforce_deterministic_ops()

    import os

    flags = os.environ["XLA_FLAGS"]
    assert "--xla_dump_to=/tmp/x" in flags
    assert DETERMINISM_FLAG in flags


def test_enforcement_does_not_duplicate_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLA_FLAGS", DETERMINISM_FLAG)
    enforce_deterministic_ops()

    import os

    assert os.environ["XLA_FLAGS"].count("xla_gpu_deterministic_ops") == 1


def test_enforcement_sets_the_documented_precision() -> None:
    enforce_fp32_matmul()
    assert current_matmul_precision() == MATMUL_PRECISION


def test_loading_an_env_enforces_it() -> None:
    """No caller should have to remember this."""
    import quant_control_bench.precision as precision_module

    precision_module._applied = False
    from quant_control_bench.envs import TIER0_ENV, load_env

    load_env(TIER0_ENV)
    assert current_matmul_precision() == MATMUL_PRECISION


def test_enforced_matmul_matches_numpy() -> None:
    """The actual property that matters, measured rather than assumed."""
    import jax.numpy as jnp

    enforce_fp32_matmul()

    rng = np.random.default_rng(0)
    a = rng.normal(size=(256, 512)).astype(np.float32)
    b = rng.normal(size=(512, 256)).astype(np.float32)
    reference = a.astype(np.float64) @ b.astype(np.float64)
    scale = np.abs(reference).max()

    on_device = np.asarray(jnp.asarray(a) @ jnp.asarray(b))
    relative_error = np.abs(on_device - reference).max() / scale

    # numpy's own float32 error against float64 is ~5e-7 for this shape; the
    # bfloat16-truncated default is ~3e-4, two and a half orders of magnitude
    # worse. 1e-5 sits between the two with room to spare.
    assert relative_error < 1e-5, f"matmul is not running in float32: {relative_error:.3e}"
