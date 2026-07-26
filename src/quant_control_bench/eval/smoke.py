"""End-to-end smoke pipeline for `make verify`.

Runs the real Tier 0 path in miniature, on the trained CartpoleBalance policy
committed under `data/smoke/policy-cartpole/`:

    load bundle -> deterministic MJX rollout -> ONNX export -> parity check

Every stage that a published result depends on is exercised, so a regression in
any of them fails the build rather than surfacing later as a wrong number. It is
kept to a short horizon and a small episode count to stay well inside the
verify budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from quant_control_bench.config import QcbConfig
from quant_control_bench.export.bundle import PolicyBundle
from quant_control_bench.hardware import gpu_memory_used_mb
from quant_control_bench.manifest import collect_manifest

SMOKE_POLICY = Path("data/smoke/policy-cartpole")

# Same threshold the export CLI enforces, from the design.
PARITY_TOL = 1e-4


def run_smoke(
    cfg: QcbConfig,
    config_path: str | Path,
    steps: int = 200,
    num_envs: int = 32,
    seed: int = 0,
    policy_dir: str | Path = SMOKE_POLICY,
) -> dict[str, Any]:
    """Roll out the smoke policy and check its ONNX export. JSON-serializable."""
    import tempfile

    from quant_control_bench.eval.rollout import rollout
    from quant_control_bench.export.onnx_export import export_onnx, run_onnx

    bundle = PolicyBundle.load(policy_dir)
    if bundle.env != cfg.env:
        raise ValueError(
            f"smoke policy was trained on {bundle.env!r} but the config asks for {cfg.env!r}"
        )

    t0 = time.perf_counter()
    result = rollout(bundle, num_episodes=num_envs, seed=seed, horizon=steps)
    elapsed = time.perf_counter() - t0

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = export_onnx(bundle, Path(tmp) / "policy.onnx")
        obs = np.random.default_rng(seed).normal(size=(256, bundle.obs_dim)).astype(np.float32)
        parity = float(np.abs(run_onnx(onnx_path, obs) - bundle.act(obs)).max())

    return {
        "name": cfg.name,
        "env": cfg.env,
        "policy": str(policy_dir),
        "steps": steps,
        "num_episodes": num_envs,
        "seed": seed,
        "mean_return": result.mean_return,
        "failure_rate": result.failure_rate,
        "mean_action_jitter": result.mean_jitter,
        "onnx_parity_max_abs": parity,
        "onnx_parity_ok": parity < PARITY_TOL,
        "env_steps_per_s": float(steps * num_envs / elapsed),
        "wall_clock_s": elapsed,
        "vram_used_mb": gpu_memory_used_mb(),
        "manifest": asdict(collect_manifest(config_path, cfg)),
    }


def write_result(result: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2) + "\n")
