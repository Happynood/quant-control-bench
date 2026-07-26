# Adapted from Happynood/quant-reasoning-bench (src/quantthink/manifest.py), which came
# from Happynood/quant-toolcall-bench (src/quantcall/manifest.py).
# Diff: RunManifest carries the control-benchmark axes (env, schemes, num_envs that
# actually fit, MJX implementation, eval seeds) instead of model/backend/kv_dtype, and
# records the pinned versions of the simulator stack, which determine whether a
# rollout is reproducible at all.
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from quant_control_bench.config import QcbConfig
from quant_control_bench.hardware import GpuInfo, collect_hardware
from quant_control_bench.precision import deterministic_ops_enabled

# Packages whose versions change rollout numerics. Pinned in uv.lock; snapshotted
# per run so a result JSON is self-describing.
_TRACKED_PACKAGES = (
    "jax",
    "jaxlib",
    "mujoco",
    "mujoco_playground",
    "brax",
    "onnx",
    "onnxruntime",
    "numpy",
)


@dataclass(frozen=True)
class RunManifest:
    timestamp: str
    git_commit: str | None
    git_dirty: bool | None
    config_sha256: str
    config_path: str
    name: str
    env: str
    schemes: list[str]
    perturbations: list[str]
    num_envs: int
    mjx_impl: str
    # Whether XLA was told to use bit-reproducible GPU kernels. Without it, a
    # contact-rich rollout is not reproducible run to run and no paired
    # comparison in this project is valid — see quant_control_bench.precision.
    deterministic_ops: bool
    train_seed: int
    eval_seeds: list[int]
    eval_episodes: int
    package_versions: dict[str, str]
    python_version: str
    platform_info: str
    cpu_model: str
    cpu_count: int | None
    gpu: GpuInfo | None


def collect_manifest(
    config_path: str | Path,
    cfg: QcbConfig,
    mjx_impl: str = "jax",
) -> RunManifest:
    hw = collect_hardware()
    return RunManifest(
        timestamp=datetime.now(UTC).isoformat(),
        git_commit=_git_commit(),
        git_dirty=_git_dirty(),
        config_sha256=_file_sha256(config_path),
        config_path=str(config_path),
        name=cfg.name,
        env=cfg.env,
        schemes=list(cfg.schemes),
        perturbations=list(cfg.perturbations),
        num_envs=cfg.train.num_envs,
        mjx_impl=mjx_impl,
        deterministic_ops=deterministic_ops_enabled(),
        train_seed=cfg.train.seed,
        eval_seeds=list(cfg.eval.seeds),
        eval_episodes=cfg.eval.episodes,
        package_versions=collect_package_versions(),
        python_version=hw.python_version,
        platform_info=hw.platform_info,
        cpu_model=hw.cpu_model,
        cpu_count=hw.cpu_count,
        gpu=hw.gpu,
    )


def collect_package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg.replace("_", "-"))
        except Exception:
            try:
                mod = importlib.import_module(pkg)
                versions[pkg] = str(getattr(mod, "__version__", "unknown"))
            except Exception:
                versions[pkg] = "not installed"
    return versions


def write_manifest(manifest: RunManifest, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def _file_sha256(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


def _git_commit() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _git_dirty() -> bool | None:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        )
        return bool(r.stdout.strip()) if r.returncode == 0 else None
    except Exception:
        return None
