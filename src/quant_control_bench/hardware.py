# Vendored from Happynood/quant-reasoning-bench (src/quantthink/hardware.py), which in
# turn came from Happynood/quant-toolcall-bench (src/quantcall/hardware.py).
# Diff: the torch-CUDA probe is replaced by a JAX-device probe, since this project's
# accelerator stack is JAX/MJX, not torch. Everything else is unchanged.
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuInfo:
    name: str | None
    driver_version: str | None
    cuda_version: str | None
    vram_total_mb: int | None
    jax_devices: list[str] | None
    jax_default_backend: str | None


@dataclass(frozen=True)
class HardwareInfo:
    python_version: str
    platform_info: str
    cpu_model: str
    cpu_count: int | None
    gpu: GpuInfo | None


def collect_hardware() -> HardwareInfo:
    return HardwareInfo(
        python_version=sys.version,
        platform_info=platform.platform(),
        cpu_model=_cpu_model(),
        cpu_count=os.cpu_count(),
        gpu=_collect_gpu_info(),
    )


def gpu_memory_used_mb() -> int | None:
    """Currently allocated VRAM, for the design  VRAM protocol."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return int(r.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _collect_gpu_info() -> GpuInfo | None:
    name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    vram_total_mb: int | None = None
    jax_devices: list[str] | None = None
    jax_default_backend: str | None = None

    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,cuda_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
            if len(parts) >= 4:
                name, driver_version, cuda_version = parts[0], parts[1], parts[2]
                try:
                    vram_total_mb = int(parts[3])
                except ValueError:
                    pass
    except Exception:
        pass

    try:
        jax_module = importlib.import_module("jax")
        jax_devices = [str(d) for d in jax_module.devices()]
        jax_default_backend = str(jax_module.default_backend())
    except Exception:
        pass

    if all(v is None for v in (name, driver_version, cuda_version, vram_total_mb, jax_devices)):
        return None

    return GpuInfo(
        name=name,
        driver_version=driver_version,
        cuda_version=cuda_version,
        vram_total_mb=vram_total_mb,
        jax_devices=jax_devices,
        jax_default_backend=jax_default_backend,
    )
