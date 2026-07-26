# Adapted from Happynood/quant-reasoning-bench (src/quantthink/config.py).
# Diff: the LLM-shaped fields (model, backend, kv_quant, thinking_cap, decoding
# temperature) are replaced by control-shaped fields (env, num_envs, PPO budget,
# rollout horizon, quantization schemes, perturbation axes). The Pydantic-v2 base
# pattern and the YAML loader are otherwise unchanged.
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Quantization scheme IDs, spec . `fp32` is the baseline every delta is
# measured against.
SchemeId = Literal[
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

# Perturbation axes for the robustness frontier, spec .
PerturbationAxis = Literal[
    "push_impulse",
    "mass_scale",
    "friction_scale",
    "actuator_delay",
    "obs_noise",
]


class TrainConfig(BaseModel):
    """PPO budget. Hyperparameters themselves come from Playground's tuned
    defaults for the env (spec : do not hand-roll PPO); only the knobs
    this project needs to override live here."""

    num_timesteps: int | None = None  # None = Playground's tuned default
    num_envs: int = Field(default=4096, ge=1)  # spec  VRAM protocol start value
    seed: int = 0
    checkpoint_every: int = Field(default=10, ge=1)  # in PPO iterations


class EvalConfig(BaseModel):
    episodes: int = Field(default=100, ge=1)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])  # fixed, not time-derived
    horizon: int | None = None  # None = the env's own episode length
    divergence_eps: float = Field(default=0.1, gt=0.0)  # T_div threshold,
    replay_buffer_size: int = Field(default=10_000, ge=1)  # open-loop error buffer,


class BootstrapConfig(BaseModel):
    resamples: int = Field(default=10_000, ge=1)
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: int = 0


class QcbConfig(BaseModel):
    name: str = "unnamed"
    env: str = "CartpoleBalance"
    schemes: list[SchemeId] = Field(default_factory=lambda: ["fp32"])
    perturbations: list[PerturbationAxis] = Field(default_factory=list)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)


def load_config(path: str | Path) -> QcbConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    return QcbConfig.model_validate(data)
