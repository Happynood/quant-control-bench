"""Self-contained representation of a trained deterministic policy.

Everything downstream — quantization, ONNX export, the rollout harness, the
browser demo — reads a :class:`PolicyBundle` rather than Brax parameter
pytrees. Three reasons:

* **One architecture description.** The exporter has to rebuild the exact
  network that was trained. Storing the shape alongside the weights removes the
  guessing that otherwise makes an exported policy silently stop matching.
* **Named tensor groups.** The hypotheses under test are about *which* tensors
  are quantized: the action head separately from the trunk, and the observation
  normalization statistics separately from the weights. Those groups are
  addressable here by name.
* **Framework independence.** A bundle is plain NumPy, so a quantizer never
  needs a JAX device and the parity test can compare implementations that share
  no code.

Forward pass, matching Brax's deterministic PPO policy exactly:

    x      = (obs - norm_mean) / norm_std
    h_i    = activation(x @ W_i + b_i)     for every layer but the last
    logits = h @ W_last + b_last
    loc    = logits[..., :action_dim]      (the other half is the log-std,
                                            unused when acting deterministically)
    action = tanh(loc)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Group names used by the quantizers. `trunk` is every layer but the last,
# `head` is the final layer that maps directly to actuator commands, and
# `obs_norm` is the normalization statistics.
GROUP_TRUNK = "trunk"
GROUP_HEAD = "head"
GROUP_OBS_NORM = "obs_norm"

_ACTIVATIONS = ("swish", "relu", "tanh")


@dataclass
class PolicyBundle:
    env: str
    obs_dim: int
    action_dim: int
    activation: str
    obs_key: str | None
    norm_mean: np.ndarray
    norm_std: np.ndarray
    kernels: list[np.ndarray]
    biases: list[np.ndarray]
    # Set only by activation-quantizing schemes. One scalar scale per layer,
    # applied to that layer's *input* — so entry 0 covers the normalized
    # observation and the rest cover the hidden activations. Calibrated from a
    # replay buffer, never guessed, so it lives on the bundle rather than being
    # recomputed by whoever happens to run the forward pass.
    activation_scales: list[np.ndarray] | None = None
    activation_qmax: int = 127

    def __post_init__(self) -> None:
        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"unsupported activation {self.activation!r}")
        if self.activation_scales is not None and len(self.activation_scales) != len(self.kernels):
            raise ValueError(
                f"got {len(self.activation_scales)} activation scales "
                f"for {len(self.kernels)} layers"
            )
        if len(self.kernels) != len(self.biases):
            raise ValueError("kernel/bias count mismatch")
        if not self.kernels:
            raise ValueError("policy has no layers")
        if self.norm_mean.shape != (self.obs_dim,):
            raise ValueError(
                f"norm_mean has shape {self.norm_mean.shape}, expected {(self.obs_dim,)}"
            )
        if self.norm_std.shape != (self.obs_dim,):
            raise ValueError(
                f"norm_std has shape {self.norm_std.shape}, expected {(self.obs_dim,)}"
            )
        if self.kernels[0].shape[0] != self.obs_dim:
            raise ValueError(
                f"first layer expects {self.kernels[0].shape[0]} inputs, "
                f"observation is {self.obs_dim}"
            )
        # Brax's tanh-normal head emits (loc, scale); only loc is used when
        # acting deterministically, so the head is twice the action width.
        if self.kernels[-1].shape[1] != 2 * self.action_dim:
            raise ValueError(
                f"head emits {self.kernels[-1].shape[1]} values, "
                f"expected {2 * self.action_dim} for action_dim={self.action_dim}"
            )

    @property
    def hidden_sizes(self) -> list[int]:
        return [int(k.shape[1]) for k in self.kernels[:-1]]

    @property
    def num_layers(self) -> int:
        return len(self.kernels)

    def layer_group(self, index: int) -> str:
        """Which quantizable group layer `index` belongs to."""
        return GROUP_HEAD if index == self.num_layers - 1 else GROUP_TRUNK

    # ── forward ───────────────────────────────────────────────────────────────

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.norm_mean) / self.norm_std

    @property
    def quantizes_activations(self) -> bool:
        return self.activation_scales is not None

    def quantize_activation(self, x: np.ndarray, layer: int) -> np.ndarray:
        """Round a layer's input onto its calibrated grid, or pass it through."""
        if self.activation_scales is None:
            return x
        scale = self.activation_scales[layer]
        q = np.clip(np.round(x / scale), -self.activation_qmax, self.activation_qmax)
        return (q * scale).astype(np.float32)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        h = self.normalize(np.asarray(obs, dtype=np.float32))
        for i, (w, b) in enumerate(zip(self.kernels, self.biases, strict=True)):
            h = self.quantize_activation(h, i)
            h = h @ w + b
            if i < self.num_layers - 1:
                h = activate(h, self.activation)
        return h

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action: tanh of the distribution mean."""
        loc = self.logits(obs)[..., : self.action_dim]
        return np.tanh(loc)

    # ── serialization ─────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {"norm_mean": self.norm_mean, "norm_std": self.norm_std}
        for i, (w, b) in enumerate(zip(self.kernels, self.biases, strict=True)):
            arrays[f"kernel_{i}"] = w
            arrays[f"bias_{i}"] = b
        if self.activation_scales is not None:
            for i, s in enumerate(self.activation_scales):
                arrays[f"act_scale_{i}"] = np.asarray(s, dtype=np.float32)
        np.savez(d / "policy.npz", **arrays)
        (d / "policy.json").write_text(
            json.dumps(
                {
                    "env": self.env,
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "activation": self.activation,
                    "obs_key": self.obs_key,
                    "hidden_sizes": self.hidden_sizes,
                    "num_layers": self.num_layers,
                    "quantizes_activations": self.quantizes_activations,
                    "activation_qmax": self.activation_qmax,
                },
                indent=2,
            )
            + "\n"
        )
        return d

    @classmethod
    def load(cls, directory: str | Path) -> PolicyBundle:
        d = Path(directory)
        meta = json.loads((d / "policy.json").read_text())
        arrays = np.load(d / "policy.npz")
        n = int(meta["num_layers"])
        scales = (
            [arrays[f"act_scale_{i}"] for i in range(n)]
            if meta.get("quantizes_activations")
            else None
        )
        return cls(
            env=meta["env"],
            obs_dim=int(meta["obs_dim"]),
            action_dim=int(meta["action_dim"]),
            activation=meta["activation"],
            obs_key=meta["obs_key"],
            norm_mean=arrays["norm_mean"],
            norm_std=arrays["norm_std"],
            kernels=[arrays[f"kernel_{i}"] for i in range(n)],
            biases=[arrays[f"bias_{i}"] for i in range(n)],
            activation_scales=scales,
            activation_qmax=int(meta.get("activation_qmax", 127)),
        )

    def copy(self) -> PolicyBundle:
        return PolicyBundle(
            env=self.env,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            activation=self.activation,
            obs_key=self.obs_key,
            norm_mean=self.norm_mean.copy(),
            norm_std=self.norm_std.copy(),
            kernels=[k.copy() for k in self.kernels],
            biases=[b.copy() for b in self.biases],
            activation_scales=(
                None
                if self.activation_scales is None
                else [np.asarray(s).copy() for s in self.activation_scales]
            ),
            activation_qmax=self.activation_qmax,
        )


def activate(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "swish":
        return x * _sigmoid(x)
    if kind == "relu":
        return np.maximum(x, 0.0)
    return np.tanh(x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free sigmoid.

    The naive ``1 / (1 + exp(-x))`` overflows for strongly negative inputs. The
    result is still correct after the division, but it raises a warning on every
    call, which would train the reader to ignore warnings from this module.
    """
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out
