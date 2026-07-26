"""Compare a browser MuJoCo rollout against the Python one.

This comparison is asked for and, importantly, does *not* assume it comes out
at zero. The two sides run different engines: training and evaluation use MJX on
the GPU, the demo uses the MuJoCo C engine compiled to WebAssembly, and the
published WASM package tracks a different MuJoCo release than the Python stack.
The job here is to measure the divergence and report it, not to assert it away.

Three things are held fixed so that only the engines differ:

* **The same compiled model.** Both sides build from the exported scene bundle,
  including the post-parse overrides the environment applies.
* **The same actions.** A fixed analytic sequence, not a policy, so the comparison
  cannot be contaminated by ONNX-versus-NumPy differences in the network.
* **No observation noise.** The environment draws it from JAX's counter-based
  PRNG, which JavaScript cannot reproduce; with noise on, no browser rollout could
  match a Python one however correct the physics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ParityReport:
    steps: int
    nq: int
    # Per-step Euclidean distance between the two qpos trajectories.
    max_abs_qpos_error: float
    final_abs_qpos_error: float
    first_step_over_1e_3: int | None
    per_step_error: list[float]
    browser_user_agent: str
    note: str

    def to_json(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "nq": self.nq,
            "max_abs_qpos_error": self.max_abs_qpos_error,
            "final_abs_qpos_error": self.final_abs_qpos_error,
            "first_step_over_1e_3": self.first_step_over_1e_3,
            "per_step_error": self.per_step_error,
            "browser_user_agent": self.browser_user_agent,
            "note": self.note,
        }


def scripted_action(t: int, nu: int) -> np.ndarray:
    """The action sequence both sides replay. Must match `browser_check.mjs`."""
    return np.array([0.3 * np.sin(0.05 * t + i) for i in range(nu)], dtype=np.float64)


def python_trajectory(scene_dir: str | Path, steps: int) -> np.ndarray:
    """Roll the exported scene forward in the MuJoCo C engine via Python."""
    import mujoco

    from quant_control_bench.export.scene import SceneBundle, apply_overrides

    scene_path = Path(scene_dir)
    bundle_json = json.loads((scene_path / "scene.json").read_text())
    bundle = SceneBundle(**bundle_json)

    assets = {name: (scene_path / name).read_bytes() for name in bundle.files}
    mj: Any = mujoco
    model = mj.MjModel.from_xml_string((scene_path / bundle.top_xml).read_text(), assets=assets)
    apply_overrides(model, bundle)
    data = mj.MjData(model)

    mj.mj_resetData(model, data)
    if model.nkey > 0:
        data.qpos[:] = model.key_qpos[0]
    mj.mj_forward(model, data)

    if bundle.policy_interface is None:
        raise ValueError(
            "the exported scene carries no policy_interface; re-run `qcb export-scene`"
        )
    pose = np.asarray(bundle.policy_interface["default_pose"])
    scale = float(bundle.policy_interface["action_scale"])

    out = np.empty((steps, model.nq))
    for t in range(steps):
        action = scripted_action(t, model.nu)
        data.ctrl[:] = pose + action * scale
        for _ in range(bundle.n_substeps):
            mj.mj_step(model, data)
        out[t] = data.qpos
    return out


def compare(browser_json: str | Path, scene_dir: str | Path) -> ParityReport:
    payload = json.loads(Path(browser_json).read_text())
    browser = np.asarray(payload["qpos"], dtype=np.float64)
    steps = int(payload["steps"])

    python = python_trajectory(scene_dir, steps)
    if python.shape != browser.shape:
        raise ValueError(f"shape mismatch: python {python.shape}, browser {browser.shape}")

    per_step = np.abs(python - browser).max(axis=1)
    over = np.nonzero(per_step > 1e-3)[0]

    return ParityReport(
        steps=steps,
        nq=int(python.shape[1]),
        max_abs_qpos_error=float(per_step.max()),
        final_abs_qpos_error=float(per_step[-1]),
        first_step_over_1e_3=int(over[0]) if over.size else None,
        per_step_error=[float(x) for x in per_step],
        browser_user_agent=payload.get("version", {}).get("user_agent", ""),
        note=(
            "MJX/GPU is not involved: both sides run the MuJoCo C engine, Python "
            "natively and the browser through WebAssembly, from the same exported "
            "model with observation noise disabled. Any divergence is the engines "
            "and their build differing, and it compounds because the dynamics are "
            "contact-rich."
        ),
    )


def compare_observation(browser_json: str | Path) -> dict[str, Any]:
    """Compare the browser's 48-dim observation against the environment's own.

    Physics parity does not cover this. That test replays scripted actions and
    never calls `observe()`, so a wrong observation passes it untouched — which is
    what happened: `get_gravity` is not the `upvector` sensor but the world
    down-vector rotated into the IMU *site* frame, and reading the sensor instead
    inverted the policy's orientation signal. The robot stopped walking and every
    physics check stayed green.

    The reference is the environment's own accessors, not a second
    implementation of the same reading, so a shared misunderstanding cannot pass.
    """
    import jax.numpy as jnp
    from mujoco import mjx

    from quant_control_bench.envs import TIER1_ENV, load_env

    payload = json.loads(Path(browser_json).read_text())
    env = load_env(TIER1_ENV)
    mjx_model = env.mjx_model

    rows = []
    for state in payload["states"]:
        data = mjx.make_data(mjx_model)
        data = data.replace(
            qpos=jnp.asarray(state["qpos"]),
            qvel=jnp.asarray(state["qvel"]),
        )
        data = mjx.forward(mjx_model, data)

        reference = {
            "gravity": np.asarray(env.get_gravity(data), dtype=np.float64),
            "gyro": np.asarray(env.get_gyro(data), dtype=np.float64),
            "local_linvel": np.asarray(env.get_local_linvel(data), dtype=np.float64),
        }
        browser = {
            "gravity": np.asarray(state["gravity"], dtype=np.float64),
            # The observation layout is linvel(3), gyro(3), gravity(3), ...
            "local_linvel": np.asarray(state["obs"][0:3], dtype=np.float64),
            "gyro": np.asarray(state["obs"][3:6], dtype=np.float64),
        }
        rows.append(
            {
                "state": state["label"],
                "max_abs_error": {
                    key: float(np.abs(reference[key] - browser[key]).max()) for key in reference
                },
                "reference": {k: v.tolist() for k, v in reference.items()},
                "browser": {k: v.tolist() for k, v in browser.items()},
            }
        )

    return {
        "env": TIER1_ENV,
        "browser_user_agent": payload.get("version", {}).get("user_agent", ""),
        "states": rows,
        "max_abs_error": max(error for row in rows for error in row["max_abs_error"].values()),
    }
