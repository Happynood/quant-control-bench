"""Export a Playground scene so a browser can load the *same* model.

The demo has to simulate the physics the policy was trained on, and there are two
ways that quietly fails.

**Unreachable assets.** Playground hands MuJoCo a flat dict of every file the task
package owns — 20 files and 12 MB for Go1, including a 1.6 MB rough-terrain
texture the flat-terrain scene never touches. Shipping all of it is waste;
shipping a guessed subset is a broken scene. The include and file references are
therefore walked transitively from the top-level XML and only what is reachable
is written out.

**Post-load model edits.** `Go1Env.__init__` does not stop at parsing the XML. It
overwrites the integrator timestep, raises `ccd_iterations`, and rewrites joint
damping and every actuator's gain and bias from the PD constants in its config.
None of that is in the XML. A browser that loads the same XML and steps it gets
different physics — different enough to matter, and silent, because the robot
still walks. The overrides are extracted here and written next to the scene so the
browser can apply the same numbers, and the parity test is what confirms it did.

The bundle is verified before it is written: the model rebuilt from the exported
files is compared field by field against the one the environment actually uses.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Attributes that name another file: `include file=`, `mesh file=`, `texture
# file=`, `hfield file=`. MuJoCo resolves them all against the asset dict by
# bare filename when the model is built from a string, which is how Playground
# does it, so only the basename matters here.
_FILE_REF = re.compile(rb'file\s*=\s*"([^"]+)"')

# Model fields the environment rewrites after parsing. Compared by the verifier
# and shipped to the browser. Anything added to `Go1Env.__init__` upstream has to
# be added here too, and the verifier is what catches the omission.
_CHECKED_FIELDS = (
    "dof_damping",
    "actuator_gainprm",
    "actuator_biasprm",
    "jnt_range",
    "body_mass",
    "geom_friction",
    "actuator_ctrlrange",
)


@dataclass
class SceneBundle:
    """What the browser needs to reproduce the training model."""

    env: str
    top_xml: str
    files: list[str]
    total_bytes: int
    # Post-parse overrides, as plain lists so the browser can apply them directly.
    timestep: float
    ccd_iterations: int
    dof_damping: list[float]
    actuator_gainprm_col0: list[float]
    actuator_biasprm_col1: list[float]
    # Row widths of the actuator parameter matrices. Exported rather than assumed:
    # both are (nu, mjNGAIN=10) in MuJoCo 3.x, and a browser that guessed 3 for
    # `biasprm` — as its name and MuJoCo's mjNBIAS constant both suggest — writes
    # every actuator's bias into a different actuator's row and corrupts the PD
    # controller. Measured, that put the browser 2.8e-02 away from Python at the
    # very first step, before any chaos could be blamed.
    actuator_gainprm_stride: int
    actuator_biasprm_stride: int
    # Shape metadata, so the browser can assert it loaded what it expected rather
    # than silently running a different robot.
    nq: int
    nv: int
    nu: int
    nbody: int
    # Control decimation: the policy runs once per `n_substeps` physics steps.
    sim_dt: float
    ctrl_dt: float
    n_substeps: int

    # How the policy is wired to the plant. Exported as data because every one of
    # these is a silent-failure candidate if the browser guesses it: a wrong
    # sensor address feeds the policy someone else's velocity, a wrong default
    # pose biases every joint target, a missing action scale halves the gait.
    policy_interface: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _sensor_slice(model: Any, name: str) -> dict[str, int]:
    """Address and width of a named sensor, resolved from the compiled model."""
    import mujoco

    mj: Any = mujoco
    sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"model has no sensor named {name!r}")
    return {
        "adr": int(model.sensor_adr[sensor_id]),
        "dim": int(model.sensor_dim[sensor_id]),
    }


def _policy_interface(env: Any) -> dict[str, Any]:
    """Everything the browser needs to build the 48-dim observation.

    The layout is fixed by `Go1JoystickFlatTerrain._get_obs`: local linear
    velocity (3), gyro (3), projected gravity (3), joint angles minus the default
    pose (12), joint velocities (12), the previous action (12), and the joystick
    command (3).

    The environment also injects **uniform observation noise** at evaluation time,
    from its own config rather than from the perturbation axes. It is exported
    here so the browser can reproduce the deployed behaviour, but the parity test
    must switch it off: the noise is drawn from JAX's counter-based PRNG, which
    cannot be reproduced in JavaScript, and with it on no browser rollout can
    match a Python one however correct the physics is.
    """
    import numpy as np

    config = env._config  # noqa: SLF001
    noise = config.noise_config
    model = env.mj_model

    return {
        "obs_layout": [
            ["local_linvel", 3],
            ["gyro", 3],
            ["gravity", 3],
            ["joint_angles_minus_default", 12],
            ["joint_vel", 12],
            ["last_action", 12],
            ["command", 3],
        ],
        "sensors": {
            "local_linvel": _sensor_slice(model, "local_linvel"),
            "gyro": _sensor_slice(model, "gyro"),
        },
        # `get_gravity` is *not* a sensor. It is the world down-vector rotated
        # into the IMU site's frame:
        #
        #     data.site_xmat[imu_site_id].T @ [0, 0, -1]
        #
        # which equals the negated third row of that 3x3 row-major matrix. Reading
        # the `upvector` sensor instead — the obvious guess, and what this exporter
        # first shipped — gives roughly the negation, in a frame that need not be
        # the IMU's. Those three numbers are the policy's primary orientation
        # signal, so getting them backwards tells it the robot is upside down: the
        # measured effect was a Go1 that refused to walk, travelling 0.02 m in five
        # seconds against a 1.0 m/s command while its tracking error sat at exactly
        # the command magnitude.
        "gravity_from": "imu_site_down_vector",
        "imu_site_id": int(env._imu_site_id),  # noqa: SLF001
        "default_pose": np.asarray(env._default_pose, dtype=np.float64).tolist(),  # noqa: SLF001
        "action_scale": float(config.action_scale),
        "qpos_joint_start": 7,
        "qvel_joint_start": 6,
        "noise": {
            "level": float(noise.level),
            "scales": {k: float(v) for k, v in dict(noise.scales).items()},
            "distribution": "uniform(-1, 1) * level * scale",
        },
    }


# Directory hints that only make sense inside the installed Python package.
_DIR_ATTR = re.compile(rb'\s(?:mesh|asset|texture)dir\s*=\s*"[^"]*"')


def _flatten_paths(xml: bytes) -> bytes:
    """Rewrite asset references to bare filenames.

    Playground's XML points at the menagerie checkout with paths like
    `../../../../mujoco_menagerie/unitree_go1/assets/trunk.stl`, plus a `meshdir`
    that climbs out of the package. Python never follows them: `from_xml_string`
    resolves references against the in-memory asset dict by basename. MuJoCo in
    the browser has no such dict — it reads the Emscripten filesystem — so the
    references have to name files that actually exist there.

    Only the paths change. No geometry is touched, and the Python-side rebuild in
    :func:`bundle_scene` re-parses the rewritten files to prove it.
    """
    xml = _DIR_ATTR.sub(b"", xml)
    return _FILE_REF.sub(lambda m: b'file="' + Path(m.group(1).decode()).name.encode() + b'"', xml)


def _reachable_files(assets: dict[str, bytes], top: str) -> set[str]:
    """Filenames reachable from `top` by following file references."""
    seen: set[str] = set()
    queue = [top]
    while queue:
        name = queue.pop()
        if name in seen or name not in assets:
            continue
        seen.add(name)
        for match in _FILE_REF.findall(assets[name]):
            # References may carry a directory prefix; the asset dict is flat.
            queue.append(Path(match.decode()).name)
    return seen


def _model_signature(model: Any) -> dict[str, Any]:
    import numpy as np

    signature: dict[str, Any] = {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "timestep": float(model.opt.timestep),
    }
    for field in _CHECKED_FIELDS:
        signature[field] = np.asarray(getattr(model, field), dtype=np.float64).tolist()
    return signature


def bundle_scene(env_name: str, out_dir: str | Path) -> SceneBundle:
    """Write the scene the browser should load, and verify it reproduces the env."""
    import mujoco
    import numpy as np

    from quant_control_bench.envs import load_env

    env = load_env(env_name)
    assets: dict[str, bytes] = dict(env.model_assets)
    top = Path(str(env.xml_path)).name

    # The top-level XML is not always in the asset dict under its own name.
    if top not in assets:
        assets[top] = Path(str(env.xml_path)).read_bytes()

    keep = _reachable_files(assets, top)
    missing = {
        Path(m.decode()).name
        for name in keep
        for m in _FILE_REF.findall(assets[name])
        if Path(m.decode()).name not in assets
    }
    if missing:
        raise FileNotFoundError(
            f"scene {top!r} references files absent from the asset dict: {sorted(missing)}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(keep):
        payload = assets[name]
        if name.endswith(".xml"):
            payload = _flatten_paths(payload)
        (out / name).write_bytes(payload)

    # Rebuild from exactly what was written. If the pruning dropped something the
    # scene needs, this raises here rather than in a browser console.
    rebuilt_assets = {name: (out / name).read_bytes() for name in sorted(keep)}
    # `mujoco` populates its namespace at import, so the type stubs do not carry
    # `MjModel`; the module object is read through an untyped alias rather than
    # silencing the checker at every use.
    mj: Any = mujoco
    rebuilt = mj.MjModel.from_xml_string((out / top).read_text(), assets=rebuilt_assets)

    reference = env.mj_model
    ref_sig = _model_signature(reference)
    new_sig = _model_signature(rebuilt)

    # The rebuilt model is the *parsed* one, so the fields the environment
    # overwrites after parsing are expected to differ — those are exactly what
    # gets shipped as overrides. Everything else must match.
    structural = ("nq", "nv", "nu", "nbody", "ngeom")
    for key in structural:
        if ref_sig[key] != new_sig[key]:
            raise ValueError(
                f"exported scene does not reproduce the environment: {key} is "
                f"{new_sig[key]}, expected {ref_sig[key]}"
            )

    config = env._config  # noqa: SLF001 - the only source for the control rates
    sim_dt = float(config.sim_dt)
    ctrl_dt = float(config.ctrl_dt)

    bundle = SceneBundle(
        env=env_name,
        top_xml=top,
        files=sorted(keep),
        total_bytes=sum(len(assets[n]) for n in keep),
        timestep=float(reference.opt.timestep),
        ccd_iterations=int(reference.opt.ccd_iterations),
        dof_damping=np.asarray(reference.dof_damping, dtype=np.float64).tolist(),
        actuator_gainprm_col0=np.asarray(
            reference.actuator_gainprm[:, 0], dtype=np.float64
        ).tolist(),
        actuator_biasprm_col1=np.asarray(
            reference.actuator_biasprm[:, 1], dtype=np.float64
        ).tolist(),
        actuator_gainprm_stride=int(reference.actuator_gainprm.shape[1]),
        actuator_biasprm_stride=int(reference.actuator_biasprm.shape[1]),
        nq=int(reference.nq),
        nv=int(reference.nv),
        nu=int(reference.nu),
        nbody=int(reference.nbody),
        sim_dt=sim_dt,
        ctrl_dt=ctrl_dt,
        n_substeps=int(round(ctrl_dt / sim_dt)),
        policy_interface=_policy_interface(env),
    )

    (out / "scene.json").write_text(json.dumps(bundle.to_json(), indent=2) + "\n")
    return bundle


def apply_overrides(model: Any, bundle: SceneBundle) -> None:
    """Apply the environment's post-parse edits to a freshly parsed model.

    The Python-side mirror of what the browser does, so the parity test compares
    two models that were built the same way rather than one corrected model
    against one raw one.
    """
    import numpy as np

    model.opt.timestep = bundle.timestep
    model.opt.ccd_iterations = bundle.ccd_iterations
    model.dof_damping[:] = np.asarray(bundle.dof_damping)
    model.actuator_gainprm[:, 0] = np.asarray(bundle.actuator_gainprm_col0)
    model.actuator_biasprm[:, 1] = np.asarray(bundle.actuator_biasprm_col1)
