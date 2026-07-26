"""Scene export for the browser demo.

The interesting failures here are silent ones: a scene that loads and walks but
is not the model the policy was benchmarked on.
"""

from __future__ import annotations

import pytest

from quant_control_bench.export.scene import _flatten_paths, _reachable_files


def test_only_reachable_assets_are_kept() -> None:
    """Playground hands over every file the task package owns, most unused.

    Shipping all 20 files sends a 1.6 MB rough-terrain texture the flat scene
    never touches; shipping a guessed subset produces a scene that fails to
    compile in a browser console.
    """
    assets = {
        "scene.xml": b'<mujoco><include file="robot.xml"/></mujoco>',
        "robot.xml": b'<mujoco><mesh file="../../assets/trunk.stl"/></mujoco>',
        "trunk.stl": b"solid",
        "unused_texture.png": b"png",
        "other_scene.xml": b"<mujoco/>",
    }
    assert _reachable_files(assets, "scene.xml") == {"scene.xml", "robot.xml", "trunk.stl"}


def test_reachability_follows_directory_prefixed_references() -> None:
    """The asset dict is flat; references are not."""
    assets = {
        "top.xml": b'<mujoco><include file="sub/dir/inner.xml"/></mujoco>',
        "inner.xml": b"<mujoco/>",
    }
    assert _reachable_files(assets, "top.xml") == {"top.xml", "inner.xml"}


def test_paths_are_flattened_to_bare_filenames() -> None:
    """MuJoCo in the browser reads a filesystem, not an in-memory asset dict.

    Playground's XML points outside the installed package entirely.
    """
    xml = (
        b'<mujoco><compiler meshdir="../../../../mujoco_menagerie/unitree_go1/assets"/>'
        b'<asset><mesh file="../../../../mujoco_menagerie/unitree_go1/assets/trunk.stl"/>'
        b"</asset></mujoco>"
    )
    out = _flatten_paths(xml)
    assert b'file="trunk.stl"' in out
    assert b"meshdir" not in out
    assert b"mujoco_menagerie" not in out


def test_flattening_leaves_other_attributes_alone() -> None:
    xml = b'<mujoco model="go1"><geom name="floor" size="0 0 0.01"/></mujoco>'
    assert _flatten_paths(xml) == xml


@pytest.mark.gpu
@pytest.mark.slow
def test_exported_bundle_reproduces_the_environment(tmp_path) -> None:
    """The whole point: the browser must get the benchmarked model.

    Also pins the actuator parameter strides. `actuator_biasprm` is (nu, 10), not
    (nu, 3) as its name and MuJoCo's mjNBIAS constant suggest, and a browser that
    assumed 3 scattered every actuator's PD bias into another actuator's row —
    worth 2.8e-02 of qpos divergence at the first step.
    """
    pytest.importorskip("mujoco_playground", reason="sim extra not installed")

    from quant_control_bench.envs import TIER1_ENV
    from quant_control_bench.export.scene import bundle_scene

    bundle = bundle_scene(TIER1_ENV, tmp_path)

    assert bundle.top_xml in bundle.files
    assert bundle.nq == 19 and bundle.nv == 18 and bundle.nu == 12
    assert bundle.n_substeps == round(bundle.ctrl_dt / bundle.sim_dt)
    assert bundle.actuator_gainprm_stride == 10
    assert bundle.actuator_biasprm_stride == 10

    interface = bundle.policy_interface
    assert interface is not None
    assert sum(width for _, width in interface["obs_layout"]) == 48
    assert set(interface["sensors"]) == {"local_linvel", "gyro", "upvector"}
    for slot in interface["sensors"].values():
        assert slot["dim"] == 3 and slot["adr"] >= 0
