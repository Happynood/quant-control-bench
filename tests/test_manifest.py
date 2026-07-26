import json
from dataclasses import asdict
from pathlib import Path

from quant_control_bench.config import load_config
from quant_control_bench.manifest import collect_manifest, write_manifest

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


def test_manifest_pins_the_run_to_a_config_hash_and_impl() -> None:
    cfg = load_config(SMOKE_CONFIG)
    m = collect_manifest(SMOKE_CONFIG, cfg)
    assert len(m.config_sha256) == 64
    assert m.mjx_impl == "jax"
    assert m.env == "CartpoleBalance"
    assert m.eval_seeds == [0, 1, 2, 3, 4]


def test_manifest_records_simulator_stack_versions() -> None:
    cfg = load_config(SMOKE_CONFIG)
    m = collect_manifest(SMOKE_CONFIG, cfg)
    for pkg in ("jax", "mujoco", "numpy"):
        assert pkg in m.package_versions


def test_manifest_round_trips_to_json(tmp_path: Path) -> None:
    cfg = load_config(SMOKE_CONFIG)
    m = collect_manifest(SMOKE_CONFIG, cfg)
    out = tmp_path / "nested" / "manifest.json"
    write_manifest(m, out)
    assert json.loads(out.read_text()) == json.loads(json.dumps(asdict(m)))
