from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_control_bench.config import QcbConfig, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_default_config_is_tier0_fp32_baseline() -> None:
    cfg = QcbConfig()
    assert cfg.env == "CartpoleBalance"
    assert cfg.schemes == ["fp32"]
    assert cfg.eval.seeds == [0, 1, 2, 3, 4]


def test_eval_seeds_are_fixed_constants_not_time_derived() -> None:
    # The spec forbids time-derived seeds; two constructions must agree.
    assert QcbConfig().eval.seeds == QcbConfig().eval.seeds


@pytest.mark.parametrize("path", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_shipped_configs_load(path: str) -> None:
    cfg = load_config(CONFIGS / path)
    assert cfg.name
    assert "fp32" in cfg.schemes, "every config must include the fp32 baseline"


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QcbConfig.model_validate({"schemes": ["int3-magic"]})


def test_unknown_perturbation_axis_is_rejected() -> None:
    with pytest.raises(ValidationError):
        QcbConfig.model_validate({"perturbations": ["gravity_scale"]})


def test_bootstrap_defaults_match_spec() -> None:
    b = QcbConfig().bootstrap
    assert b.resamples == 10_000
    assert b.confidence == 0.95
