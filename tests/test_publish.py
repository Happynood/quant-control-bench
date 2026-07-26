"""Publishable bundle assembly.

The model card is generated from the result JSON rather than transcribed, because
a transcribed table drifts from the data it claims to describe. These tests pin
the generation, not the prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quant_control_bench.publish import (
    MODEL_BASE_URL,
    MODEL_REPO,
    model_card,
    model_card_inputs,
    model_card_is_generatable,
    space_card,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Asked of the module rather than restated here. `results/` is committed and
# `artifacts/` is not, so a clean checkout has one but not the other; a guard that
# checked only the committed half let these tests run in CI and crash on the
# missing one.
HAS_RESULTS = model_card_is_generatable(REPO_ROOT)


def test_space_card_declares_the_static_sdk() -> None:
    """A Space that declares the wrong SDK does not build."""
    card = space_card()
    assert card.startswith("---\n")
    assert "sdk: static" in card
    assert "license: mit" in card


def test_space_loads_weights_from_the_model_repo() -> None:
    """One source of truth for weights, by design."""
    assert MODEL_BASE_URL.startswith(f"https://huggingface.co/{MODEL_REPO}/resolve/")
    assert MODEL_REPO in space_card()


@pytest.mark.skipif(not HAS_RESULTS, reason="Tier 1 run artifacts not present")
def test_model_card_numbers_come_from_the_results() -> None:
    import json

    card = model_card(REPO_ROOT)
    payload = json.loads((REPO_ROOT / "results" / "tier1_sweep.json").read_text())
    rows: list[dict[str, Any]] = payload["schemes"]

    assert "sdk:" not in card.split("---")[1]  # model cards must not claim a Space SDK
    for row in rows:
        assert f"`{row['scheme']}`" in card
        assert f"{row['mean_return']:.2f}" in card


@pytest.mark.skipif(not HAS_RESULTS, reason="Tier 1 run artifacts not present")
def test_model_card_does_not_claim_a_speedup() -> None:
    """Quantization here is simulated: no file shrinks and nothing runs faster.

    A model card that let a reader assume otherwise would be the most likely way
    for this project to mislead someone.
    """
    card = model_card(REPO_ROOT).lower()
    assert "simulated" in card
    assert "same size" in card


@pytest.mark.skipif(not HAS_RESULTS, reason="Tier 1 run artifacts not present")
def test_model_card_reports_the_negative_result() -> None:
    """H1 was not supported; the card must say so rather than omit it."""
    assert "not* supported" in model_card(REPO_ROOT) or "not supported" in model_card(REPO_ROOT)


def test_the_skip_guard_covers_every_input_the_card_reads(tmp_path: Path) -> None:
    """A guard that names only some of its dependencies is the CI failure again.

    `artifacts/` is gitignored, so on a clean checkout the training record is
    absent while the results are present. The guard has to reject that state.
    """
    inputs = model_card_inputs(tmp_path)
    assert len(inputs) >= 2
    assert not model_card_is_generatable(tmp_path)

    for path in inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    assert model_card_is_generatable(tmp_path)

    # Removing any single input must be enough to fail the guard.
    for path in inputs:
        path.unlink()
        assert not model_card_is_generatable(tmp_path), path
        path.write_text("{}")
