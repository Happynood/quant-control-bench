# Structure adapted from Happynood/quant-reasoning-bench (Makefile).
.PHONY: install install-sim install-all verify lint lint-check format format-check \
        typecheck test smoke env-check clean

# MJX memory protocol: never preallocate the whole device, cap at 90%.
export XLA_PYTHON_CLIENT_PREALLOCATE = false
export XLA_PYTHON_CLIENT_MEM_FRACTION = 0.9

SMOKE_RESULT   = /tmp/qcb-smoke-result.json
SMOKE_MANIFEST = /tmp/qcb-smoke-manifest.json

# ── Dependencies ──────────────────────────────────────────────────────────────

install:
	uv sync

install-sim:
	uv sync --extra sim

install-all:
	uv sync --all-extras


# ── Quality gate (must be green before every push) ────────────────────────────

verify: install-all lint-check format-check typecheck test smoke

lint-check:
	uv run ruff check .

format-check:
	uv run ruff format --check .

typecheck:
	uv run pyright

test:
	uv run pytest -q -m "not slow"

smoke:
	uv run qcb smoke --config configs/smoke.yaml \
		--output $(SMOKE_RESULT) --manifest $(SMOKE_MANIFEST)
	@uv run python -c "import json,sys; \
d=json.load(open('$(SMOKE_RESULT)')); \
assert d['env']=='CartpoleBalance', 'wrong env'; \
assert d['manifest']['mjx_impl']=='jax', 'wrong mjx impl'; \
assert d['onnx_parity_ok'], 'ONNX parity broke: %.3e' % d['onnx_parity_max_abs']; \
assert d['mean_return'] > 0.9 * d['steps'], \
  'trained smoke policy no longer balances: %.1f over %d steps' % (d['mean_return'], d['steps']); \
print('smoke result OK: return=%.1f parity=%.2e' % (d['mean_return'], d['onnx_parity_max_abs']))"

# ── Dev helpers ───────────────────────────────────────────────────────────────

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

env-check:
	uv run qcb env-check

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ $(SMOKE_RESULT) $(SMOKE_MANIFEST)
