# Reproducing the benchmark

Everything below runs on a single 4 GB consumer GPU. Wall clocks are the ones measured on the reference machine (RTX 3050 Laptop, 4096 MiB); yours will differ, but the VRAM figures are the binding constraint and should not.

## 0. Setup

```bash
git clone https://github.com/Happynood/quant-control-bench
cd quant-control-bench
uv sync --all-extras
make env-check     # prints the accelerator and simulator stack actually in use
make verify        # lint, format, types, tests, 30-second end-to-end pipeline
```

`make verify` must be green before anything else. It runs the full pipeline in miniature on `CartpoleBalance` and will catch a broken install faster than a training run will.

Two environment settings matter and are exported by the Makefile:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

A third, `--xla_gpu_deterministic_ops=true`, is set by the package itself at import time. Do not unset it — see [`methodology.md`](methodology.md), section "Reproducibility of the simulator". Without it a contact-rich rollout is not reproducible run to run and every comparison in this benchmark is invalid.

## 1. Train the baselines

Tier 0 is the smoke tier and takes about an hour; Tier 1 is the headline policy.

```bash
uv run qcb train --config configs/tier0_cartpole.yaml --out artifacts/tier0-fp32
uv run qcb train --config configs/tier1_go1.yaml     --out artifacts/tier1-go1
```

Both use MuJoCo Playground's tuned hyperparameters unmodified. `--num-envs` and `--timesteps` override them; any reduction is recorded in `training.json` and surfaced by `budget_was_reduced`.

Training writes checkpoints every epoch. If a run dies, continue with:

```bash
uv run qcb train --config configs/tier1_go1.yaml --out artifacts/tier1-go1 \
    --restore-from artifacts/tier1-go1/checkpoints/<step>
```

Brax restores the network weights and the observation normalizer but **not** the optimizer state or the step counter, so a resumed run is not equivalent to an uninterrupted one of the same length. The training record says so in `restored_from`.

## 2. Export and check parity

```bash
uv run qcb export --policy artifacts/tier1-go1 --out artifacts/tier1-go1/policy.onnx
```

This fails the build if `‖π_onnx(s) − π_numpy(s)‖_∞ ≥ 1e-4` over 1000 random observations. Observation normalization is exported *into* the graph, so the ONNX file is self-contained.

## 3. The quantization sweep

Every scheme against the fp32 baseline, with paired bootstrap intervals:

```bash
uv run qcb sweep --config configs/tier1_go1.yaml \
    --policy artifacts/tier1-go1 \
    --output results/tier1_sweep.json \
    --table  results/tier1_sweep_table.md
```

The `fp32` row is a control, not a formality: it must come out at exactly +0.000% with a zero-width interval and `T_div` equal to the full horizon. If it does not, the harness is measuring noise and nothing below it can be trusted.

Add `--quantize-obs-norm` to quantize the observation-normalization statistics along with the weights (H4).

## 4. The robustness frontier

One invocation per axis, so a failure costs one axis rather than the whole sweep:

```bash
for axis in push_impulse mass_scale friction_scale actuator_delay obs_noise; do
    uv run qcb frontier --config configs/tier1_go1.yaml \
        --policy artifacts/tier1-go1 \
        --axis "$axis" \
        --output "results/tier1_frontier_${axis}.json"
done
```

Success is a return threshold set at half the **fp32 baseline's** unperturbed return and shared by every scheme. Grading each policy against its own return lets a collapsed policy pass by failing consistently — see [`methodology.md`](methodology.md).

## 5. Figures

```bash
uv run qcb plot --sweep results/tier1_sweep.json \
    --frontier results/tier1_frontier_*.json \
    --out plots --prefix tier1
```

Figures are rendered from the result JSON only. Nothing is recomputed, refitted or smoothed, so a plot cannot drift away from the table it illustrates.

## Result provenance

Every result file embeds a manifest: config SHA-256, git commit and dirty flag, the pinned versions of the simulator stack, the seeds, whether deterministic kernels were on, and the hardware. A result whose manifest says `git_dirty: true` was produced from uncommitted code and should be regenerated before it is cited.
