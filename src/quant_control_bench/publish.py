"""Assemble the publishable bundles: the HF model repo and the HF Space.

Both are built from files already in the repository rather than regenerated, so
what gets published is exactly what was measured. The manifest ties every artifact
to the run that produced it, and the model card's numbers are read out of the
result JSON rather than transcribed by hand — a transcribed table is a table that
drifts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

MODEL_REPO = "happynood/quant-control-bench-policies"
SPACE_REPO = "happynood/quant-control-bench-demo"
GITHUB_REPO = "https://github.com/Happynood/quant-control-bench"

# Where the Space fetches policies from. There is exactly one source of
# truth for weights, so the deployed page loads them from the model repo rather
# than from a copy bundled beside it.
MODEL_BASE_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/onnx"


# Artwork the cards reference. Copied into each bundle rather than hot-linked
# back to GitHub, so a card renders even if the repository moves.
ASSETS = ("logo.svg", "demo.gif")


def _copy_assets(root: Path, out: Path) -> None:
    (out / "assets").mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        source = root / "docs" / "assets" / name
        if source.exists():
            shutil.copy2(source, out / "assets" / name)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing input: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _sweep_rows(root: Path) -> list[dict[str, Any]]:
    payload = json.loads((root / "results" / "tier1_sweep.json").read_text())
    return payload["schemes"]


def _frontier_rows(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "results").glob("tier1_frontier_*.json")):
        for row in json.loads(path.read_text())["frontiers"]:
            out.setdefault(row["scheme"], {})[row["axis"]] = row
    return out


def build_model_repo(root: Path, out: Path) -> dict[str, Any]:
    """Checkpoint, every ONNX variant, the scene, and a manifest tying them together."""
    out.mkdir(parents=True, exist_ok=True)

    policy_dir = root / "artifacts" / "tier1-go1"
    for name in ("policy.npz", "policy.json", "training.json", "manifest.json"):
        source = policy_dir / name
        if not source.exists():
            raise FileNotFoundError(f"missing trained artifact: {source}")
        shutil.copy2(source, out / f"checkpoint_{name}" if name != "policy.npz" else out / name)

    _copy_tree(root / "web" / "assets" / "onnx", out / "onnx")
    _copy_tree(root / "web" / "assets" / "scene", out / "scene")
    _copy_assets(root, out)

    training = json.loads((policy_dir / "training.json").read_text())
    run_manifest = json.loads((policy_dir / "manifest.json").read_text())
    variants = json.loads((out / "onnx" / "variants.json").read_text())

    manifest = {
        "env": training["env"],
        "github": GITHUB_REPO,
        "training": {
            "seed": training["seed"],
            "num_timesteps_executed": training["num_timesteps_actual"],
            "num_timesteps_tuned": training["num_timesteps_tuned"],
            "num_envs": training["num_envs"],
            "wall_clock_s": training["wall_clock_s"],
            "peak_vram_mb": training["peak_vram_mb"],
            "final_reward": training["final_reward"],
            "final_reward_std": training["final_reward_std"],
            "extraction_max_abs_error": training["extraction_max_abs_error"],
            "restored_from": training["restored_from"],
        },
        "provenance": {
            "git_commit": run_manifest.get("git_commit"),
            "git_dirty": run_manifest.get("git_dirty"),
            "package_versions": run_manifest.get("package_versions"),
            "gpu": run_manifest.get("gpu"),
            "deterministic_ops": run_manifest.get("deterministic_ops"),
        },
        "artifacts": {
            "checkpoint": ["policy.npz", "checkpoint_policy.json"],
            "onnx": {k: f"onnx/{v['file']}" for k, v in variants["variants"].items()},
            "scene": "scene/",
        },
        "onnx_parity": {
            k: {
                "visited_states": v["onnx_parity_visited_states"],
                "gaussian": v["onnx_parity_gaussian"],
            }
            for k, v in variants["variants"].items()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "README.md").write_text(model_card(root))
    return manifest


def model_card_inputs(root: Path) -> list[Path]:
    """Files :func:`model_card` reads.

    Declared here rather than restated by callers. `results/` is committed but
    `artifacts/` is not — the trained checkpoint is 200 MB of run output — so on a
    clean checkout the card is not generatable, and anything that wants to skip
    rather than crash needs the *whole* list. Listing one of the two by hand is
    exactly what broke CI: the guard checked `results/` and the card then reached
    for `artifacts/`.
    """
    return [
        root / "results" / "tier1_sweep.json",
        root / "artifacts" / "tier1-go1" / "training.json",
    ]


def model_card_is_generatable(root: Path) -> bool:
    return all(path.exists() for path in model_card_inputs(root))


def model_card(root: Path) -> str:
    """Model card, with the table generated from the measured results."""
    rows = _sweep_rows(root)
    frontiers = _frontier_rows(root)
    training = json.loads((root / "artifacts" / "tier1-go1" / "training.json").read_text())

    lines = [
        "---",
        "license: mit",
        "library_name: onnx",
        "tags:",
        "  - robotics",
        "  - reinforcement-learning",
        "  - quantization",
        "  - mujoco",
        "pipeline_tag: robotics",
        "---",
        "",
        '<p align="center"><img src="assets/logo.svg" alt="quant-control-bench" width="820"></p>',
        "",
        "# Go1 locomotion policies at nine precisions",
        "",
        f'<p align="center"><a href="https://huggingface.co/spaces/{SPACE_REPO}">'
        "<b>Try them in your browser</b></a> · "
        f'<a href="{GITHUB_REPO}"><b>Code and methodology</b></a></p>',
        "",
        '<p align="center"><img src="assets/demo.gif" alt="three precisions walking" '
        'width="820"></p>',
        "",
        "One PPO policy for `Go1JoystickFlatTerrain`, exported to ONNX nine times at",
        "different weight precisions, plus the scene it was trained in. The point of the",
        "set is the comparison: how far precision can be reduced before a closed-loop",
        "controller stops working, and where that boundary actually sits.",
        "",
        f"Full methodology, code and raw results: {GITHUB_REPO}",
        "",
        "## Results",
        "",
        "100 episodes x 5 fixed seeds, 1000-step horizon, deterministic policy (mean",
        "action, no sampling). Return deltas are paired bootstrap 95% intervals over",
        "10 000 resamples. `P50` is the perturbation magnitude at which the success rate",
        "crosses 50%; on `friction_scale`, swept downward, a larger number is worse.",
        "",
        "| scheme | bits/weight | mean return | Δreturn vs fp32 (95% CI) "
        "| P50 friction | P50 obs noise |",
        "|---|---|---|---|---|---|",
    ]

    for row in rows:
        scheme = row["scheme"]
        rel = row["relative_delta_return"]
        delta = (
            "—"
            if scheme == "fp32"
            else (
                "no measurable loss"
                if rel["ci_low"] <= 0.0 <= rel["ci_high"]
                else f"{rel['estimate']:+.3%} [{rel['ci_low']:+.2%}, {rel['ci_high']:+.2%}]"
            )
        )
        axes = frontiers.get(scheme, {})

        # `axes` is bound as a default so the closure cannot capture the loop
        # variable and report one scheme's frontier under another's name.
        def p50(axis: str, axes: dict[str, Any] = axes) -> str:
            entry = axes.get(axis)
            return "n/a (not run)" if entry is None else f"{entry['p50']:.3f}"

        lines.append(
            f"| `{scheme}` | {row['quantization']['mean_bits_per_weight']:.2f} "
            f"| {row['mean_return']:.2f} | {delta} "
            f"| {p50('friction_scale')} | {p50('obs_noise')} |"
        )

    lines += [
        "",
        "## What the numbers say",
        "",
        "- **int8 is free.** Every int8 variant, including one that quantizes activations",
        "  as well as weights, is statistically indistinguishable from fp32 on return and",
        "  on all five robustness axes.",
        "- **The boundary is between int8 and int4, and grouping matters more than bits.**",
        "  `int4-channel` loses only 2.3% of return on flat ground yet its friction and",
        "  observation-noise frontiers separate from fp32 with non-overlapping intervals.",
        "  `int4-group32`, at the same 4.00 bits, does not.",
        "- **Quantizing the observation-normalization statistics is catastrophic.** They",
        "  are 0.05% of the parameters. Quantized with the same scheme as the weights,",
        "  every int8 variant becomes measurably lossy and every 4-bit variant stops",
        "  producing finite actions at all, because a strictly positive scale vector",
        "  quantized symmetrically rounds entries to zero and the policy divides by them.",
        "",
        "## Intended use and limitations",
        "",
        "These are research artifacts for studying quantization of closed-loop control.",
        "They are not tuned for deployment on hardware and have never been run on a",
        "physical robot.",
        "",
        "- **Quantization is simulated.** Weights are rounded to the target grid and",
        "  stored back as float32, so every ONNX file is the same size and none of them",
        "  runs faster. The benchmark measures what precision does to control, not what",
        "  it does to storage or throughput. A deployment would need packed kernels, and",
        "  the accuracy results here would carry over while the timing results would not.",
        "- **One training seed.** Precision effects cannot be fully separated from the",
        "  luck of a single checkpoint.",
        "- **One task, flat terrain.** `Go1JoystickFlatTerrain` only. The headline",
        "  hypothesis (that open-loop action error mispredicts closed-loop performance)",
        "  is *not* supported on this task, and that negative result is reported with the",
        "  same prominence as the positive ones.",
        "- **Browser physics differs from training physics.** The demo runs MuJoCo 3.3.8",
        "  compiled to WebAssembly; training used MJX with MuJoCo 3.10.0. Measured",
        "  divergence and the reasoning behind it are in the repository's methodology.",
        "",
        "## Training",
        "",
        "| | |",
        "|---|---|",
        f"| Environment steps executed | {training['num_timesteps_actual']:,} |",
        f"| Wall clock | {training['wall_clock_s'] / 60:.1f} min |",
        f"| Parallel environments | {training['num_envs']:,} |",
        f"| Peak VRAM | {training['peak_vram_mb']} MiB |",
        f"| Final training reward | {training['final_reward']:.3f} "
        f"± {training['final_reward_std']:.3f} |",
        f"| Weight extraction vs Brax | {training['extraction_max_abs_error']:.2e} |",
        "| Hardware | NVIDIA RTX 3050 Laptop, 4096 MiB |",
        "",
        "Every ONNX graph carries the observation normalization inside it, so the input",
        "is the raw 48-dim observation and the output is the tanh-squashed action.",
        "",
    ]
    return "\n".join(lines)


def build_space(root: Path, out: Path) -> None:
    """The static Space: the demo, pointed at the model repo for its weights."""
    out.mkdir(parents=True, exist_ok=True)
    web = root / "web"

    for name in ("index.html", "style.css", "favicon.svg"):
        shutil.copy2(web / name, out / name)
    _copy_assets(root, out)
    _copy_tree(web / "src", out / "src")
    _copy_tree(web / "vendor", out / "vendor")
    _copy_tree(web / "assets" / "scene", out / "assets" / "scene")

    # Weights are *not* copied. The spec requires one source of truth for them, so
    # the page fetches from the model repo; `variants.json` travels with them.
    config = f'window.QCB_MODEL_BASE = "{MODEL_BASE_URL}";\n'
    (out / "model-base.js").write_text(config)

    html = (out / "index.html").read_text()
    html = html.replace(
        '<script type="module" src="./src/app.js"></script>',
        '<script src="./model-base.js"></script>\n'
        '    <script type="module" src="./src/app.js"></script>',
    )
    (out / "index.html").write_text(html)

    (out / "README.md").write_text(space_card())


def space_card() -> str:
    return "\n".join(
        [
            "---",
            "title: quant-control-bench — how few bits before it falls over?",
            "emoji: 🦿",
            "colorFrom: green",
            "colorTo: gray",
            "sdk: static",
            "pinned: false",
            "license: mit",
            "---",
            "",
            '<p align="center"><img src="assets/logo.svg" alt="quant-control-bench" '
            'width="820"></p>',
            "",
            "# How few bits does a closed-loop robot controller need?",
            "",
            '<p align="center"><img src="assets/demo.gif" alt="three precisions walking" '
            'width="820"></p>',
            "",
            "The same trained Go1 policy, quantized several ways, walking in real MuJoCo",
            "physics compiled to WebAssembly. Everything runs in your browser: the",
            "simulation, the neural networks, and the timing.",
            "",
            "Pick precisions, drive them with the joystick sliders, then break them:",
            "push the robots, change torso mass and ground friction, add actuator delay",
            "or observation noise. Every variant shares an initial state and a command,",
            "so the only difference between them is the precision of the weights.",
            "",
            f"Weights are loaded from [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}),",
            "which is the single source of truth for them.",
            "",
            f"Methodology, raw results and the code: {GITHUB_REPO}",
            "",
            "## What the benchmark found",
            "",
            "- int8 costs nothing measurable, on return or on any of five robustness axes.",
            "- The interesting boundary is between int8 and int4 — and at 4 bits, *how* you",
            "  group the scales matters more than the bit width does.",
            "- Quantizing the observation-normalization statistics, 0.05% of the parameters,",
            "  breaks the policy outright at 4 bits.",
            "",
            "The live figures on this page (steps survived, whether the robot fell, action",
            "jitter, inference milliseconds) are computed in your browser. The benchmark",
            "tables come from recorded runs on an RTX 3050 Laptop and are not recomputed",
            "here.",
            "",
        ]
    )
