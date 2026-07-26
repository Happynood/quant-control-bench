"""Export a :class:`PolicyBundle` to ONNX opset 17.

The graph is built by hand rather than traced, because the thing being exported
is small, fully known, and must be byte-for-byte predictable — `onnxruntime-web`
runs it in a visitor's browser and there is no second chance to debug it there.

The observation normalization is part of the graph, not a preprocessing step
the caller is trusted to remember. The input is the raw observation and the
output is the actuator command.

    obs -> Sub(mean) -> Div(std) -> [Gemm -> act]* -> Gemm -> Slice(loc) -> Tanh
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx import TensorProto, checker, helper, numpy_helper

from quant_control_bench.export.bundle import PolicyBundle

OPSET = 17  # required by onnxruntime-web
INPUT_NAME = "obs"
OUTPUT_NAME = "action"

# Swish is not in this table: it has no single opset-17 operator and is emitted
# as Sigmoid + Mul below.
_ACTIVATION_OP = {"relu": "Relu", "tanh": "Tanh"}


def build_model(bundle: PolicyBundle, batch_dim: str = "batch"):
    """Build the ONNX ModelProto for `bundle`."""
    nodes = []
    initializers = []

    def constant(name: str, array: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    x = INPUT_NAME

    # ── observation normalization, inside the graph ───────────────────────────
    centered = "obs_centered"
    nodes.append(
        helper.make_node(
            "Sub", [x, constant("norm_mean", bundle.norm_mean.astype(np.float32))], [centered]
        )
    )
    x = "obs_normalized"
    nodes.append(
        helper.make_node(
            "Div", [centered, constant("norm_std", bundle.norm_std.astype(np.float32))], [x]
        )
    )

    # ── MLP ───────────────────────────────────────────────────────────────────
    for i, (w, b) in enumerate(zip(bundle.kernels, bundle.biases, strict=True)):
        if bundle.activation_scales is not None:
            # Activation fake-quantization has to be in the graph too. Leaving it
            # to the caller would make the browser run a different function from
            # the one the closed-loop numbers were measured on.
            scale = float(np.asarray(bundle.activation_scales[i]))
            limit = float(bundle.activation_qmax)
            scaled, rounded, clipped = (
                f"act_{i}_scaled",
                f"act_{i}_rounded",
                f"act_{i}_clipped",
            )
            scale_name = constant(f"act_scale_{i}", np.array(scale, dtype=np.float32))
            nodes.append(helper.make_node("Div", [x, scale_name], [scaled]))
            nodes.append(helper.make_node("Round", [scaled], [rounded]))
            nodes.append(
                helper.make_node(
                    "Clip",
                    [
                        rounded,
                        constant(f"act_lo_{i}", np.array(-limit, dtype=np.float32)),
                        constant(f"act_hi_{i}", np.array(limit, dtype=np.float32)),
                    ],
                    [clipped],
                )
            )
            x = f"act_{i}_dequant"
            nodes.append(helper.make_node("Mul", [clipped, scale_name], [x]))

        pre = f"layer_{i}_pre"
        nodes.append(
            helper.make_node(
                "Gemm",
                [
                    x,
                    constant(f"W_{i}", w.astype(np.float32)),
                    constant(f"B_{i}", b.astype(np.float32)),
                ],
                [pre],
                alpha=1.0,
                beta=1.0,
                transA=0,
                transB=0,  # kernels are stored (in_features, out_features)
            )
        )
        is_last = i == bundle.num_layers - 1
        if is_last:
            x = pre
            break
        x = f"layer_{i}_out"
        nodes.extend(_activation_nodes(bundle.activation, pre, x, i))

    # ── deterministic head: tanh of the first half of the logits ──────────────
    loc = "loc"
    nodes.append(
        helper.make_node(
            "Slice",
            [
                x,
                constant("loc_start", np.array([0], dtype=np.int64)),
                constant("loc_end", np.array([bundle.action_dim], dtype=np.int64)),
                constant("loc_axis", np.array([-1], dtype=np.int64)),
            ],
            [loc],
        )
    )
    nodes.append(helper.make_node("Tanh", [loc], [OUTPUT_NAME]))

    graph = helper.make_graph(
        nodes,
        f"{bundle.env}_policy",
        inputs=[
            helper.make_tensor_value_info(
                INPUT_NAME, TensorProto.FLOAT, [batch_dim, bundle.obs_dim]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                OUTPUT_NAME, TensorProto.FLOAT, [batch_dim, bundle.action_dim]
            )
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="quant-control-bench",
    )
    model.ir_version = 8  # onnxruntime-web does not accept newer IR versions
    checker.check_model(model)
    return model


def _activation_nodes(kind: str, src: str, dst: str, index: int) -> list:
    """Activation as ONNX nodes.

    Swish (x * sigmoid(x)) has no single opset-17 operator, so it is emitted as
    Sigmoid followed by Mul. `HardSwish` is deliberately *not* used: it is a
    piecewise-linear approximation, and substituting it would change every
    action by an amount comparable to the quantization error this project is
    trying to measure.
    """
    if kind == "swish":
        sig = f"layer_{index}_sigmoid"
        return [
            helper.make_node("Sigmoid", [src], [sig]),
            helper.make_node("Mul", [src, sig], [dst]),
        ]
    op = _ACTIVATION_OP[kind]
    return [helper.make_node(op, [src], [dst])]


def export_onnx(bundle: PolicyBundle, path: str | Path) -> Path:
    import onnx

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_model(bundle), p)
    return p


def run_onnx(path: str | Path, obs: np.ndarray) -> np.ndarray:
    """Run an exported policy through onnxruntime (CPU)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run([OUTPUT_NAME], {INPUT_NAME: np.asarray(obs, dtype=np.float32)})
    return np.asarray(out[0])
