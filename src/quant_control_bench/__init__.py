"""quant-control-bench: post-training quantization of closed-loop control policies."""

from quant_control_bench.precision import enforce_deterministic_ops

# Runs at import, before anything in this package can touch JAX. XLA reads its
# flags once, when the backend initializes; setting this lazily at the first
# rollout would be too late and would silently leave the simulator
# non-reproducible. See quant_control_bench.precision for the measurement.
enforce_deterministic_ops()

__version__ = "0.1.0"

__all__ = ["__version__"]
