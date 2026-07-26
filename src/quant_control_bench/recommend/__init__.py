"""Precision recommender: the cheapest scheme meeting a robustness and latency budget."""

from quant_control_bench.recommend.precision import (
    Candidate,
    Recommendation,
    Requirement,
    recommend,
    render,
)

__all__ = ["Candidate", "Recommendation", "Requirement", "recommend", "render"]
