"""Check benchmark execution contracts without collecting performance numbers."""
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from benchmarks.test_bspline_parity_benchmarks import compiled_cases as _cases


def test_eager_preparation_benchmark_does_not_trace_away_host_validation():
    calls = []

    class Plan(NamedTuple):
        query: object

        def __call__(self, values):
            return jnp.ones_like(self.query) * values[0]

    def construct(sites, query):
        assert not isinstance(sites, jax.core.Tracer)
        assert not isinstance(query, jax.core.Tracer)
        calls.append((sites.shape, query.shape))
        return Plan(query)

    module = SimpleNamespace(CubicBSplinePlan=construct, cubic_b_spline_interpolation=None)
    prepare = _cases.__wrapped__(({name: module for name in ("dense", "banded_dense", "compact")}, ""))
    functions = prepare("tiny", "plan_construction_eager")
    assert len(calls) == 3  # One untimed warmup per implementation.
    for function, arguments, _ in functions.values():
        assert function is construct
        jax.block_until_ready(function(*arguments))
    assert len(calls) == 6  # Every measured invocation still performs host work.
