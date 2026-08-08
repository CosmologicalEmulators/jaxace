"""Steady-state JAX benchmarks for reusable spline objects and plans."""

import jax
import jax.numpy as jnp
import pytest

from jaxace import (
    AkimaSplinePlan,
    CubicBSplinePlan,
    CubicSpline,
    CubicSplinePlan,
    akima_interpolation,
    cubic_b_spline_interpolation,
    cubic_spline_interpolation,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def spline_benchmark_data():
    """Build production-shaped fixed grids and compile each benchmark path."""
    k = jnp.arange(512)
    t = jnp.sort(2 + 0.5 * (jnp.cos(jnp.pi * k / 511) + 1) * (9000 - 2))
    u = jnp.exp(-t / 3000) * (1 + 0.1 * jnp.sin(t / 40))
    t_new = jnp.arange(2.0, 9001.0)

    akima_plan = AkimaSplinePlan(t, t_new)
    cubic_spline = CubicSpline(u, t)
    cubic_plan = CubicSplinePlan(t, t_new)

    functions = {
        "akima_pure": jax.jit(lambda values: akima_interpolation(values, t, t_new)),
        "akima_plan": jax.jit(akima_plan),
        "cubic_pure": jax.jit(
            lambda values: cubic_spline_interpolation(values, t, t_new)
        ),
        "cubic_plan": jax.jit(cubic_plan),
        "cubic_query_pure": jax.jit(
            lambda query: cubic_spline_interpolation(u, t, query)
        ),
        "cubic_query_prepared": jax.jit(cubic_spline),
    }

    arguments = {
        "akima_pure": u,
        "akima_plan": u,
        "cubic_pure": u,
        "cubic_plan": u,
        "cubic_query_pure": t_new,
        "cubic_query_prepared": t_new,
    }

    for name, function in functions.items():
        function(arguments[name]).block_until_ready()

    return functions, arguments


@pytest.mark.parametrize(
    "name",
    (
        "akima_pure",
        "akima_plan",
        "cubic_pure",
        "cubic_plan",
        "cubic_query_pure",
        "cubic_query_prepared",
    ),
)
def test_spline_runtime(benchmark, spline_benchmark_data, name):
    """Benchmark compiled execution with compilation excluded and synchronized."""
    functions, arguments = spline_benchmark_data
    function = functions[name]
    argument = arguments[name]

    def run():
        result = function(argument)
        result.block_until_ready()
        return result

    benchmark.pedantic(run, rounds=30, iterations=1, warmup_rounds=5)
    benchmark.extra_info["source_nodes"] = 512
    benchmark.extra_info["target_nodes"] = 8999
    benchmark.extra_info["jit"] = True
    benchmark.extra_info["device"] = str(jax.devices()[0])


@pytest.mark.parametrize(
    ("case", "n_query", "matrix"),
    (
        ("vec_40_40", 40, False),
        ("vec_40_8192", 8192, False),
        ("mat_40x161_8192", 8192, True),
    ),
)
@pytest.mark.parametrize("path", ("on_the_fly", "plan"))
@pytest.mark.parametrize("operation", ("forward", "gradient"))
def test_cubic_b_spline_comparison_runtime(
    benchmark,
    case,
    n_query,
    matrix,
    path,
    operation,
):
    """Benchmark the same shapes and scalar loss as the Julia comparison."""
    n_sites = 40
    k = jnp.arange(n_sites)
    t = jnp.sort(2 + 0.5 * (jnp.cos(jnp.pi * k / (n_sites - 1)) + 1) * (9000 - 2))
    query = jnp.linspace(t[0], t[-1], n_query)
    values = jnp.exp(-t / 3000) * (1 + 0.1 * jnp.sin(t / 40))
    if matrix:
        values = jnp.column_stack(
            tuple(values * (1 + 0.001 * column) for column in range(1, 162))
        )

    if path == "on_the_fly":

        def evaluate(ordinates):
            return cubic_b_spline_interpolation(ordinates, t, query)

    else:
        plan = CubicBSplinePlan(t, query)
        evaluate = plan

    if operation == "gradient":
        function = jax.jit(
            jax.grad(lambda ordinates: jnp.sum(evaluate(ordinates) ** 2))
        )
    else:
        function = jax.jit(evaluate)

    function(values).block_until_ready()

    def run():
        result = function(values)
        result.block_until_ready()
        return result

    benchmark.pedantic(run, rounds=30, iterations=1, warmup_rounds=5)
    benchmark.extra_info["case"] = case
    benchmark.extra_info["path"] = path
    benchmark.extra_info["operation"] = operation
    benchmark.extra_info["source_nodes"] = n_sites
    benchmark.extra_info["target_nodes"] = n_query
    benchmark.extra_info["series"] = 161 if matrix else 1
    benchmark.extra_info["jit"] = True
    benchmark.extra_info["device"] = str(jax.devices()[0])
