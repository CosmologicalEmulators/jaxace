"""Paired CPU growth timings; compilation excluded and every result synchronized.

Run from a git checkout with Gerrit's 4a5871a reference available.
"""
import subprocess
import sys
import types

import jax
import jax.numpy as jnp
import pytest

from jaxace import background


@pytest.fixture(scope="module")
def original():
    source = subprocess.check_output(
        ["git", "show", "4a5871a:jaxace/background.py"], text=True
    )
    module = types.ModuleType("jaxace._growth_reference")
    module.__package__ = "jaxace"
    sys.modules[module.__name__] = module
    exec(compile(source, "growth_reference_4a5871a.py", "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("implementation", ["original", "current"])
@pytest.mark.parametrize("mode", ["primal", "gradient"])
def test_growth(benchmark, original, implementation, mode):
    module = original if implementation == "original" else background
    z = jnp.linspace(0., 5., 50)
    def prediction(x):
        return module.D_z(z, x[0], x[1], mν=x[2], species="m")
    fn = jax.jit(prediction if mode == "primal" else jax.grad(lambda x: prediction(x).sum()))
    x = jnp.array([.3, .67, .06])
    jax.block_until_ready(fn(x))
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)),
                       rounds=50, iterations=1, warmup_rounds=5)
