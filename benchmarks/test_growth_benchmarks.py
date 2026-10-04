"""Paired CPU growth timings; compilation excluded and every result synchronized.

Run from a git checkout with pre-review dff94dc available.
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
        ["git", "show", "dff94dc:jaxace/background.py"], text=True
    )
    module = types.ModuleType("jaxace._growth_reference")
    module.__package__ = "jaxace"
    sys.modules[module.__name__] = module
    exec(compile(source, "growth_reference_dff94dc.py", "exec"), module.__dict__)
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


@pytest.mark.parametrize("policy", ["temperature", "radiation"])
@pytest.mark.parametrize("species", ["cb", "m"])
@pytest.mark.parametrize("mode", ["primal", "gradient"])
@pytest.mark.parametrize("implementation", ["original", "current"])
def test_neff_growth(benchmark, original, implementation, policy, species, mode):
    module = original if implementation == "original" else background
    z=jnp.linspace(0.,5.,50)
    def prediction(x):
        return module.D_z(z,x[0],x[1],mν=x[2:5],Neff=x[5],
            neutrino_prescription=policy,species=species,reltol=1e-10,abstol=1e-12)
    fn=jax.jit(prediction if mode=="primal" else jax.grad(lambda x: prediction(x).sum()))
    x=jnp.array([.31,.67,0.,.02,.05,4.])
    jax.block_until_ready(fn(x))
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)),
                       rounds=50,iterations=1,warmup_rounds=5)


@pytest.mark.parametrize("implementation", ["original", "current"])
@pytest.mark.parametrize("observable", ["E_z", "r_z"])
@pytest.mark.parametrize("mode", ["primal", "gradient"])
def test_neff_background(benchmark, original, implementation, observable, mode):
    module = original if implementation == "original" else background
    z = jnp.linspace(0., 5., 50)
    def prediction(x):
        return getattr(module, observable)(z, x[0], x[1], mν=x[2:5], Neff=x[5])
    fn = jax.jit(prediction if mode == "primal" else jax.grad(lambda x: prediction(x).sum()))
    x = jnp.array([.31, .67, 0., .02, .05, 4.])
    jax.block_until_ready(fn(x))
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)),
                       rounds=50, iterations=1, warmup_rounds=5)
