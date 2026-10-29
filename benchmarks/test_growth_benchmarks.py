"""Three-mass CPU timings; compilation excluded and every result synchronized.

Use pytest-benchmark --benchmark-save/--benchmark-compare across revisions.
Historical scalar implementations are not imported into the current process.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxace import background


@pytest.mark.parametrize("policy", ["temperature", "radiation"])
@pytest.mark.parametrize("species", ["cb", "m"])
@pytest.mark.parametrize("mode", ["primal", "gradient"])
def test_neff_growth(benchmark, policy, species, mode):
    benchmark.extra_info["source_file"] = background.__file__
    benchmark.extra_info["jax_version"] = jax.__version__
    z=jnp.linspace(0.,5.,50)
    def prediction(x):
        return background.D_z(z,x[0],x[1],mν=x[2:5],Neff=x[5],
            neutrino_prescription=policy,species=species,reltol=1e-10,abstol=1e-12)
    fn=jax.jit(prediction if mode=="primal" else jax.grad(lambda x: prediction(x).sum()))
    x=jnp.array([.31,.67,0.,.02,.05,4.])
    jax.block_until_ready(fn(x))
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)),
                       rounds=50,iterations=1,warmup_rounds=5)


@pytest.mark.parametrize("observable", ["E_z", "r_z"])
@pytest.mark.parametrize("mode", ["primal", "gradient"])
def test_neff_background(benchmark, observable, mode):
    benchmark.extra_info["source_file"] = background.__file__
    benchmark.extra_info["jax_version"] = jax.__version__
    z = jnp.linspace(0., 5., 50)
    def prediction(x):
        return getattr(background, observable)(z, x[0], x[1], mν=x[2:5], Neff=x[5])
    fn = jax.jit(prediction if mode == "primal" else jax.grad(lambda x: prediction(x).sum()))
    x = jnp.array([.31, .67, 0., .02, .05, 4.])
    jax.block_until_ready(fn(x))
    benchmark.pedantic(lambda: jax.block_until_ready(fn(x)),
                       rounds=50, iterations=1, warmup_rounds=5)
