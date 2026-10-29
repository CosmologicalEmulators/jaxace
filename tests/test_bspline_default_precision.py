"""Exercise the package's x64 opt-out independently of suite-global config."""
import os
import subprocess
import sys


def test_float32_opt_out_public_paths_in_fresh_process():
    source = r'''
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import make_interp_spline
from jaxace import CubicBSplinePlan, CubicBSplineBasis, basis_matrix, basis_stencil
assert not jax.config.jax_enable_x64
t = jnp.array([0.,.1,.2,.35,.6,.75,.88,1.])
q = jnp.array([-.1,0.,.33,.71,1.,1.1])
u = jnp.stack((t**3+2*t,2*(t**3+2*t)),axis=1)
assert t.dtype == jnp.float32
for policy in ('clamp','zero'):
    run = jax.jit(lambda values,sites,queries:CubicBSplinePlan(sites,queries,policy)(values))
    result = run(u,t,q)
    bounded = np.clip(np.asarray(q),0,1)
    operator = make_interp_spline(np.asarray(t),np.eye(len(t)))(bounded)
    inside = (np.asarray(q)>=0)&(np.asarray(q)<=1)
    if policy == 'zero': operator[~inside] = 0
    np.testing.assert_allclose(result,operator@np.asarray(u),rtol=8e-5,atol=8e-6)
    gradient = jax.jit(jax.grad(lambda values:jnp.sum(run(values,t,q)**2)))(u)
    np.testing.assert_allclose(gradient,2*operator.T@(operator@np.asarray(u)),rtol=8e-5,atol=8e-6)
    query_gradient = jax.jit(jax.grad(lambda queries:jnp.sum(run(u,t,queries))))(q)
    expected = np.where(inside,3*(2+3*bounded**2),0)
    np.testing.assert_allclose(query_gradient,expected,rtol=8e-5,atol=8e-6)
    assert result.dtype == gradient.dtype == query_gradient.dtype == jnp.float32
# int32/uint32 must not wrap through JAX's x64-disabled promotion to int32.
knots = jnp.array([0]*4 + [2**32-1]*4, dtype=jnp.uint32)
queries = jnp.array([-1, 0, 2**30, 2**31-1], dtype=jnp.int32)
basis = CubicBSplineBasis(knots)
r = np.asarray(queries, dtype=np.float64) / (2**32-1)
bernstein = np.stack(((1-r)**3, 3*r*(1-r)**2, 3*r*r*(1-r), r**3), axis=1)
bernstein[0] = 0
np.testing.assert_allclose(jax.jit(basis_matrix)(basis, queries), bernstein, rtol=2e-6, atol=2e-7)
stencil = jax.jit(lambda b, z: basis_stencil(b, z, 'clamp'))(basis, queries)
bernstein[0] = [1, 0, 0, 0]
np.testing.assert_allclose(stencil.weights, bernstein, rtol=2e-6, atol=2e-7)
print('DEFAULT_FLOAT32_PASS')
'''
    environment = dict(os.environ,JAX_ENABLE_X64="0",JAXACE_ENABLE_X64="false")
    result = subprocess.run([sys.executable,"-c",source],env=environment,
                            capture_output=True,text=True,timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    assert "DEFAULT_FLOAT32_PASS" in result.stdout
