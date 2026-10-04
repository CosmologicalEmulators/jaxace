"""Report CI-stack sensitivities against the frozen original-state oracle.

Run after tests/generate_original_growth_reference.py. Prints Markdown; redirect
to a new report rather than overwriting an earlier measurement.
"""
from pathlib import Path
import importlib.metadata as metadata
import jax
import jax.numpy as jnp
import numpy as np
from jaxace import background as bg


def predict(m,ocb,h,w0,wa,k,z,tol,species):
    return jnp.array(bg.D_f_z(z,ocb,h,mν=m,w0=w0,wa=wa,Ωk0=k,species=species,
                             reltol=tol,abstol=tol/100))


gradient = jax.jit(jax.jacrev(predict,argnums=0),static_argnames=("species",))


if __name__ == "__main__":
    rows = np.loadtxt(Path(__file__).parents[1]/"tests/data/scalar_growth_original_reference.txt")
    print("# Scalar growth sensitivity convergence\n",flush=True)
    print("Original D/Dprime NumPy/SciPy DOP853 oracle, restarted at every Akima knot; complex-step mass derivatives.\n")
    print("Environment: " + ", ".join(f"{p} {metadata.version(p)}" for p in ("jax","jaxlib","diffrax","quadax","scipy")) + ". CPU float64.\n")
    print("Both schemes use the frozen scalar thermal prescription; the public JAX polynomial table is rebuilt at import.\n")
    print("| source | mass | Ocb | z | reltol | dD/dm | abs error D | relative error D | df/dm | abs error f | relative error f |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        s,m,ocb,h,w0,wa,k,z,_,_,gd,gf = row
        if z != (2. if m == .1 else 1.):
            continue
        species = "m" if s else "cb"
        for tol in (1e-8,1e-9,1e-10,1e-11,1e-12,1e-13):
            actual = np.asarray(gradient(m,ocb,h,w0,wa,k,z,tol,species))
            error = abs(actual-np.array([gd,gf]))
            relative = error/abs(np.array([gd,gf]))
            print(f"| {species} | {m:g} | {ocb:g} | {z:g} | {tol:.0e} | {actual[0]:.10e} | {error[0]:.3e} | {relative[0]:.3e} | {actual[1]:.10e} | {error[1]:.3e} | {relative[1]:.3e} |",flush=True)
    print("\nTighter primal tolerances need not produce monotonically smaller sensitivity errors. Absolute errors matter for small/cancelled derivatives; the table is evidence, not a promise of monotonic convergence.")
