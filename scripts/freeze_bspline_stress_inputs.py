"""Save exact Float64 inputs for independent higher-precision Julia checks."""
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1] / "tests/data/bspline_banded"
OUT = ROOT / "stress_highprecision"
if OUT.exists():
    raise RuntimeError("stress references already exist; never overwrite")
OUT.mkdir()
log_grid = jnp.asarray(np.loadtxt(ROOT / "julia_n81_inputs.txt")[:, 0])
for name, t in (("raw_k", jnp.exp(log_grid)),
                ("clustered", jnp.array([0., 1e-8, .2, .35, .5, .8, .9, 1.]))):
    u = jnp.stack((jnp.sin(t), jnp.cos(t)), axis=1)
    np.savetxt(OUT / f"{name}_inputs.txt", np.column_stack((t, u)), fmt="%.17g")
