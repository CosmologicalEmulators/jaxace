"""Save the pre-change dense JAX results after the Julia input fixtures exist."""
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from jaxace import CubicBSpline, CubicBSplinePlan
import jaxace.utils as utils

jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/data/bspline_banded"
EXPECTED_UTILS_SHA256 = "7bc002c1ebad80f2e3d281b959488aef1ea6e3469ad554aed0882ce254354d22"
if (OUT / "jax_metadata.json").exists() or list(OUT.glob("jax_n*.txt")):
    raise RuntimeError("dense reference already exists; never overwrite frozen outputs")
if hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest() != EXPECTED_UTILS_SHA256:
    raise RuntimeError("reference generation requires the original dense utils.py")
for n in (4, 5, 6, 8, 81):
    inputs = np.loadtxt(OUT / f"julia_n{n}_inputs.txt")
    t, u, g = jnp.array(inputs[:, 0]), jnp.array(inputs[:, 1:4]), jnp.array(inputs[:, 4:])
    q = jnp.array(np.loadtxt(OUT / f"julia_n{n}_query.txt"))
    fields = {
        "coefficients": CubicBSpline(u, t).coefficients,
        "values": CubicBSplinePlan(t, q)(u),
        "grad_sites": jax.grad(lambda x: jnp.sum(CubicBSpline(u, x).coefficients * g))(t),
        "grad_values": jax.grad(lambda x: jnp.sum(CubicBSpline(x, t).coefficients * g))(u),
    }
    for name, value in fields.items():
        np.savetxt(OUT / f"jax_n{n}_{name}.txt", np.asarray(value), header=f"Original dense JAX; N={n}; {name}")
metadata = {
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "utils_sha256": hashlib.sha256(Path(utils.__file__).read_bytes()).hexdigest(),
    "jax_version": jax.__version__, "x64": True, "rhs_columns": 3,
    "cases": [4, 5, 6, 8, 81],
}
(OUT / "jax_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("Frozen dense JAX references:", OUT)
