"""Freeze scalar growth before refactoring (Gerrit branch 4a5871a).

Run once from the repository root. Deliberately refuses to overwrite the fixture.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxace.background import D_f_z, E_z


if __name__ == "__main__":
    path = Path(__file__).parent / "data" / "growth_scalar_reference.txt"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(exist_ok=True)
    zs = jnp.array([0., .5, 1., 3., 5.])
    rows = []
    for mass in (0., .06, .75):
        for species in ("cb", "m"):
            d, f = D_f_z(zs, .1424 / .67**2, .67, mν=mass, species=species)
            e = E_z(zs, .1424 / .67**2, .67, mν=mass)
            for i, z in enumerate(zs):
                rows.append([mass, int(species == "m"), z, e[i], d[i], f[i]])
    np.savetxt(path, rows, header=(
        "Gerrit growth-species-prescriptions 4a5871a; JAX 0.4.38 CPU x64\n"
        "h=.67 omega_cb=.1424 w0=-1 wa=0 Omega_k=0; original tolerances\n"
        "scalar_mass species_m z E D_unnormalized f"))
