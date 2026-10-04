"""Report precision against saved Julia outputs without regenerating references."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

from jaxace import background as bg

path = Path(__file__).parents[1]/"tests/data/neutrino_neff/julia_growth_reference.txt"
groups = {}
for line in path.read_text().splitlines():
    if line.startswith("#"):
        continue
    policy,species,*values = line.split()
    row = np.array(values,dtype=float)
    groups.setdefault((policy,species,*row[:4]),[]).append(row)

h=.67
ocb=.1424/h**2
def loss(x,policy,species):
    return bg.D_z(jnp.array([0.,.5,1.,3.,5.]),ocb,h,mν=x[:3],Neff=x[3],
        neutrino_prescription=policy,species=species,reltol=1e-10,abstol=1e-12).sum()
gradient = jax.jit(jax.grad(loss),static_argnames=("policy","species"))
report = {}
for (policy,species,n,m1,m2,m3),rows in groups.items():
    rows=np.array(rows); z=jnp.asarray(rows[:,4]); x=jnp.array([m1,m2,m3,n])
    kw=dict(mν=x[:3],Neff=n,neutrino_prescription=policy)
    d,f=bg.D_f_z(z,ocb,h,**kw,species=species,reltol=1e-10,abstol=1e-12)
    outputs={"E":(bg.E_z(z,ocb,h,**kw),rows[:,5]),
             "distance_Mpc":(bg.r_z(z,ocb,h,**kw),rows[:,6]),
             "D_raw":(d,rows[:,7]),"f":(f,rows[:,8])}
    grad=gradient(x,policy,species)
    for i,name in enumerate(("m1","m2","m3","Neff")):
        outputs["gradient_"+name]=(np.array([grad[i]]),rows[:1,9+i])
    for name,(value,reference) in outputs.items():
        err=abs(np.asarray(value)-reference)
        # Relative gradients near zero are ill-conditioned; keep their absolute
        # error in the report rather than pretending they have a useful ratio.
        floor=1e-6 if name.startswith("gradient") else 0.
        relative=np.divide(err,abs(reference),out=np.zeros_like(err),where=abs(reference)>floor)
        item=report.setdefault(name,dict(max_absolute=0.,max_relative=0.,relative_reference_floor=floor))
        if err.max()>item["max_absolute"]:
            item["max_absolute"]=float(err.max())
            item["absolute_case"]=dict(policy=policy,species=species,Neff=n,masses=[m1,m2,m3],index=int(err.argmax()))
        item["max_relative"]=max(item["max_relative"],float(relative.max()))
        if name.startswith("gradient"):
            np.testing.assert_allclose(value,reference,rtol=2e-5,atol=8e-8)
        else:
            np.testing.assert_allclose(value,reference,rtol=3e-9,atol=1e-9)
print(json.dumps(report,indent=2))
