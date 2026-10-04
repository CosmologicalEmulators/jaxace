"""Three-mass/Neff port: saved CLASS and Julia references, JIT and AD."""
from pathlib import Path
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import background as bg
from jaxace import _neutrinos as nu

DATA = Path(__file__).parent / "data/neutrino_neff"
H = .67
OCB = .1424/H**2


@pytest.mark.parametrize("policy", ["temperature", "radiation"])
@pytest.mark.parametrize("bad", ["neff", "nan_neff", "mass", "nan_mass"])
def test_masked_invalid_models_do_not_poison_shared_gradients(policy, bad):
    invalid_neff = jnp.nan if bad == "nan_neff" else (2. if policy == "radiation" else -1.)
    def models(x):
        masses, neff = x[6:9], x[9]
        invalid_mass = masses.at[0].set(jnp.nan if bad == "nan_mass" else -.01)
        ns = jnp.array([neff, invalid_neff if "neff" in bad else neff])
        ms = jnp.stack([masses, invalid_mass if "mass" in bad else masses])
        return ms, ns
    def observables(x, m, n):
        h, ocb, w0, wa, curvature, z = x[:6]
        kw = dict(mν=m, Neff=n, neutrino_prescription=policy, w0=w0, wa=wa, Ωk0=curvature)
        distances = [f(z, ocb, h, **kw) for f in
                     (bg.r̃_z, bg.d̃M_z, bg.d̃A_z, bg.r_z, bg.dM_z, bg.dA_z, bg.dL_z)]
        a = 1/(1+z)
        return jnp.array([bg.E_z(z, ocb, h, **kw), *distances,
                          bg.ρc_z(z, ocb, h, **kw)/1e11,
                          bg.dlogEdloga(a, ocb, h, **kw),
                          bg.Ωm_a(a, ocb, h, **kw), bg.Ωm_a_total(a, ocb, h, **kw)])
    def masked(x):
        ms, ns = models(x)
        values = jax.vmap(lambda m, n: observables(x, m, n))(ms, ns)
        return jnp.where(jnp.array([True, False])[:, None], values, 0.).sum(axis=0)
    x = jnp.array([.67, .31, -1., .05, .01, 1., .01, .02, .03, 3.5])
    ms, ns = models(x)
    assert np.all(np.isnan(observables(x, ms[1], ns[1])))
    expected = jax.jacrev(lambda v: observables(v, v[6:9], v[9]))(x)
    for derivative in (jax.jacrev(masked), jax.jit(jax.jacrev(masked))):
        actual = derivative(x)
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)


def _cross_loss(x, policy, species):
    return bg.D_z(jnp.array([0.,.5,1.,3.,5.]),OCB,H,mν=x[:3],Neff=x[3],
                  neutrino_prescription=policy,species=species,reltol=1e-10,abstol=1e-12).sum()


@pytest.mark.parametrize("policy", ["temperature", "radiation"])
@pytest.mark.parametrize("species", ["cb", "m"])
def test_masked_growth_models_preserve_shared_parameter_gradients(policy, species):
    def single(x, mass, neff):
        kw = dict(mν=mass,Neff=neff,neutrino_prescription=policy,species=species)
        d,f = bg.D_f_z(.7,x[0],x[1],**kw)
        return jnp.array([bg.D_z(.7,x[0],x[1],**kw),bg.f_z(.7,x[0],x[1],**kw),d,f])
    def masked(x, bad_mass, bad_neff):
        masses = jnp.stack([x[2:5],bad_mass])
        ns = jnp.array([x[5],bad_neff])
        values = jax.vmap(lambda m,n:single(x,m,n))(masses,ns)
        return jnp.where(jnp.array([True,False])[:,None],values,0.).sum(axis=0)
    derivative = jax.jacrev(masked,argnums=0)
    compiled = jax.jit(derivative)
    x = jnp.array([.31,.67,.01,.02,.03,3.5])
    expected = jax.jacrev(lambda v:single(v,v[2:5],v[5]))(x)
    invalids = [(x[2:5],-1. if policy=="temperature" else 2.),
                (x[2:5],jnp.nan), (jnp.array([-.01,.02,.03]),3.5),
                (jnp.array([jnp.nan,.02,.03]),3.5)]
    for mass,neff in invalids:
        assert np.all(np.isnan(single(x,mass,neff)))
        for fn in (derivative,compiled):
            actual = fn(x,mass,neff)
            assert np.all(np.isfinite(actual))
            np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)


def test_invalid_thermal_model_is_not_overridden_by_infinite_redshift():
    assert np.isinf(bg.E_a(0., OCB, H, mν=jnp.zeros(3)))
    assert np.isnan(bg.E_a(0., OCB, H, mν=jnp.zeros(3), Neff=-1.))
    assert np.all(np.isnan(bg.E_a(jnp.array([0., .5]), OCB, H,
                                 mν=jnp.array([-.01, .02, .03]))))


_cross_gradient = jax.jit(jax.grad(_cross_loss),static_argnames=("policy","species"))


@pytest.mark.parametrize("policy", ["temperature","radiation"])
@pytest.mark.parametrize("species", ["cb","m"])
def test_julia_growth_and_gradient_reference(policy,species):
    groups={}
    for line in (DATA/"julia_growth_reference.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        p,s,*values=line.split()
        if (p,s)!=(policy,species):
            continue
        row=np.array(values,dtype=float)
        groups.setdefault(tuple(row[:4]),[]).append(row)
    for (n,m1,m2,m3),rows in groups.items():
        rows=np.array(rows); z=jnp.asarray(rows[:,4]); x=jnp.array([m1,m2,m3,n])
        kw=dict(mν=x[:3],Neff=n,neutrino_prescription=policy)
        d,f=bg.D_f_z(z,OCB,H,**kw,species=species,reltol=1e-10,abstol=1e-12)
        np.testing.assert_allclose(d,rows[:,7],rtol=3e-9)
        np.testing.assert_allclose(f,rows[:,8],rtol=3e-9)
        np.testing.assert_allclose(_cross_gradient(x,policy,species),rows[0,9:],rtol=2e-5,atol=8e-8)


@pytest.mark.parametrize("policy", ["temperature", "radiation"])
def test_saved_class_references(policy):
    cache = {}
    for line in (DATA/"class_neff_reference.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        name, preset, *numbers = line.split()
        if preset != policy:
            continue
        neff,m1,m2,m3,w0,wa,z,hubble,chi,dref,fref = map(float,numbers)
        kw = dict(mν=jnp.array([m1,m2,m3]),Neff=neff,neutrino_prescription=policy,w0=w0,wa=wa)
        np.testing.assert_allclose(bg.E_z(z,OCB,H,**kw)*100*H,hubble,rtol=2e-7)
        if z<=5:
            np.testing.assert_allclose(bg.r_z(z,OCB,H,**kw),chi,rtol=2e-8,atol=1e-9)
            key=(name,neff)
            if key not in cache:
                cache[key]=bg.D_f_z(jnp.array([0.,.5,1.,3.,5.]),OCB,H,**kw)
            d,f=cache[key]; i=[0.,.5,1.,3.,5.].index(z)
            np.testing.assert_allclose(d[i]/d[0],dref,rtol=5e-5)
            np.testing.assert_allclose(f[i],fref,rtol=1e-4)


def test_saved_julia_default_reference():
    for row in np.loadtxt(DATA/"pre_neff_baseline.txt"):
        m1,m2,m3,z,e,r,d,f=row
        kw=dict(mν=jnp.array([m1,m2,m3]))
        np.testing.assert_allclose(bg.E_z(z,OCB,H,**kw),e,rtol=2e-7)
        np.testing.assert_allclose(bg.r_z(z,OCB,H,**kw),r,rtol=2e-8,atol=1e-9)
        ds,fs=bg.D_f_z(jnp.array([0.,z]),OCB,H,**kw)
        np.testing.assert_allclose(ds[1]/ds[0],d,rtol=5e-5)
        np.testing.assert_allclose(fs[1],f,rtol=1e-4)


@pytest.mark.parametrize("neff", [2.,3.044,5.])
def test_massless_density_and_closure(neff):
    m=jnp.zeros(3); a=jnp.array([.001,.01,.2,1.])
    omega=nu.OMEGA_GAMMA_H2/H**2
    ratio=bg.ΩνE2(a,omega,m,neff)/(omega/a**4)
    np.testing.assert_allclose(ratio,neff*7/8*(4/11)**(4/3),rtol=1e-12)
    np.testing.assert_allclose(bg.E_a(1.,OCB,H,mν=jnp.array([.01,.02,.03]),Neff=neff),1.,atol=1e-14)
    np.testing.assert_allclose(bg.Ωm_a_total(a,OCB,H,mν=m,Neff=neff),
                               bg.Ωm_a(a,OCB,H,mν=m,Neff=neff),rtol=1e-14)


def test_species_axis_permutations_and_vmap():
    a=jnp.array([.05,.2,.4,.8,1.])
    m=jnp.array([0.,.0086,.0502])
    ref=bg.E_a(a,OCB,H,mν=m,Neff=5.)
    for perm in itertools.permutations([0,1,2]):
        np.testing.assert_allclose(bg.E_a(a,OCB,H,mν=m[jnp.array(perm)],Neff=5.),ref,rtol=1e-14)
    np.testing.assert_allclose(jax.vmap(lambda ai: bg.E_a(ai,OCB,H,mν=m,Neff=5.))(a),ref,rtol=1e-14)
    jac=jax.jacrev(lambda mass: bg.E_a(a,OCB,H,mν=mass,Neff=5.))(m)
    assert jac.shape==(5,3) and np.isfinite(jac).all()
    np.testing.assert_allclose(jac[:,0],0.,atol=1e-14)


@pytest.mark.parametrize("policy", ["temperature","radiation"])
def test_all_object_wrappers(policy):
    cosmo=bg.w0waCDMCosmology(3.,.965,H,.0224,.12,m_nu=(.01,.02,.03),Neff=5.,neutrino_prescription=policy)
    z=jnp.array([1.5,.2,3.,.7])
    kw=dict(mν=cosmo.m_nu,Neff=5.,neutrino_prescription=policy)
    for name in ("E_z","r_z","dM_z","dA_z","dL_z","ρc_z","r̃_z","d̃M_z","d̃A_z"):
        np.testing.assert_allclose(getattr(cosmo,name)(z),getattr(bg,name)(z,OCB,H,**kw),rtol=1e-12)
    for species in ("cb","m"):
        for name in ("D_z","f_z","D_f_z"):
            np.testing.assert_allclose(getattr(cosmo,name)(z,species=species),
                getattr(bg,name)(z,OCB,H,**kw,species=species),rtol=1e-12)


@pytest.mark.parametrize("policy", ["temperature","radiation"])
@pytest.mark.parametrize("species", ["cb","m"])
def test_live_neff_mass_gradients(policy,species):
    z=jnp.array([.2,1.,3.]); x=jnp.array([.01,.02,.03,3.5])
    def loss(x):
        return bg.D_z(z,OCB,H,mν=x[:3],Neff=x[3],neutrino_prescription=policy,
                      species=species,reltol=1e-10,abstol=1e-12).sum()
    gradient=jax.jit(jax.grad(loss))(x)
    assert np.isfinite(gradient).all()
    # Diffrax's default checkpointed reverse adjoint does not support jvp.
    # Compare against finite differences of the complete public solve instead.
    for i in range(4):
        step=jnp.zeros(4).at[i].set(1e-4)
        finite=(loss(x+step)-loss(x-step))/2e-4
        np.testing.assert_allclose(gradient[i],finite,rtol=2e-4,atol=2e-8)
    background=lambda v: bg.E_z(z,OCB,H,mν=v[:3],Neff=v[3],neutrino_prescription=policy).sum()
    np.testing.assert_allclose(jax.jacfwd(background)(x),jax.grad(background)(x),rtol=1e-10,atol=1e-12)


def test_invalid_domains_are_not_clamped_into_valid_models():
    for n in (0.,-1.,np.nan,np.inf):
        assert np.isnan(bg.E_z(1.,OCB,H,mν=jnp.zeros(3),Neff=n))
    assert np.isnan(bg.E_z(1.,OCB,H,mν=.06,Neff=5.))
    assert np.isnan(bg.r_z(0.,OCB,H,mν=jnp.zeros(3),Neff=-1.))
    assert np.isnan(bg.E_z(1.,OCB,H,mν=jnp.zeros(3),Neff=2.,neutrino_prescription="radiation"))
    for m in ([0.,0.,-.1],[0.,0.,np.nan]):
        assert np.isnan(bg.E_z(1.,OCB,H,mν=jnp.array(m)))
    assert np.isnan(bg.D_z(1.,OCB,H,mν=jnp.zeros(3),Neff=2.,neutrino_prescription="radiation"))
    with pytest.raises(ValueError,match="exactly three"):
        bg.E_z(1.,OCB,H,mν=jnp.zeros(2))
    with pytest.raises(ValueError,match="neutrino_prescription"):
        bg.E_z(1.,OCB,H,mν=jnp.zeros(3),neutrino_prescription="typo")
