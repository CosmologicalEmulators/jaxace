# Neutrino thermal prescriptions

The analytic background treats **three explicit physical neutrino masses** with
zero-chemical-potential Fermi–Dirac distributions, plus a massless radiation
remainder. Changing `Neff` is not enough to specify how that massive-neutrino
sector changes: jaxace provides two distinct thermal prescriptions.

This page describes the analytic model, **not** the frozen neural-emulator
training labels. Neural emulators still take total neutrino mass with the
splitting used in their training data. See [Cosmology API](api.md)
for the callable API and the separate growth approximations.

For example, compare expansion at the same physical parameters:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxace

z = jnp.array([0.0, 1.0, 10.0])
common = dict(ln10As=3.044, ns=0.965, h=0.67, omega_b=0.0224, omega_c=0.12,
              m_nu=(0.01, 0.02, 0.03), Neff=4.0)
temperature = jaxace.w0waCDMCosmology(
    **common, neutrino_prescription="temperature")
radiation = jaxace.w0waCDMCosmology(
    **common, neutrino_prescription="radiation")
E_temperature = temperature.E_z(z)
E_radiation = radiation.E_z(z)
assert jnp.allclose(E_temperature[0], 1.0)
assert jnp.allclose(E_radiation[0], 1.0)
print(jnp.column_stack((z, E_temperature, E_radiation)))
```

## Energy density and the meaning of Neff

Let $a=1/(1+z)$, with photon temperature
$T_\gamma(a)=T_{\gamma,0}/a$ and $T_{\gamma,0}=2.7255\,\mathrm{K}$.
All three species share the temperature ratio $r=T_\nu/T_\gamma$ within each
prescription; their masses may differ. Here $m_i$ denotes the rest energy in
eV ($c=1$), and $k_{\mathrm{B}}$ is expressed in eV/K. Define

$$
F(y)=\int_0^\infty \frac{q^2\sqrt{q^2+y^2}}{e^q+1}\,\mathrm{d}q,
\qquad
y_i(a)=\frac{m_i a}{k_{\mathrm{B}}rT_{\gamma,0}},
\qquad F(0)=\frac{7\pi^4}{120}.
$$

With unit degeneracy per species in CLASS's `deg_ncdm` convention, including
the usual neutrino/antineutrino pair, the total neutrino-sector density is

$$
\frac{\rho_\nu(a)}{\rho_\gamma(a)}
=\frac{15r^4}{\pi^4}\sum_{i=1}^{3} F\!\left(y_i(a)\right)
+\frac{7}{8}r_{\mathrm{inst}}^4 N_{\mathrm{ur}},
\qquad r_{\mathrm{inst}}=\left(\frac{4}{11}\right)^{1/3}.
$$

Here $N_{\mathrm{ur}}$ is the effective count of the **massless remainder**,
not the total `Neff`. It can be fractional. Throughout this page and the figure,
$\rho_\nu$ includes both the three Fermi–Dirac species and that remainder,
but **not photons**.

$N_{\mathrm{eff}}$ specifies the density in the relativistic limit:

$$
\left.\frac{\rho_\nu}{\rho_\gamma}\right|_{y_i\to0}
=\frac{7}{8}r_{\mathrm{inst}}^4 N_{\mathrm{eff}},
\qquad
N_{\mathrm{eff}}=3\left(\frac{r}{r_{\mathrm{inst}}}\right)^4+N_{\mathrm{ur}}.
$$

It is **not** a formula for the late-time energy density once neutrinos become
nonrelativistic. The mass-dependent integral must still be evaluated. jaxace
tabulates the integral at import and uses lookup interpolation, extended by a
moment series at large arguments, at runtime. It does not perform fresh momentum
quadrature at every background call.

## Common reference convention

Both prescriptions share the reference

$$
N_{\mathrm{eff,ref}}=3.044,\qquad r_{\mathrm{ref}}=0.71611,
\qquad
N_{\mathrm{ur,ref}}
=3.044-3\left(\frac{0.71611}{r_{\mathrm{inst}}}\right)^4
\simeq 0.00439512.
$$

The temperatures use the CLASS anchor, and the remainder makes the total
relativistic density correspond to exactly 3.044. This is an effective thermal
model, not an evolution of neutrino decoupling or a nonthermal distribution.
The photon temperature and individual physical masses are held fixed when
`Neff` changes.

## Temperature rescaling

The default `neutrino_prescription="temperature"` sets

$$
s=\frac{N_{\mathrm{eff}}}{N_{\mathrm{eff,ref}}},\qquad
r=r_{\mathrm{ref}}s^{1/4},\qquad
N_{\mathrm{ur}}=N_{\mathrm{ur,ref}}s.
$$

Both terms in the relativistic density therefore scale by $s$. The supported
input domain requires finite $N_{\mathrm{eff}}>0$. Increasing it heats the
three species; decreasing it cools them. In particular,

$$
n_i(a)\propto r^3a^{-3},\qquad
\frac{n_i}{n_{i,\mathrm{ref}}}=s^{3/4},\qquad
\rho_{\nu,i}\simeq m_i n_i\quad\text{when }y_i\gg1.
$$

Thus varying `Neff` also changes the late-time massive-neutrino density at fixed
masses, and shifts when each species becomes nonrelativistic. At
$N_{\mathrm{eff}}=4$, temperatures increase by about 7.07% and massive-species
number densities by about 22.73% relative to the reference.

## Fixed temperatures, variable massless radiation

With `neutrino_prescription="radiation"`, the three species keep their reference
temperatures and only the massless remainder changes:

$$
r=r_{\mathrm{ref}},\qquad
N_{\mathrm{ur}}=N_{\mathrm{eff}}
-3\left(\frac{r_{\mathrm{ref}}}{r_{\mathrm{inst}}}\right)^4.
$$

Their number densities and nonrelativistic transitions are unchanged. The
additional massless density redshifts as $a^{-4}$. Nonnegative massless density
requires

$$
N_{\mathrm{eff}}\ge
3\left(\frac{r_{\mathrm{ref}}}{r_{\mathrm{inst}}}\right)^4
\simeq 3.03960488.
$$

For example, `Neff=2` cannot represent this prescription. jaxace returns NaN
for invalid numerical domains, including under JIT, rather than silently switching
to temperature rescaling. Incorrect mass shapes and unknown static prescription
names raise `ValueError`.

Both prescriptions coincide at $N_{\mathrm{eff}}=3.044$. At other allowed
values they have the same relativistic-limit energy density but generally
different massive-neutrino backgrounds. If all three masses are exactly zero,
their total neutrino-sector density remains identical at every scale factor
for the same `Neff`.

## Expansion rate and closure

Neutrinos enter the background through their full energy density:

$$
E^2(a)=\frac{H^2(a)}{H_0^2}
=\Omega_{\mathrm{cb},0}a^{-3}+\Omega_{\gamma,0}a^{-4}
+\frac{\rho_\nu(a)}{\rho_{\mathrm{crit},0}}
+\Omega_{\mathrm{k},0}a^{-2}
+\Omega_{\mathrm{DE},0}f_{\mathrm{DE}}(a),
$$

$$
f_{\mathrm{DE}}(a)=a^{-3(1+w_0+w_a)}e^{3w_a(a-1)},\qquad
\Omega_{\mathrm{DE},0}=1-\Omega_{\mathrm{cb},0}-\Omega_{\gamma,0}
-\frac{\rho_\nu(1)}{\rho_{\mathrm{crit},0}}-\Omega_{\mathrm{k},0}.
$$

$\Omega_{\mathrm{cb},0}=\Omega_{\mathrm{c},0}+\Omega_{\mathrm{b},0}$ excludes
neutrinos. At fixed $H_0$, changing the neutrino sector changes the dark-energy
density required by closure. Consequently all curves have the same $H(0)=H_0$;
the comparison does not also hold $\Omega_{\mathrm{DE},0}$ fixed.

## jaxace compared with matched CLASS calculations

![jaxace neutrino density and expansion for two thermal prescriptions, with matched CLASS references](https://github.com/user-attachments/assets/1205530e-39ac-499a-a060-2afb6f946f4c)

[Vector SVG](https://github.com/user-attachments/assets/e256d558-4223-4d18-9e2e-f56a8c5ee9dd)
· [Image storage and caption](https://github.com/CosmologicalEmulators/jaxace/issues/58)

The solid curves are **jaxace only**; circles are CLASS v3.3.4. Blue denotes
temperature rescaling, orange denotes the fixed-temperature radiation
prescription. The upper panels show percentage changes relative to a common
**CLASS** reference at $N_{\mathrm{eff}}=3.044$, with the same masses,
$H_0$, $\omega_{\mathrm{b}}$ and $\omega_{\mathrm{c}}$. The lower panels
show jaxace/CLASS fractional residuals, multiplied by the factors printed on the
axes. Those tiny residuals are numerical differences, not the physical
difference between the two prescriptions.

For this example, $m_\nu=(0.01,0.02,0.03)\,\mathrm{eV}$,
$N_{\mathrm{eff}}=4$, $h=0.67$, $\omega_{\mathrm{b}}=0.0224$,
$\omega_{\mathrm{c}}=0.12$, $\Omega_{\mathrm{k},0}=0$ and $w=-1$.
The illustrative mass split is not an imposed oscillation hierarchy. The
maximum absolute fractional residuals on the plotted grid are below
$3\times10^{-8}$ for density and $2\times10^{-9}$ for expansion.
These are **this configuration's** diagnostics, not a bound over all parameters
or a validation of scale-dependent perturbation growth.

### Matching CLASS, rather than comparing unmatched defaults

For each prescription CLASS is configured with the same `m_ncdm`, `T_ncdm`,
`deg_ncdm=1` per massive species and `N_ur` specified above. Its `T_ncdm` input
is the **ratio to the photon temperature**, not a temperature in kelvin.

| CLASS input at $N_{\mathrm{eff}}=4$ | Temperature rescaling | Fixed-temperature radiation |
|:--|:--|:--|
| `N_ncdm` | 3 | 3 |
| `m_ncdm` [eV] | `0.01,0.02,0.03` | `0.01,0.02,0.03` |
| `T_ncdm` (each species) | `0.766714319146159` | `0.71611` |
| `N_ur` | `0.00577545013072` | `0.96039511754948` |

Both calculations use `T_cmb=2.7255`, `YHe=0.245`,
`tol_ncdm_bg=1e-12`, `tol_background_integration=1e-12` and
`background_integration_stepsize=0.01`. Only CLASS's background module is needed.
The 321 plotted samples per prescription are selected from its native background
rows over $0\le z\lesssim10^5$. Density residuals compare directly to those
rows; only the common reference in the upper panels is interpolated in
$\ln a$ and $\ln\rho_\nu$/$\ln H$.

When matching a case containing an exactly zero mass, transfer that species
to CLASS's `N_ur` with contribution $(r/r_{\mathrm{inst}})^4$ instead of
passing a zero `m_ncdm`; keep the same total relativistic density. The figure
uses three positive masses, so that conversion is not needed here.

The image records jaxace source `b4d4508`. Broader numerical references and their
CLASS generation settings are maintained under
`tests/data/neutrino_neff/`; `tests/test_neutrino_neff.py` exercises the saved CLASS
references. The source CLASS generator is maintained in the ACE repository at
`test/fixtures/neutrino_neff/generate_class_reference.py`.
The PNG/SVG are hosted in the repository's `images` issue, not stored in Git.
