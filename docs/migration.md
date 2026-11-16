# Migration notes

## 0.8.0: explicit masses in analytic cosmology

Analytic background and growth functions now require exactly three physical
neutrino masses in eV. This also applies to `w0waCDMCosmology.m_nu`.
Scalar inputs, including zero and traced scalars, raise `ValueError`; they are
not silently interpreted as a total mass or split into equal masses.

```python
# Choose the physical splitting explicitly; these are different models.
E = jaxace.E_z(1.0, 0.3, 0.67, mν=(0.02, 0.02, 0.02), Neff=3.044)
E_one_massive = jaxace.E_z(1.0, 0.3, 0.67, mν=(0.06, 0.0, 0.0), Neff=3.044)
```

Omitting the masses selects `(0.0, 0.0, 0.0)`: three massless species with
the documented Neff prescription. The retired scalar model represented only
one species, so even its zero-mass result is not the new default radiation
model. Existing explicit-three-mass calculations retain their conventions.

Neural emulator inputs are unchanged: they still accept the total mass, with
the splitting fixed by the training data. This change neither retrains nor
reinterprets existing artifacts. The old scalar quadrature helpers and their
unused `quadax` dependency have been removed.

## 0.7.0

### New reusable interpolation objects

The release adds `CubicSpline`, `CubicSplinePlan`, and `AkimaSplinePlan`.

Use a prepared spline when the ordinates and source knots are fixed but the
query grid changes:

```python
spline = jaxace.AkimaSpline(u, t)
values = spline(t_new)
```

Use a plan when both grids are fixed but the ordinates change:

```python
plan = jaxace.CubicSplinePlan(t, t_new)
values = plan(u)
```

The one-shot functions remain available for single-use interpolation.

### AkimaSpline constructor change

`AkimaSpline` is now a frozen registered JAX PyTree and supports the symmetric
public constructor:

```python
AkimaSpline(u, t)
```

The previous stored-coefficient form remains accepted:

```python
AkimaSpline(u, t, b, c, d)
```

However, the object is no longer a `NamedTuple`. Code relying on tuple-specific
operations such as indexing, unpacking, `len(spline)`, or `tuple(spline)` must
use the named fields instead:

```python
spline.u
spline.t
spline.b
spline.c
spline.d
```

This is an intentional public API change in the 0.7.0 feature release.
