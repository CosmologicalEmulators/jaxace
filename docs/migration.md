# Migration notes

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
