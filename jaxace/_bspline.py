"""Local cubic B-spline geometry, matching ACE's public basis conventions."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._bspline_banded import local_basis, collocation_bands, factor_bands, solve_factored, _geometry_arrays


def _validate_knots(knots):
    if knots.ndim != 1 or knots.shape[0] < 8:
        raise ValueError("a cubic basis requires a one-dimensional knot vector of length >= 8")
    if isinstance(knots, jax.core.Tracer):
        return
    t = np.asarray(knots)
    if not np.all(np.isfinite(t)) or np.any(t[1:] < t[:-1]):
        raise ValueError("knots must be finite and nondecreasing")
    if t[0] != t[3] or t[3] == t[4] or t[-4] != t[-1] or t[-5] == t[-4]:
        raise ValueError("endpoints must have exactly multiplicity four")
    if np.any(t[:-4] == t[4:]):
        raise ValueError("degenerate four-interval support")
    if np.any(t[4:-7] == t[7:-4]):
        raise ValueError("interior knots must have multiplicity at most three")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class CubicBSplineBasis:
    """Degree-three basis with explicit knots; indices are zero-based.

    Supply either a full knot vector (positional or ``knot_vector=``), or
    ``domain=(lo, hi)`` and optional ``internal_knots``. Under transformations,
    finite ordered knots with the documented multiplicities are preconditions.
    """

    _knots: jnp.ndarray

    def __init__(self, knots=None, *, domain=None, internal_knots=None, knot_vector=None):
        if knots is not None and knot_vector is not None:
            raise ValueError("provide the full knot vector only once")
        full = knot_vector if knots is None else knots
        if full is not None:
            if domain is not None or internal_knots is not None:
                raise ValueError("cannot combine a full knot vector with domain/internal_knots")
        else:
            if domain is None:
                raise ValueError("provide a full knot vector or a domain")
            endpoints = jnp.asarray(domain)
            if endpoints.shape != (2,):
                raise ValueError("domain must contain two endpoints")
            interior = (jnp.empty((0,), dtype=endpoints.dtype) if internal_knots is None
                        else jnp.asarray(internal_knots))
            if interior.ndim != 1:
                raise ValueError("internal knots must be one-dimensional")
            full = jnp.concatenate((jnp.repeat(endpoints[:1], 4), interior,
                                    jnp.repeat(endpoints[1:], 4)))
        full = jnp.array(full, copy=True)
        _validate_knots(full)
        object.__setattr__(self, "_knots", full)

    @property
    def knot_vector(self):
        return self._knots

    def tree_flatten(self):
        return (self._knots,), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        result = object.__new__(cls)
        object.__setattr__(result, "_knots", children[0])
        return result


def knot_vector(basis):
    """Return the complete stored knot vector, including repeated endpoints."""
    return basis._knots


def internal_knots(basis):
    """Return interior knots with their multiplicities."""
    return knot_vector(basis)[4:-4]


def nbasis(basis):
    """Number of cubic basis functions: knot count minus four."""
    return knot_vector(basis).shape[0] - 4


def bspline_domain(basis):
    """Return the fundamental domain's two endpoints."""
    t = knot_vector(basis)
    return t[3], t[-4]


def basis_support(basis, index):
    """Support of basis function ``index`` (zero-based)."""
    i = jnp.asarray(index)
    if i.ndim != 0 or not jnp.issubdtype(i.dtype, jnp.integer):
        raise TypeError("basis index must be an integer scalar")
    if not isinstance(i, jax.core.Tracer) and not 0 <= int(i) < nbasis(basis):
        raise IndexError("basis index outside [0, nbasis)")
    t = knot_vector(basis)
    return t[i], t[i + 4]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class CubicBSplineRow:
    """Four zero-based indices and their local basis values; a JAX PyTree."""
    indices: jnp.ndarray
    values: jnp.ndarray

    def tree_flatten(self):
        return (self.indices, self.values), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class CubicBSplineStencil:
    """Compact Q×4 indices/weights and the original query vector."""
    indices: jnp.ndarray
    weights: jnp.ndarray
    query: jnp.ndarray

    @property
    def i1(self): return self.indices[:, 0]
    @property
    def i2(self): return self.indices[:, 1]
    @property
    def i3(self): return self.indices[:, 2]
    @property
    def i4(self): return self.indices[:, 3]
    @property
    def w1(self): return self.weights[:, 0]
    @property
    def w2(self): return self.weights[:, 1]
    @property
    def w3(self): return self.weights[:, 2]
    @property
    def w4(self): return self.weights[:, 3]

    def tree_flatten(self):
        return (self.indices, self.weights, self.query), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        return cls(*children)


def _queries(query):
    q = jnp.asarray(query)
    if q.ndim > 1:
        raise ValueError("queries must be scalar or one-dimensional")
    if not isinstance(q, jax.core.Tracer) and not np.all(np.isfinite(np.asarray(q))):
        raise ValueError("query coordinates must be finite")
    return jnp.atleast_1d(q)


def _validate_extrapolation(policy):
    if policy not in ("throw", "clamp", "zero"):
        raise ValueError("extrapolation must be 'clamp', 'throw', or 'zero'")


def _cubic_b_spline_query(t, query, policy):
    _validate_extrapolation(policy)
    q = jnp.asarray(query)
    _queries(q)  # Validate rank and eager finiteness without changing scalar rank.
    # Use the declared fundamental domain, including its raw-knot cotangents.
    lo, hi = t[3], t[-4]
    inside = (q >= lo) & (q <= hi)
    if policy == "throw":
        if isinstance(q, jax.core.Tracer) or isinstance(t, jax.core.Tracer):
            raise ValueError("dynamic extrapolation='throw' is unsupported under JAX transformations")
        if not np.all(np.asarray(inside)):
            raise ValueError("query point outside the cubic B-spline domain")
    # Strict comparisons preserve interior derivatives at endpoint equality.
    # Bound before recurrence evaluation: masking an overflow afterwards is too late.
    finite = jnp.isfinite(q)
    bounded = jnp.where(finite, jnp.where(q < lo, lo, jnp.where(q > hi, hi, q)), lo)
    return bounded, inside


def _basis_rows(basis, query, policy):
    q = _queries(query)
    t, q = _geometry_arrays(knot_vector(basis), q)
    bounded, inside = _cubic_b_spline_query(t, q, policy)
    indices, weights = local_basis(t, bounded)
    if policy == "zero":
        indices = jnp.where(inside[:, None], indices, jnp.arange(4))
        weights = jnp.where(inside[:, None], weights, 0)
    weights = jnp.where(jnp.isfinite(q)[:, None], weights, jnp.nan)
    return indices, weights


def basis_row(basis, query):
    """Four local indices/weights, zero outside support; scalar query only."""
    if jnp.asarray(query).ndim != 0:
        raise ValueError("basis_row requires a scalar query")
    indices, weights = _basis_rows(basis, query, "zero")
    return CubicBSplineRow(indices[0], weights[0])


def basis_stencil(basis, query, extrapolation="throw"):
    """Local four-entry query rows; default domain checking is eager-only."""
    q = jnp.array(_queries(query), copy=True)
    indices, weights = _basis_rows(basis, q, extrapolation)
    return CubicBSplineStencil(indices, weights, q)


def basis_matrix(basis, query, storage="dense"):
    """Explicit dense basis matrix; ordinary interpolation need not allocate it."""
    if storage != "dense":
        raise ValueError("only dense storage is supported for basis_matrix")
    indices, weights = _basis_rows(basis, query, "zero")
    count, n = weights.shape[0], nbasis(basis)
    if count * n * np.dtype(weights.dtype).itemsize > 512 * 1024**2:
        raise ValueError("dense basis matrix exceeds 512 MiB")
    return jnp.zeros((count, n), dtype=weights.dtype).at[
        jnp.arange(count)[:, None], indices
    ].set(weights)


def _sites(t):
    t = jnp.asarray(t)
    t = jnp.array(t, dtype=jnp.result_type(t, 1.0), copy=True)
    if t.ndim != 1 or t.shape[0] < 4:
        raise ValueError("cubic B-spline interpolation requires at least four sites")
    if not isinstance(t, jax.core.Tracer):
        x = np.asarray(t)
        if not np.all(np.isfinite(x)) or np.any(x[1:] <= x[:-1]):
            raise ValueError("cubic B-spline sites must be finite and strictly increasing")
    return t


def _canonical_basis(t):
    return CubicBSplineBasis(jnp.concatenate((jnp.repeat(t[:1], 4), t[2:-2],
                                             jnp.repeat(t[-1:], 4))))


def _factorize(bands):
    factor = factor_bands(jax.lax.stop_gradient(bands))
    if not isinstance(factor, jax.core.Tracer):
        # Match Julia's eager acceptance threshold, in the geometry dtype.
        pivots = np.asarray(factor[3])
        if np.any(np.abs(pivots) < 10 * np.finfo(pivots.dtype).eps):
            raise np.linalg.LinAlgError("numerically singular cubic B-spline collocation")
    return factor


def _values(u, n):
    u = jnp.asarray(u)
    if u.ndim not in (1, 2) or u.shape[0] != n:
        raise ValueError("values must be an N-vector or N×M matrix matching the sites")
    return u


def _evaluate_stencil(stencil, coefficients):
    shape = (stencil.indices.shape[0],) + coefficients.shape[1:]
    result = jnp.zeros(shape, dtype=jnp.result_type(stencil.weights, coefficients))
    # Fixed bandwidth only; no loop unrolled over sites or query count.
    for column in range(4):
        weights = stencil.weights[:, column]
        if coefficients.ndim == 2:
            weights = weights[:, None]
        result = result + weights * coefficients[stencil.indices[:, column]]
    return result


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class CubicBSpline:
    """Prepared not-a-knot spline with local O(Q) query geometry.

    Matrix values represent independent columns. Query derivatives use the
    interior polynomial at endpoint equality. Geometry is validated eagerly;
    dynamic ``throw`` is unsupported under JAX transformations.
    """

    u: jnp.ndarray
    t: jnp.ndarray
    basis: CubicBSplineBasis
    coefficients: jnp.ndarray
    extrapolation: str

    def __init__(self, u, t, extrapolation="clamp"):
        _validate_extrapolation(extrapolation)
        t = _sites(t)
        u = jnp.array(_values(u, len(t)), copy=True)
        basis = _canonical_basis(t)
        bands = collocation_bands(knot_vector(basis), t)
        coefficients = solve_factored(bands, _factorize(bands), u)
        for name, value in (("u", u), ("t", t), ("basis", basis),
                            ("coefficients", coefficients), ("extrapolation", extrapolation)):
            object.__setattr__(self, name, value)

    @property
    def knots(self):
        return knot_vector(self.basis)

    @property
    def sites(self):
        return self.t

    def __call__(self, query):
        scalar = jnp.asarray(query).ndim == 0
        stencil = basis_stencil(self.basis, query, self.extrapolation)
        result = _evaluate_stencil(stencil, self.coefficients)
        return result[0] if scalar else result

    def tree_flatten(self):
        return (self.u, self.t, self.basis, self.coefficients), self.extrapolation

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        result = object.__new__(cls)
        for name, value in zip(("u", "t", "basis", "coefficients"), children):
            object.__setattr__(result, name, value)
        object.__setattr__(result, "extrapolation", auxiliary)
        return result


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False, init=False)
class CubicBSplinePlan:
    """Compact fixed-grid plan: O(N+Q) geometry and reusable band factors.

    ``plan(u)`` uses a band solve and four local entries per query. The computed
    ``operator`` property explicitly materializes a dense Q×N operator (64 MiB
    limit); it is neither cached nor used by ordinary plan application.
    """

    t: jnp.ndarray
    basis: CubicBSplineBasis
    _bands: jnp.ndarray
    _factor: jnp.ndarray
    stencil: CubicBSplineStencil
    extrapolation: str

    def __init__(self, t, t_new, extrapolation="clamp"):
        _validate_extrapolation(extrapolation)
        t = _sites(t)
        query = jnp.asarray(t_new)
        if query.ndim != 1:
            raise ValueError("a plan requires a one-dimensional query grid")
        basis = _canonical_basis(t)
        stencil = basis_stencil(basis, query, extrapolation)
        bands = collocation_bands(knot_vector(basis), t)
        factor = _factorize(bands)
        for name, value in (("t", t), ("basis", basis), ("_bands", bands),
                            ("_factor", factor), ("stencil", stencil),
                            ("extrapolation", extrapolation)):
            object.__setattr__(self, name, value)

    @property
    def sites(self):
        return self.t

    @property
    def t_new(self):
        return self.stencil.query

    @property
    def operator(self):
        n, q = len(self.t), len(self.t_new)
        dtype = jnp.result_type(self._bands, self.stencil.weights)
        if n * q * np.dtype(dtype).itemsize > 64 * 1024**2:
            raise ValueError("dense CubicBSplinePlan operator exceeds 64 MiB")
        # The stencil already includes the selected outside-domain policy.
        evaluation = jnp.zeros((q, n), dtype=dtype).at[
            jnp.arange(q)[:, None], self.stencil.indices
        ].set(self.stencil.weights)
        return solve_factored(self._bands, self._factor, evaluation.T, transpose=True).T

    def __call__(self, u):
        return _evaluate_stencil(self.stencil, bspline_coefficients(self, u))

    def tree_flatten(self):
        return (self.t, self.basis, self._bands, self._factor, self.stencil), self.extrapolation

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        result = object.__new__(cls)
        for name, value in zip(("t", "basis", "_bands", "_factor", "stencil"), children):
            object.__setattr__(result, name, value)
        object.__setattr__(result, "extrapolation", auxiliary)
        return result


def bspline_basis(spline_or_plan):
    """Return the basis used by a prepared spline or plan."""
    return spline_or_plan.basis


def bspline_coefficients(spline_or_plan, u=None):
    """Get spline coefficients or solve with a plan's existing factorization."""
    if isinstance(spline_or_plan, CubicBSpline):
        if u is not None:
            raise TypeError("a prepared spline already has fixed values")
        return spline_or_plan.coefficients
    if not isinstance(spline_or_plan, CubicBSplinePlan) or u is None:
        raise TypeError("expected a spline, or a plan and values")
    plan = spline_or_plan
    return solve_factored(plan._bands, plan._factor, _values(u, len(plan.t)))


def cubic_b_spline_interpolation(u, t, t_new, extrapolation="clamp"):
    """Not-a-knot interpolation of vector or column-batched matrix values."""
    return CubicBSpline(u, t, extrapolation)(t_new)


def prepare_cubic_b_spline(u, t, extrapolation="clamp"):
    """Prepare coefficients for fixed values/sites and changing queries."""
    return CubicBSpline(u, t, extrapolation)


def evaluate_cubic_b_spline(spline, t_new):
    """Evaluate a prepared spline at scalar or vector query coordinates."""
    return spline(t_new)


def prepare_cubic_b_spline_plan(t, t_new, extrapolation="clamp"):
    """Prepare a compact fixed-grid plan for changing values."""
    return CubicBSplinePlan(t, t_new, extrapolation)
