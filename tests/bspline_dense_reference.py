"""Frozen dense numerical oracle from jaxace 368596b; tests/benchmarks only."""
import numpy as np
import jax
import jax.numpy as jnp

_MAX_CUBIC_B_SPLINE_OPERATOR_BYTES = 64 * 1024**2
_MAX_CUBIC_B_SPLINE_WORKSPACE_BYTES = 512 * 1024**2


def _check_cubic_b_spline_workspace(rows, columns, dtype):
    # Conservative eight-array estimate for dense basis recursion/solve work,
    # separate from stored plan size. Not a bound on total process/AD memory.
    workspace_bytes = 8 * rows * columns * np.dtype(dtype).itemsize
    if workspace_bytes > _MAX_CUBIC_B_SPLINE_WORKSPACE_BYTES:
        raise ValueError("cubic B-spline dense construction/evaluation workspace "
                         "estimate exceeds 512 MiB; use smaller grids")

def _cubic_b_spline_knots(t):
    """Return the not-a-knot cubic B-spline knot vector derived from ``t``."""
    t = jnp.asarray(t)
    if t.ndim != 1 or len(t) < 4:
        raise ValueError("cubic B-spline interpolation requires at least four sites")
    if not isinstance(t, jax.core.Tracer):
        sites = np.asarray(t)
        if not (np.all(np.isfinite(sites)) and np.all(np.diff(sites) > 0)):
            raise ValueError("cubic B-spline sites must be finite and strictly increasing")
    _check_cubic_b_spline_workspace(len(t), len(t) + 6, jnp.result_type(t, 1.0))
    return jnp.concatenate((jnp.repeat(t[:1], 4), t[2:-2], jnp.repeat(t[-1:], 4)))


def _cubic_b_spline_basis_matrix(knots, query):
    """Evaluate every cubic B-spline basis function at ``query``."""
    knots = jnp.asarray(knots)
    query = jnp.atleast_1d(query)
    dtype = jnp.result_type(knots, query, 1.0)
    query = query.astype(dtype)
    knots = knots.astype(dtype)
    _check_cubic_b_spline_workspace(len(query), len(knots) - 1, dtype)

    # Close the last nonempty degree-zero interval, then differentiate the
    # ordinary polynomial recursion. A final one-hot override loses all query
    # derivatives at the right endpoint despite giving the correct value.
    # Open cubic knots have four equal right-end knots: interval len(knots)-5
    # is the last nonempty one. No knot-value searches are necessary here.
    basis = ((query[:, None] >= knots[:-1]) & (query[:, None] < knots[1:])).astype(dtype)
    last = len(knots)-5
    closed_last = (query >= knots[last]) & (query <= knots[last+1])
    basis = basis.at[:, last].set(closed_last.astype(dtype))

    for degree in range(1, 4):
        n_columns = len(knots) - degree - 1
        left_denominator = knots[degree:degree + n_columns] - knots[:n_columns]
        right_denominator = (
            knots[degree + 1:degree + 1 + n_columns]
            - knots[1:1 + n_columns]
        )

        safe_left = jnp.where(
            left_denominator != 0,
            left_denominator,
            jnp.ones_like(left_denominator),
        )
        safe_right = jnp.where(
            right_denominator != 0,
            right_denominator,
            jnp.ones_like(right_denominator),
        )
        left_factor = jnp.where(
            left_denominator != 0,
            (query[:, jnp.newaxis] - knots[:n_columns]) / safe_left,
            0,
        )
        right_factor = jnp.where(
            right_denominator != 0,
            (knots[degree + 1:degree + 1 + n_columns] - query[:, jnp.newaxis])
            / safe_right,
            0,
        )
        basis = left_factor * basis[:, :n_columns] + right_factor * basis[:, 1:n_columns + 1]

    return basis
