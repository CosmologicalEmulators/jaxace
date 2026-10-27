"""Local cubic collocation and band solves matching ACE's Julia algorithms.

Bands use zero-based B[3+i-j, j] = A[i, j]. Only the site dimension is
sequential; all independent RHS columns are carried together by XLA scans.
"""
import jax
import jax.numpy as jnp


def local_basis(knots, query):
    """Four active indices/weights, with the interior polynomial at endpoints."""
    # Match the dense basis: promote before geometry arithmetic, not after
    # integer differences or sums have already overflowed.
    dtype = jnp.result_type(knots, query, 1.0)
    knots, query = jnp.asarray(knots, dtype=dtype), jnp.asarray(query, dtype=dtype)
    n = knots.shape[0] - 4
    span = jnp.clip(jnp.searchsorted(knots, query, side="right") - 1, 3, n - 1)
    left1 = query - knots[span]
    right1 = knots[span + 1] - query
    temp = 1 / (right1 + left1)
    n10, n11 = right1 * temp, left1 * temp
    left2 = query - knots[span - 1]
    right2 = knots[span + 2] - query
    temp = n10 / (right1 + left2)
    n20, saved = right1 * temp, left2 * temp
    temp = n11 / (right2 + left1)
    n21, n22 = saved + right2 * temp, left1 * temp
    left3 = query - knots[span - 2]
    right3 = knots[span + 3] - query
    temp = n20 / (right1 + left3)
    n30, saved = right1 * temp, left3 * temp
    temp = n21 / (right2 + left2)
    n31, saved = saved + right2 * temp, left2 * temp
    temp = n22 / (right3 + left1)
    n32, n33 = saved + right3 * temp, left1 * temp
    return span[:, None] - 3 + jnp.arange(4), jnp.stack((n30, n31, n32, n33), axis=1)


def collocation_bands(knots, sites):
    columns, values = local_basis(knots, sites)
    rows = jnp.arange(sites.shape[0])[:, None]
    return jnp.zeros((7, sites.shape[0]), dtype=values.dtype).at[
        3 + rows - columns, columns
    ].set(values)


def factor_bands(bands):
    """Pivot-free Doolittle LU with the same update order as Julia's band LU."""
    n = bands.shape[1]

    def eliminate(k, a):
        # The bandwidth, not N, is unrolled. Truncated boundary updates are
        # genuinely skipped, not masked after dividing by an inactive pivot.
        for di in range(1, 4):
            i = k + di

            def update_row(a):
                a = a.at[3 + di, k].set(a[3 + di, k] / a[3, k])
                for dj in range(1, 4):
                    j = k + dj
                    a = jax.lax.cond(
                        j < n,
                        lambda a: a.at[3 + di - dj, j].add(-a[3 + di, k] * a[3 - dj, j]),
                        lambda a: a,
                        a,
                    )
                return a

            a = jax.lax.cond(i < n, update_row, lambda a: a, a)
        return a

    return jax.lax.fori_loop(0, n, eliminate, bands)


def band_matvec(bands, x, *, transpose=False):
    """Apply the live collocation matrix without materializing its dense form."""
    n = bands.shape[1]
    result = jnp.zeros_like(x)
    for offset in range(-3, 4):
        start, end = max(0, -offset), min(n, n - offset)
        values = bands[3 + offset, start:end]
        if x.ndim == 2:
            values = values[:, None]
        if transpose:
            result = result.at[start:end].add(values * x[start + offset:end + offset])
        else:
            result = result.at[start + offset:end + offset].add(values * x[start:end])
    return result


def _substitute(rhs, coefficients, diagonal=None, *, reverse=False, order=(0, 1, 2)):
    # History is furthest-to-nearest; reverse scans retain the next three rows.
    def step(history, data):
        value, weights, pivot = data
        for slot in order:
            value = value - weights[slot] * history[slot]
        if diagonal is not None:
            value = value / pivot
        return jnp.stack((history[1], history[2], value)), value

    pivots = jnp.ones(rhs.shape[0], dtype=rhs.dtype) if diagonal is None else diagonal
    _, result = jax.lax.scan(step, jnp.zeros((3, rhs.shape[1]), dtype=rhs.dtype),
                             (rhs, coefficients, pivots), reverse=reverse)
    return result


def full_solve(factor, rhs, *, transpose=False):
    """Full bandwidth-three substitution; RHS is always an N×M matrix here."""
    if transpose:
        value = _substitute(rhs, factor[:3].T, factor[3])
        return _substitute(value, factor[4:][::-1].T, reverse=True, order=(2, 1, 0))
    lower = jnp.stack((jnp.pad(factor[6, :-3], (3, 0)),
                       jnp.pad(factor[5, :-2], (2, 0)),
                       jnp.pad(factor[4, :-1], (1, 0))), axis=1)
    upper = jnp.stack((jnp.pad(factor[0, 3:], (0, 3)),
                       jnp.pad(factor[1, 2:], (0, 2)),
                       jnp.pad(factor[2, 1:], (0, 1))), axis=1)
    value = _substitute(rhs, lower)
    return _substitute(value, upper, factor[3], reverse=True)


def is_bordered(factor):
    """Exact factor-content guard, not a guess based on the spline or its size."""
    n = factor.shape[1]
    if n < 6:
        return jnp.array(False)
    columns = jnp.arange(n)
    return (
        jnp.all(factor[0] == 0) & jnp.all(factor[6] == 0)
        & jnp.all(jnp.where(columns == 3, jnp.isfinite(factor[1]), factor[1] == 0))
        & jnp.all(jnp.where(columns == n - 4, jnp.isfinite(factor[5]), factor[5] == 0))
        & jnp.all(jnp.isfinite(factor[3]) & (factor[3] != 0))
        & jnp.all(jnp.isfinite(factor[2, 1:]))
        & jnp.all(jnp.isfinite(factor[4, :-1]))
    )


def _border_substitute(rhs, neighbor, diagonal, extra, extra_row, *, reverse, extra_first):
    def step(history, data):
        i, value, weight, pivot = data
        if extra_first:
            value = jax.lax.cond(i == extra_row, lambda v: v - extra * history[0], lambda v: v, value)
        value = value - weight * history[1]
        if not extra_first:
            value = jax.lax.cond(i == extra_row, lambda v: v - extra * history[0], lambda v: v, value)
        if diagonal is not None:
            value = value / pivot
        return jnp.stack((history[1], value)), value

    pivots = jnp.ones(rhs.shape[0], dtype=rhs.dtype) if diagonal is None else diagonal
    _, result = jax.lax.scan(step, jnp.zeros((2, rhs.shape[1]), dtype=rhs.dtype),
                             (jnp.arange(rhs.shape[0]), rhs, neighbor, pivots), reverse=reverse)
    return result


def bordered_solve(factor, rhs, *, transpose=False):
    n = factor.shape[1]
    alpha, beta = factor[5, n - 4], factor[1, 3]
    if transpose:
        value = _border_substitute(rhs, factor[2], factor[3], beta, 3,
                                   reverse=False, extra_first=True)
        return _border_substitute(value, factor[4], None, alpha, n - 4,
                                  reverse=True, extra_first=False)
    lower = jnp.pad(factor[4, :-1], (1, 0))
    upper = jnp.pad(factor[2, 1:], (0, 1))
    value = _border_substitute(rhs, lower, None, alpha, n - 2,
                               reverse=False, extra_first=True)
    return _border_substitute(value, upper, factor[3], beta, 1,
                              reverse=True, extra_first=True)


def factor_solve(factor, rhs, *, transpose=False):
    vector = rhs.ndim == 1
    matrix = rhs[:, None] if vector else rhs

    def general(_):
        return full_solve(factor, matrix, transpose=transpose)

    def specialized(_):
        fast = bordered_solve(factor, matrix, transpose=transpose)
        finite_columns = jnp.all(jnp.isfinite(fast), axis=0)
        # As in Julia, retry nonfinite columns from the ORIGINAL RHS with the
        # full-band algorithm. The rare fallback computes a full batch in XLA.
        return jax.lax.cond(
            jnp.all(finite_columns), lambda _: fast,
            lambda _: jnp.where(finite_columns[None, :], fast, general(None)), None,
        )

    if factor.shape[1] >= 6 and matrix.shape[1] >= 2 and factor.dtype in (jnp.float32, jnp.float64):
        result = jax.lax.cond(is_bordered(factor), specialized, general, None)
    else:
        result = general(None)
    return result[:, 0] if vector else result


def solve_banded(bands, rhs, *, transpose=False):
    """Implicit JVP/VJP through live bands; reuse a single stopped LU factor."""
    dtype = jnp.result_type(bands, rhs, 1.0)
    bands, rhs = jnp.asarray(bands, dtype=dtype), jnp.asarray(rhs, dtype=dtype)
    if rhs.ndim not in (1, 2) or rhs.shape[0] != bands.shape[1]:
        raise ValueError("band solve requires an N-vector or N×M right-hand side")
    factor = factor_bands(jax.lax.stop_gradient(bands))
    return solve_factored(bands, factor, rhs, transpose=transpose)


def solve_factored(bands, factor, rhs, *, transpose=False):
    """Implicit solve with existing factors; geometry stays live in the matvec."""
    dtype = jnp.result_type(bands, rhs, 1.0)
    bands, rhs = jnp.asarray(bands, dtype=dtype), jnp.asarray(rhs, dtype=dtype)
    if rhs.ndim not in (1, 2) or rhs.shape[0] != bands.shape[1]:
        raise ValueError("band solve requires an N-vector or N×M right-hand side")
    factor = jax.lax.stop_gradient(factor)
    return jax.lax.custom_linear_solve(
        lambda x: band_matvec(bands, x, transpose=transpose), rhs,
        solve=lambda _, b: factor_solve(factor, b, transpose=transpose),
        transpose_solve=lambda _, b: factor_solve(factor, b, transpose=not transpose),
    )
