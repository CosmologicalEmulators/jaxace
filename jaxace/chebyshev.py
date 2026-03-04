"""
Chebyshev polynomial interpolation and optimization tools for JAX.
"""
from typing import NamedTuple, Tuple, Union, Any, Optional, List, cast
import jax.numpy as jnp
import jax.scipy.fft as jsfft

class ChebyshevPlan(NamedTuple):
    """
    Plan for computing Chebyshev coefficients of a function evaluated at Chebyshev nodes.

    Attributes:
        K: Tuple of polynomial degrees (K+1 nodes per dimension)
        nodes: Tuple of evaluation nodes arrays
        dim: Tuple of target dimensions for decomposition
    """
    K: Tuple[int, ...]
    nodes: Tuple[jnp.ndarray, ...]
    dim: Tuple[int, ...]

def chebpoints(K: int, x_min: float, x_max: float) -> jnp.ndarray:
    """
    Generate Chebyshev roots mapped to [x_min, x_max].

    Matches AbstractCosmologicalEmulators.chebpoints
    """
    k = jnp.arange(K + 1)
    # Cosine points in [-1, 1], matching Julia convention (descending from 1 to -1)
    nodes_std = jnp.cos(jnp.pi * k / K)
    # Map to [x_min, x_max]
    nodes = x_min + 0.5 * (nodes_std + 1.0) * (x_max - x_min)
    return nodes

def prepare_chebyshev_plan(x_min: Union[float, Tuple[float, ...]],
                           x_max: Union[float, Tuple[float, ...]],
                           K: Union[int, Tuple[int, ...]],
                           size_nd: Optional[Tuple[int, ...]] = None,
                           dim: Union[int, Tuple[int, ...]] = 0) -> ChebyshevPlan:
    """
    Precomputes the Chebyshev nodes required to compute coefficients.

    K is the polynomial degree (K+1 nodes). For N-dimensional inputs, specify
    the target dimensions `dim`.

    Args:
        x_min: Minimum x value(s)
        x_max: Maximum x value(s)
        K: Polynomial degree(s)
        size_nd: Tuple representing input array shape (unused parameter kept for API parity)
        dim: Target dimension(s) for Chebyshev decomposition (default 0)

    Returns:
        ChebyshevPlan object containing nodes and settings
    """
    if isinstance(K, int):
        K_tup = (K,)
        x_min_tup = (float(cast(float, x_min)),)
        x_max_tup = (float(cast(float, x_max)),)
        dim_tup = (int(cast(int, dim)),)
    else:
        K_tup = tuple(K)
        x_min_tup = tuple(map(float, cast(Tuple[float, ...], x_min)))
        x_max_tup = tuple(map(float, cast(Tuple[float, ...], x_max)))
        dim_tup = tuple(map(int, cast(Tuple[int, ...], dim)))

    nodes = tuple(chebpoints(K_i, x_min_i, x_max_i)
                  for K_i, x_min_i, x_max_i in zip(K_tup, x_min_tup, x_max_tup))

    return ChebyshevPlan(K=K_tup, nodes=nodes, dim=dim_tup)

def chebyshev_polynomials(x_grid: jnp.ndarray, x_min: float, x_max: float, K: int) -> jnp.ndarray:
    """
    Computes the matrix of Chebyshev polynomials evaluated on `x_grid`,
    mapped to `[-1, 1]` from `[x_min, x_max]`.

    Matches AbstractCosmologicalEmulators.chebyshev_polynomials functionality.

    Args:
        x_grid: Grid of evaluation points
        x_min: Input minimum domain
        x_max: Input maximum domain
        K: Polynomial degree

    Returns:
        Matrix of size (len(x_grid), K+1)
    """
    # Map to [-1, 1] domain
    z = 2.0 * (x_grid - x_min) / (x_max - x_min) - 1.0

    T_mat = jnp.zeros((len(x_grid), K + 1), dtype=z.dtype)
    T_mat = T_mat.at[:, 0].set(1.0)

    if K > 0:
        T_mat = T_mat.at[:, 1].set(z)

    for k in range(2, K + 1):
        T_mat = T_mat.at[:, k].set(2.0 * z * T_mat[:, k-1] - T_mat[:, k-2])

    return T_mat

def _dct1(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Computes the unnormalized Discrete Cosine Transform of type 1 (DCT-I).
    This matches FFTW.REDFT00 on an N-dimensional array along a specific axis.

    For small K (typical in emulators), explicit matrix multiplication is
    fast and natively fully differentiable in JAX without complex FFT rules.
    """
    n = x.shape[axis]

    # DCT-1 is defined for N >= 2
    if n < 2:
        return x

    K = n - 1

    # Construct DCT-I matrix A_T
    # A_T[k, j] = cos(pi * j * k / K) but unnormalized, so first and last elements
    # logic differs. Formula for REDFT00 (unnormalized DCT-I) of length n is:
    # y_k = x_0 + (-1)^k x_{n-1} + 2 sum_{j=1}^{n-2} x_j cos(pi * j * k / (n-1))

    j = jnp.arange(n)
    k = jnp.arange(n)

    # Outer product equivalent to j * k
    jk = jnp.outer(k, j)

    # Base cosines matrix
    A = jnp.cos(jnp.pi * jk / K)

    # REDFT00 scales the inner elements by 2
    # The first (j=0) and last (j=n-1) elements remain unscaled
    A = A.at[:, 1:-1].multiply(2.0)

    # Flip signs if needed?
    # Julia cos(pi*k/K) is descending. If JAX points are also descending,
    # then no flip is needed compared to Julia's FFTW.
    # In Julia, nodes are x_min + 0.5 * (cos(pi*k/K) + 1.0) * (x_max - x_min).
    # k=0 -> cos(0)=1 -> node = x_max.
    # k=K -> cos(pi)=-1 -> node = x_min.
    # So both use the same node order now.

    # Move target axis to the last position
    x_moved = jnp.moveaxis(x, axis, -1)

    # Contract last axis of x_moved with last axis of A
    res_moved = jnp.dot(x_moved, A.T)

    # Move the axis back to its original position
    res = jnp.moveaxis(res_moved, -1, axis)

    return res

def chebyshev_decomposition(plan: ChebyshevPlan, f_vals: jnp.ndarray) -> jnp.ndarray:
    """
    Computes Chebyshev coefficients for a function evaluated at Chebyshev nodes.

    Fully supports batched N-dimensional JAX arrays across `plan.dim` using custom DCT-1.

    Args:
        plan: ChebyshevPlan containing interpolation configuration
        f_vals: Array of function outputs evaluated exactly on plan.nodes along plan.dim
                (must have size K+1 along plan.dim)

    Returns:
        N-Dimensional array of Chebyshev coefficients
    """
    c = f_vals

    for i, dim in enumerate(plan.dim):
        K_i = plan.K[i]

        # DCT-I corresponds to type=1
        c_raw = _dct1(c, axis=dim)

        # Scale coefficients
        c = c_raw / K_i

        # Divide first and last element of the specific dimension by 2
        idx_first: List[Any] = [slice(None)] * c.ndim
        idx_first[dim] = 0
        c = c.at[tuple(idx_first)].divide(2.0)

        idx_last: List[Any] = [slice(None)] * c.ndim
        idx_last[dim] = -1
        c = c.at[tuple(idx_last)].divide(2.0)

    return c
