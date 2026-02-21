"""
Chebyshev polynomial interpolation and optimization tools for JAX.
"""
from typing import NamedTuple, Tuple, Union, Any, Optional
import jax.numpy as jnp
import jax.scipy.fft as jsfft

class ChebyshevPlan(NamedTuple):
    """
    Plan for computing Chebyshev coefficients of a function evaluated at Chebyshev nodes.

    Attributes:
        K: Polynomial degree (K+1 nodes)
        nodes: Evaluation nodes in range [x_min, x_max]
        dim: Target dimension for decomposition
    """
    K: int
    nodes: jnp.ndarray
    dim: int

def chebpoints(K: int, x_min: float, x_max: float) -> jnp.ndarray:
    """
    Generate Chebyshev roots mapped to [x_min, x_max].

    Matches AbstractCosmologicalEmulators.chebpoints
    """
    k = jnp.arange(K + 1)
    # Cosine points in [-1, 1], reversed to be strictly increasing
    nodes_std = jnp.cos(jnp.pi * (K - k) / K)
    # Map to [x_min, x_max]
    nodes = x_min + 0.5 * (nodes_std + 1.0) * (x_max - x_min)
    return nodes

def prepare_chebyshev_plan(x_min: float, x_max: float, K: int,
                           size_nd: Optional[Tuple[int, ...]] = None,
                           dim: int = 0) -> ChebyshevPlan:
    """
    Precomputes the Chebyshev nodes required to compute coefficients.

    K is the polynomial degree (K+1 nodes). For N-dimensional inputs, specify
    the target dimension `dim`.

    Args:
        x_min: Minimum x value
        x_max: Maximum x value
        K: Polynomial degree (K+1 nodes will be generated)
        size_nd: Tuple representing input array shape (unused parameter kept for API parity)
        dim: Target dimension for Chebyshev decomposition (default 0)

    Returns:
        ChebyshevPlan object containing nodes and settings
    """
    nodes = chebpoints(K, x_min, x_max)
    return ChebyshevPlan(K=K, nodes=nodes, dim=dim)

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
    n = len(x_grid)
    # Map to [-1, 1] domain
    z = 2.0 * (x_grid - x_min) / (x_max - x_min) - 1.0

    T_mat = jnp.zeros((n, K + 1), dtype=z.dtype)
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
    # logic differs. Wait, the formula for REDFT00 (unnormalized DCT-I) of length n is:
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

    # Flip the signs of odd k coefficients to properly map strict domain evaluations.
    # Julia's `chebpoints` maps nodes as cos(pi * (K-k) / K) (increasing order),
    # instead of cos(pi * k / K), flipping the domain and requiring (-1)^k adjustment.
    sign_flip = (-1.0)**k
    A = A * sign_flip[:, jnp.newaxis]

    # Now we need to multiply A along `axis` of `x`
    # JAX tensordot allows contracting dimension `axis` of `x` with dimension 1 of `A`
    # However, to preserve the order of axes such that the contracted axis remains in
    # the same position, we use jnp.moveaxis or swapaxes to put the target axis last,
    # apply dot product, and swap back.

    # Move target axis to the last position
    x_moved = jnp.moveaxis(x, axis, -1)

    # x_moved shape: (..., n)
    # A shape: (n, n)
    # Contract last axis of x_moved with last axis of A (which is j)
    # Result shape: (..., n) where the last dimension corresponds to k
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
    dim = plan.dim

    # DCT-I corresponds to type=1
    c_raw = _dct1(f_vals, axis=dim)

    # Scale coefficients
    c = c_raw / plan.K

    # Divide first and last element of the specific dimension by 2
    # JAX doesn't have an exact mutable selectdim equivalence, but we can construct slice indices
    slices_first = [slice(None)] * f_vals.ndim
    slices_first[dim] = 0

    slices_last = [slice(None)] * f_vals.ndim
    slices_last[dim] = -1

    c = c.at[tuple(slices_first)].divide(2.0)
    c = c.at[tuple(slices_last)].divide(2.0)

    return c
