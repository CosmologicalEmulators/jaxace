# Julia-equivalent banded B-spline construction

Approved direction: 2026-10-23. JAX baseline: `368596b`; Julia reference:
ACE `132a6e5` plus the independently tested callable-specialization repair.
Reference metadata also records the actual numerical source hashes.

## Scope

Preserve not-a-knot interpolation, all public signatures, dtype promotion,
extrapolation and endpoint conventions, PyTree layouts, and derivatives of
values, sites and queries. No Julia-style global thread setter is needed: XLA
owns device execution. No fixed-geometry-only AD shortcut is permitted.

Construction will use Julia's four-entry local cubic basis recurrence,
seven-row band storage `B[3+i-j,j] = A[i,j]` (zero-based), pivot-free band LU,
and forward/transpose band substitution. Loops over sites are compiled scans
or fixed-bound loops, not Python-unrolled loops. Independent RHS columns are
batched in each operation. The bordered-tridiagonal specialization must use an
exact factor-content guard, with full-band fallback for small/general cases
and nonfinite specialized results. JAX floating-point lowering need not be
bitwise identical to Julia; reference comparisons are numerical.

Use `jax.lax.custom_linear_solve` with the live band matvec and explicit forward
and transpose solves sharing a factorization. Stop AD through the factorization,
not through the live matrix entries. This preserves both JVP and VJP geometry
derivatives via implicit differentiation. Do not use custom_vjp alone, since
that would discard existing forward-mode support.

Keep the existing dense evaluation matrix and the dense prepared-plan operator
execution for this change. Construct the latter with the transpose band solve
instead of dense LU. The plan still stores a Q×N operator, so its 64 MiB guard
remains. Dense basis evaluation retains its workspace guard. Remove only the
quadratic workspace guard attached to knot construction once that path no
longer allocates a dense collocation matrix. This is not a claim that all
evaluation paths have linear memory.

JAX cannot raise Julia's runtime SingularException inside jit. Singular or
nonfinite systems follow JAX's nonfinite-result convention; supported finite
well-conditioned grids must agree with Julia and the old dense JAX solver.
Any fallback/conditioning difference discovered during validation must be
documented rather than hidden by looser tolerances.

## Gates, in order

1. Save original Julia coefficients, factors, transpose solves, values and
   value/site gradients as text for N=4,5,6,8,81. Use the saved adaptive N81
   grid and deterministic multi-RHS data. Save pre-change dense JAX outputs
   and gradients separately. Generators refuse to overwrite fixtures.
2. Compare elementwise against those immutable fixtures. Cover Float32/64,
   mixed dtypes, vector/matrix RHS, extreme/cancellation cases, all boundary
   policies, public constructors/plans, dynamic jit inputs, vmap, JVP/VJP and
   derivatives of live sites and endpoint queries. Existing tests stay enabled.
3. Benchmark coefficient construction, prepared-plan construction, public
   evaluation and prepared gradients with pytest-benchmark, synchronized calls,
   compilation excluded, N81×8192 as the representative batch. Preserve an
   independent dense baseline. Do not infer GPU performance from CPU results.
4. Run the package suite and obtain independent review before claiming acceptance.

No version changes, commits, pushes or releases are implied by implementation.
