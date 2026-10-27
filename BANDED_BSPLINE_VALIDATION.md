# Banded B-spline validation (working-tree implementation)

Baseline jaxace: `368596b`. The design and frozen reference provenance are in
`BANDED_BSPLINE_DESIGN.md` and `tests/data/bspline_banded/README.md`.

## Implemented scope

The coefficient and prepared-plan constructors now use Julia-equivalent local
basis rows, seven-band Doolittle LU, full-band forward/transpose scans and the
exactly guarded bordered specialization. Nonfinite specialized columns retry
the full-band solve from the original RHS. `custom_linear_solve` reuses stopped
factors while retaining implicit derivatives through the live bands. Public
signatures and PyTrees are unchanged; prepared-plan application remains a dense
matrix multiplication. Query-basis evaluation still uses the old dense recursion.

This is algorithmic, not bitwise or exception-policy equivalence. In particular,
JAX does not raise Julia's SingularException or enforce its absolute 10-epsilon
pivot cutoff inside the compiled factorization. Zero/nonfinite pivots propagate
nonfinite results as in JAX linear algebra; extreme ill-conditioning is not
certified by the well-conditioned references. Independent review accepted this
policy difference. Nonsingular, canonical B-spline collocation has the
Schoenberg–Whitney/total-nonnegativity structure motivating pivot-free LU;
that does not provide condition-independent forward or gradient accuracy.

## Numerical evidence

- Immutable Julia and pre-change dense-JAX text references cover N=4,5,6,8,81,
  matrix coefficients, factors, transpose solves, evaluations, value and site
  gradients. Julia gradients use ForwardDiff. Existing fixtures are unchanged.
- On N81, the precision report's maximum absolute differences from dense JAX
  are 3.33e-16 for evaluated values, 6.66e-16 for value gradients and 1.43e-14
  for site gradients. Pointwise relative errors and their locations are reported
  separately, including small near-cancellation gradients.
- Initial focused public tests: **35/35** on JAX 0.6.2. Coverage includes JIT, dynamic
  inputs, vmap, JVP/VJP, site Hessian-vector products, empty batches, Float32/64
  and mixed precision, factor guard/fallback, existing endpoint derivatives,
  N81×8192 coefficients/interpolation, and N4096 construction with one query.
- Local complete suite on JAX 0.6.2: **339 passed**. That includes one pre-existing
  untracked personal Chebyshev test; it is not part of this proposed patch.
- JAX 0.4.38 focused rerun: **35/35**. The declared minimum 0.4.30 and GPU
  execution have not been tested. Dependencies have not been changed.
- Review-driven focused rerun: **63/63** on JAX 0.6.2, including plan JVP/VJP
  in live sites and queries (N4/5/8/81, Q1/7, clamp/zero, inside/outside),
  vmapped geometry, preserved host/throw rejection, and runtime callback proof
  that the bordered branch actually executes (N6/7/8/81, Float32/64, both
  transposes). These tests also passed in the final older-JAX rerun below.
- Astra then identified a real integer-geometry overflow. Main reproduced the
  wrong constant-one interpolation on Int32/Int64 grids, repaired promotion
  before arithmetic, and passed **4/4 direct regressions** plus **67/67 focused
  tests**. Astra independently accepted that bounded repair. The final repaired
  JAX 0.6.2 full suite passed **371/371** in 368.99 s (including the one
  pre-existing untracked personal test mentioned above). Source/test hashes
  still match `integer_fix.sha256`. The final JAX 0.4.38 focused run passed
  **67/67** in 113.82 s. All final gate exit codes are zero in
  `integer_fix_gates.txt`.

### Float32 diagnostic, not a tolerance relaxation

The initial new pairwise Float32 site-gradient check failed: the two algorithms
rounded to opposite sides of a Float64 oracle. The oracle uses the *exactly
promoted* Float32 inputs, not an unrounded replacement grid. On JAX 0.6.2 each
path was within about 8.2e-6 of that oracle, while their difference reached
1.18e-5. On 0.4.38 the old dense Float32 path itself missed the newly specified
reference budget at one small gradient component (1.19e-5 absolute error).
The port is tested against the Float64 oracle with the original tolerance,
unchanged: `rtol=8e-5, atol=8e-6`. No old package tolerance or fixture was changed.
The full-suite run predates removal of the unnecessary assertion about the old
Float32 implementation; production code and port-accuracy assertions are unchanged.

### Independent high-precision stress checks

Two new physical-coordinate Float64 gradient comparisons initially failed. We
did not declare dense Float64 exact: additional **256-bit native Julia
ForwardDiff** references were generated on exactly promoted saved Float64
inputs. No pre-port fixture was regenerated. The quadratic coefficient-loss
gradient has the following maximum *physical-coordinate* absolute errors:

| Grid | Condition number | New banded vs high precision | Old dense vs high precision |
|---|---:|---:|---:|
| Raw-k N81 (exp of saved log-k grid) | 4.8123 | 2.46e-11 | 3.39e-11 |
| N8, first spacing 1e-8 | 1.4151e7 | 0.0357 | 0.358 |

The clustered large gradient is about 1.76e5. Its reported errors are not a
claim of uniform 1e-12 physical-gradient accuracy. Small raw-k gradients suffer
cancellation inside the collocation derivative contraction even when the
collocation matrix itself is well-conditioned. The new path is not uniformly
better at every component; these are measured cases, not universal claims.

The additional stress tests compare coefficients with the high-precision gold
and require a strict collocation residual. They compare geometry derivatives in
dimensionless local-gap displacement coordinates, `t_i + gap_i*delta_i`, with
the gaps fixed, using the explicit conditioning allowance
`max(2e-12, 64*eps(Float64)*cond(A))`. Absolute physical-gradient tolerances are
not transferable across raw-k decades or near-coincident sites. **This is a
loose stress regression gate**, not a tight accuracy certification: at the
clustered endpoint its allowance corresponds to roughly 20 physical-gradient
units, versus the observed 0.0357 new-path error. Ordinary-grid/reference
tolerances remain unchanged.

`*_uncancelled.txt` and `*_collocation_scale.txt` are saved high-precision
diagnostics, not files read by unit tests. The former only measures cancellation
in the loss-Jacobian sum and misses earlier cancellation inside `A' C`; the
latter measures that earlier collocation contraction. Both and their separate
generator/provenance records are preserved rather than replacing an earlier
diagnostic with a more convenient number. A traced duplicate-site test also
confirms nonfinite results without introducing a new runtime pivot policy.

## Performance and review gates

`benchmarks/test_banded_bspline_benchmarks.py` compares old dense and new banded
coefficients, complete value/site gradients, public interpolation and plan
construction at N81×8192. Geometry remains a dynamic argument. It uses paired
A/B/B/A trials, 30 synchronized samples each, compile excluded, and available
Linux temperature/throttling checks. Rejected attempts are preserved: first a
presence-only check blocked a passive watcher; a later activity check observed
that watcher consume CPU and rejected the incomplete matrix. A complete
pre-integer-repair matrix subsequently passed, but is not the final snapshot.

The repaired-source matrix passed **20/20** checks. CPU core 4, Float64,
JAX 0.6.2, one allowed CPU/BLAS thread; compile excluded; all observed activity
and throttling gates passed; maximum post-trial temperature was 55°C. Means of
the two independent trial medians (milliseconds):

| Operation | Dense | Banded | Dense/banded |
|---|---:|---:|---:|
| Coefficients, N81×8192 | 5.220 | 2.574 | 2.03× |
| Complete value gradient | 9.371 | 5.181 | 1.81× |
| Complete site gradient | 12.264 | 18.950 | **0.65×** |
| Public interpolation, Q129 | 9.211 | 6.800 | 1.35× |
| Plan construction, N81/Q129 | 0.260 | 0.290 | **0.90×** |

Largest same-implementation repeat-median drift was 4.04%; the other operations
were below 3%. Geometry remains dynamic. Plan construction has 129 transpose
RHS columns, not 8192; its metadata records that distinction.

**This is not a universal speedup or an unqualified performance acceptance.**
Site gradients are approximately 54.5% slower and plan preparation 11.5% slower
on this CPU workload. The site-gradient cost remains an unresolved optimization
issue, not a numerically incorrect derivative. Repeated prepared-plan application
is unchanged. No GPU or other-hardware speedup is inferred. The full raw paired
samples and source hashes are in `timings_integer_fix.json`.

Opus 5.5/High found no confirmed production defect and accepted the bounded
coverage/reference repairs, conditional on their tests and reporting the stress
numbers above. Astra independently found and accepted the integer-promotion
repair; its covered source hashes matched before and after review. The observed
geometry/preparation regressions are explicitly not waived or concealed.
No GPU certification, commit, push, version change or release is claimed.

Local logs (not shipped): `.benchmarks/banded_validation/`, including the failed
Float32 diagnostic attempts, precision report and full/compatibility suite logs.
