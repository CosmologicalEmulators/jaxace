# Full Julia B-spline parity: validation ledger

Status: implementation and correctness validation complete; **performance
acceptance remains open**. Marco explicitly authorized committing and pushing
this work after receiving that caveat. This does not certify performance or
authorize a merge/release. Julia source remains unchanged.

## Source and fixtures

- Published JAX baseline: `368596b916ae11265883cfa5f8ce3d65f70693cf`.
- Julia public-basis specification: `c5f9ca06d1cd00a6b224ea505bb36f612477004e`.
- Existing banded/dense candidate preserved in
  `.benchmarks/julia_parity_20261026/before/jaxace`.
- New exact-merge basis fixtures: `tests/data/bspline_api`; mixed-dtype fixtures:
  `tests/data/bspline_mixed`. Generators refuse overwrites and record source,
  generator and dependency provenance. Earlier fixtures remain unchanged.

## Completed gates

- Initial public basis gate: 17 tests passed.
- Compact integration: 30 new tests and 79 existing spline tests passed.
- Strengthened public/storage gate: 33 tests passed.
- Initial full package on JAX 0.6.2: **404 passed**, 522.72 seconds.
- Final **full package** on JAX 0.6.2: **417 passed**, 540.28 seconds.
- Initial spline compatibility on JAX 0.4.38: **100 passed**, 190.80 seconds.
- Final **full package** on JAX 0.4.38: **417 passed**, 684.07 seconds.
- Additional frozen mixed-precision comparisons: **4 passed**, 11.95 seconds.
- Strict MkDocs build passed, without publication.

The local full-suite counts include one pre-existing untracked test in
`tests/test_jax_julia_chebyshev.py`. That unrelated file is intentionally excluded
from this B-spline commit; the published test inventory therefore has one fewer
test than these local runs.

The elementwise precision report against the existing Julia N4/5/6/8/81
fixtures passed unchanged tolerances (`precision.log`). Maxima over those cases:

| Quantity | Max absolute error | Max relative error over nonzero references |
|---|---:|---:|
| Coefficients | 4.44e-16 | 6.51e-16 |
| Predictions | 2.22e-16 | 2.68e-16 |
| Site gradients | 1.04e-14 | 8.68e-12 |
| Value gradients | 8.88e-16 | 7.49e-15 |

Absolute and relative maxima need not occur at the same point. These figures
describe the stated Float64 fixtures, not arbitrary ill-conditioned geometry
or the mixed-precision cases.

Logs live in `.benchmarks/julia_parity_20261026`. The initial full-suite counts
precede the last review-driven additions. The final focused review gate now
passes **46 tests** in 125.77 seconds, including the added coverage below and a
fresh-process x64 opt-out. The ordinary package default enables x64; explicit
opt-out requires both `JAX_ENABLE_X64=0` and `JAXACE_ENABLE_X64=false`.

The direct `jax >=0.4.30` requirement is not the effective installation floor:
`pyproject.toml` requires `diffrax ^0.7.0`, and Diffrax 0.7.0 requires JAX >=0.4.38.
The older-stack test therefore exercises the effective floor, not JAX 0.4.30.
No dependency constraints have been changed. The final 417-test inventory passed
on both JAX 0.4.38 and 0.6.2.

## Explicit endpoint difference

ForwardDiff 1.4.6 evaluates `Dual(hi,1) > hi` as true. Julia's raw basis-row
evaluation consequently takes the outside-zero branch, including changing the
Dual result's primal relative to an ordinary Float64 call. The original saved
derivatives are preserved. Separate SciPy analytic derivative fixtures enforce
the approved interior-at-equality JAX convention and agree with every original
query derivative away from the identified right endpoint.

This is a declared difference, not exact Julia AD parity at that boundary.
The outside-zero basis extension can be discontinuous there; do not advertise a
two-sided mathematical derivative. Neither implementation's source was changed
to make that comparison disappear.

## Independent read-only review

Sonnet 5.5, medium effort, session `157eec6d-32b0-43f2-bfb1-131fcd8eedad`, found no
confirmed numerical bug, but identified useful coverage gaps. Main review added:

- N4096/Q1 and large-Q JAXPR inspections, including JVP/HVP traces and a tighter
  linear-size bound;
- analytic query HVP checks including the selected endpoint convention;
- vmapped plan/site-gradient comparisons with changing geometries;
- public Float32/64 near-singular constructor rejection;
- explicit dense-basis guard and inert PyTree reconstruction tests;
- isolated default-x64-off checks.

These tests are unconditional in normal discovery and passed the 46-test gate.
Its first run exposed two **test-harness mistakes**, preserved in
`review-focused.log`: when Q=1, `(Q,N)` aliases legitimate vectors/RHS storage;
and jaxace's existing import-time precision switch overrides JAX's flag alone.
The corrected shape check keeps the linear-size cap and all Q>1 exclusions;
the subprocess uses the existing package opt-out. Production behavior and
numerical tolerances were unchanged.

The suggested use of private tracer `.primal` fields is
rejected: transformed geometry already has a documented validity precondition.
Invalid traced queries have no promised gradient contract. The review's claim
that dense materialization guards were outside the design is incorrect; the
approved design explicitly retains guards for requested dense operations.

## Performance: rejected/incomplete campaign

The three-way benchmark compares committed dense, preserved banded/dense, and
compact code. It covers vectors/matrices, tiny and production grids, large query
grids, coefficient/value/site gradients, interpolation, construction and prepared
application. Compilation is excluded, all outputs synchronized, CPU pinned to
core 2, A/B/C/C/B/A ordering, 30 rounds and 5 warmup rounds per trial.

**Rejected as an acceptance matrix:** the run stopped after 19 of 282 trials
because the CPU did not reach the <=55°C cooldown criterion within 180 seconds.
The failure was `tiny-interpolation-banded_dense-0`. Preserve `timings.log` and
`timings.json`; do not present this incomplete matrix as a speedup result or
relax thermal criteria. Several partial repeated measurements were also noisy.

The prior candidate's accepted negative results (54.5% slower site gradients,
11.5% slower preparation for its specific N81×8192 CPU campaign) remain in
`BANDED_BSPLINE_VALIDATION.md`; this port has not erased or resolved them by
assertion. Reliable full timing, including eager-construction synchronization
cost, remains open. No universal performance improvement is claimed.

## GPU scope

The workstation has an RTX 4070 Laptop GPU, but the existing JAX 0.6.2 environment
knows only CPU/TPU backends. A CUDA availability probe failed accordingly.
An isolated `.venv-bspline-gpu` environment was provisioned with the validated
CPU numerical package versions pinned. Existing environments are unchanged.
JAX 0.6.2 reports `CudaDevice(id=0)`. The first GPU run passed 77 tests and failed
eight runtime-branch instrumentation tests: `JAX_PLATFORMS=cuda` disabled the CPU
backend required by test-only `jax.debug.callback`. All eight failures report
the missing callback CPU device, not numerical mismatches. Their rerun with
`JAX_PLATFORMS=cuda,cpu` passed **8/8** in 36.71 seconds, with CUDA first/default
and a separate callback CPU device verified. Thus all **85 selected GPU checks**
passed across the two runs. No callback or tolerance was removed; production
code contains no such instrumentation callbacks.

The bounded GPU selection excludes the explicit large dense-allocation guard,
the large compact-storage test and the three CPU JAXPR shape-inspection cases.
Those five cases passed CPU validation. This is not a claim of a complete GPU
package suite or of large-grid GPU performance.
Laptop GPU timings cannot certify Narval performance;
no cluster job has been submitted.
