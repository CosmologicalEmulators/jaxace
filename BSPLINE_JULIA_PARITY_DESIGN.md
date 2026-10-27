# Full Julia B-spline parity for jaxace

**Status: approved and implemented; final validation/acceptance still in progress.**

Requested by Marco on 2026-10-26: the JAX implementation should be equivalent
to the Julia implementation, not merely reproduce the main interpolator's output.
Scope here is the B-spline subsystem, not an unrelated rewrite of backgrounds,
emulators or the rest of either package.

## 1. Source baseline and ownership

- Julia specification: ACE `origin/develop` at
  `c5f9ca06d1cd00a6b224ea505bb36f612477004e`, merging #97.
- Published JAX candidate: `origin/develop` / PR #57 at
  `368596b916ae11265883cfa5f8ce3d65f70693cf`.
- Existing **uncommitted** JAX work: `jaxace/_bspline_banded.py`, its integration
  into `utils.py`, tests, fixtures, design, review and validation documents.
  Preserve and extend this work; do not overwrite it or claim it is already in
  #57. Record source hashes before implementation.
- Preserve all unrelated dirty/untracked files in both repositories. The Julia
  primary checkout is behind the remote merge and has unrelated user edits;
  do not pull, stash, reset or replace it to obtain references.
- This document proposes no commit, push, merge, release or dependency change.

The earlier `BANDED_BSPLINE_DESIGN.md` deliberately retained dense query
evaluation and dense plans. This proposal **expands that scope**. Existing
numerical fixtures and negative results remain authoritative evidence; their
acceptance does not certify these additional APIs or algorithms.

## 2. Meaning of equivalence

Required: the same supported mathematical operations, knot/domain conventions,
shapes, extrapolation behavior, usable geometry derivatives, and compact-plan
capability. Bitwise equality, identical internal object layouts, Python copies
of Julia's AD backends, and identical CPU/GPU timings are not the target.

Python indices are zero-based; Julia reference indices must be converted once
at the comparison boundary. Do not otherwise change knots or coordinates.

### Public contract

| Julia operation | JAX target |
|---|---|
| `CubicBSplineBasis(knots)` | Immutable registered PyTree with a validated full knot vector |
| Keyword basis construction with domain/internal knots or full knots | Same mutually exclusive alternatives; omitted internal knots means four polynomial basis functions |
| `knot_vector`, `internal_knots`, `nbasis`, `bspline_domain`, `basis_support` | Public accessors; support indices follow Python's zero-based convention |
| `basis_row` | Four active indices and weights; zeros outside the domain; interior/right-hand span convention at knots |
| `basis_stencil` | Compact four-entry rows, query coordinates, `throw`/`clamp`/`zero` policies |
| `basis_matrix(...; storage=:dense)` | Explicit dense materialization from local rows; reject unsupported storage modes |
| `CubicBSpline(u,t)` | Prepared coefficients and basis, vector or column-batched matrix values |
| `cubic_b_spline_interpolation` | Construct/evaluate through the same compact implementation |
| `CubicBSplinePlan(t,q)` and `plan(u)` | Reusable band factors plus local query stencil; no mandatory dense operator |
| `bspline_basis(spline_or_plan)` | Return the actual basis |
| `bspline_coefficients(spline)` | Return prepared coefficients |
| `bspline_coefficients(plan,u)` | Reuse factors, with implicit differentiation through construction geometry |

Retain JAX's existing `prepare_cubic_b_spline`, `evaluate_cubic_b_spline` and
`prepare_cubic_b_spline_plan` helpers and the existing root/`jaxace.utils`
imports. High-level spline/plan constructors continue to derive the unique
not-a-knot basis; **custom knots are only a low-level basis capability**.
No speculative custom-knot fitting API is added to high-level constructors.

The Julia factorization type and raw `solve`/`solve_adjoint` functions are
qualified implementation interfaces, not exported high-level AD inputs.
Implement their numerical capabilities internally and test forward/transpose
solves directly; do not advertise raw LU entries as differentiable public
geometry. A request for additional public raw-factorization APIs is a separate
API decision, not necessary to provide the exported Julia functionality above.

### Shapes and domain rules

- At least four finite, strictly increasing high-level sites.
- `u`: `(N,)` or `(N,M)`, including `M=0`; first dimension must equal N.
- Spline/one-shot scalar query: scalar result or `(M,)`; vector query: `(Q,)`
  or `(Q,M)`. Plans take a one-dimensional query grid. Preserve empty queries.
- Basis knots: finite nondecreasing vector, at least eight entries, exactly
  four copies of each endpoint, interior multiplicity at most three, and no
  degenerate four-interval support. Match Julia's actual validation checks.
- `basis_row`/`basis_matrix` return zero outside support, not polynomial
  extrapolation. `basis_stencil` defaults to `throw`; high-level objects default
  to `clamp`. Do not accidentally unify these defaults.
- Reject nonfinite queries eagerly in **all** policies, matching Julia.
  Current JAX clamp/zero handling of infinite queries is not equivalent and
  must not be preserved as if it were the target behavior.
- Preserve the interior polynomial derivative at endpoint equality, zero
  query derivatives strictly outside for clamp/zero, and moving-boundary
  contributions when differentiating sites. At repeated interior knots,
  distinguish the selected one-sided derivative from a globally smooth one.

## 3. Implementation boundaries

### Files

- New `jaxace/_bspline.py`: public types, validation, accessors, extrapolation,
  compact evaluation and plan assembly.
- Existing `jaxace/_bspline_banded.py`: local numerical recurrence, band
  assembly, LU, forward/transpose solve, exact specialization guard/fallback.
- `jaxace/utils.py`: re-export the existing public B-spline names from the
  new module. Remove superseded production implementations, not duplicate
  competing paths. Move independent dense test oracles into test support with
  source provenance; adjust private-helper imports without weakening tests.
- `jaxace/__init__.py`: export the new public basis/coefficient/stencil API.
- Focused tests and new reference generators as specified below; update
  `docs/api.md` and the validation report after each accepted stage.

### Basis and local evaluation

Reuse the existing four-value `local_basis` recurrence after validating it on
general allowed knot multiplicities. Promote integer geometry **before** any
subtraction, retaining the existing integer-overflow regression.

Use binary-search spans with Julia's right-span convention and a bounded
four-entry gather. For zero-outside evaluation, select safe in-domain
coordinates before recurrence evaluation and then zero the row. Do not evaluate
an overflowing outside polynomial and hope a final mask repairs its derivative.

Represent a row with `(4,)` indices/weights and a stencil with `(Q,4)`
indices/weights plus queries. Row/stencil objects are immutable PyTrees.
Provide Julia-style read access to stencil columns where useful without storing
duplicate copies. Integer indices and policy strings are nondifferentiable;
knots, weights and queries remain ordinary differentiable arrays.

Evaluate using four gather-and-weight terms. Avoid a mandatory `(Q,4,M)`
temporary for large batches; an unrolled loop of fixed length four is acceptable.
Never materialize a `(Q,N)` basis matrix in ordinary spline evaluation.

### Compact prepared plan

Store sites, basis, **live collocation bands**, stopped LU factors, stencil and
policy. Construction costs O(N + Q log N), with O(N+Q) retained storage.
Applying to M series costs O(NM + QM), plus output storage. These bounds exclude
compiler and AD tape memory, which must be inspected separately.

Split factorization from application of the existing implicit solver:

1. Constructor builds live bands and a stopped factorization once.
2. An internal solve-with-factors operation uses `custom_linear_solve` with the
   live band matvec and captured stopped factors in both solve directions.
3. `plan(u)` and its pullback reuse those factors. They must not call
   `factor_bands` again or reconstruct a dense inverse.
4. A plan constructed from changed sites recomputes its own geometry; no global
   factor cache or object-identity cache is introduced.

For `A(t)c=u`, the derivatives must implement
`dc = A^-1 (du - dA c)` and the corresponding transpose solve in reverse mode.
Stopping factorization derivatives is correct only while A's geometry remains
live in the implicit solve. Never stop gradients through sites or live bands.

Factor precision is determined by the geometry at plan construction, as in
Julia; a later higher-precision RHS must not secretly refactor the plan. Promote
solve arithmetic appropriately and validate mixed-precision accuracy against
the geometry precision, not merely the output dtype. The current candidate
promotes bands and RHS together before factorization, so this distinction needs
an explicit mixed-dtype regression when adding persistent factors.

Retain the exact bordered-factor guard and full-band fallback from the existing
candidate. Do not assume every allowed custom basis has canonical not-a-knot
factor structure. No change to pivot/arithmetic policy is bundled silently.

### Backward-compatible dense access

Make the existing `plan.operator` a **computed property**, explicitly requesting
dense materialization, with the existing 64 MiB operator guard. It is not stored
or used by `plan(u)`. No automatic dense/compact heuristic or second default plan
type is proposed. This preserves numerical access for callers while permitting
large compact plans; document that repeated `.operator` access reconstructs it.

Preserve existing spline `u`, `t`, `knots`, `coefficients` access where possible;
properties may avoid redundant metadata storage. PyTree layout changes are
explicit: cached compiled thunks and serialized internal layouts are not
cross-version compatibility promises. `tree_unflatten` must not solve or
revalidate traced values.

Dense matrix/operator guards apply to explicitly requested dense operations,
not compact constructors. Replace the old test that requires N4096 construction
to fail with positive compact-construction tests and separate dense-limit tests.
Do not remove safeguards until traced shapes prove quadratic allocation is gone.

## 4. AD and execution contracts

- Eager/JIT/vmap, vector/matrix values, Float32/64 and mixed dtype inputs.
- JVP, VJP and selected site/query Hessian-vector products, not reverse-only
  custom rules that lose forward-mode support.
- Derivatives through public constructor composition, basis accessors,
  coefficient extraction and application. Raw independent factor leaves are
  structural; differentiating a detached object's arbitrary internal buffers
  is not the same as differentiating its construction from sites.
- One compiled function must run A/B/A inputs and two different geometries.
  Vary values, sites and queries separately; do not choose only perturbations
  that leave the derivative unchanged.
- Reusable plans must remain unchanged across forward calls, gradient calls,
  interleaved plans and repeated use. Constructor inputs are snapshots, not
  writable NumPy buffers silently shared with stored geometry.
- Compare regular and specialized solves, their transposes, and derivatives.
  Preserve direct instrumentation proving the guarded branch is actually used.

### Explicit JAX adaptations requiring agreement

1. **Runtime exceptions:** eagerly reject the same invalid domains with Python
   exceptions. Ordinary JIT cannot reproduce arbitrary data-dependent Julia
   exceptions. Keep finite/ordered traced geometry as a documented precondition
   and retain the rejection of dynamic `throw`; host-validated fixed throw plans
   remain usable under JIT. Do not insert host callbacks into production kernels
   or claim identical behavior on invalid traced inputs. Optional checkify
   integration may be evaluated separately; it is not a prerequisite for the
   compact valid-input path.
2. **Near-singular pivots:** unlike the earlier partial port, match Julia's
   10-epsilon pivot rejection in eager construction using the factor's geometry
   dtype. This is an explicit expanded-parity behavior change and needs its own
   regression tests. Ordinary JIT still cannot throw the same Python exception;
   traced solves require admissible nonsingular geometry. Do not silently clamp
   tiny pivots, change a physical grid, or imply condition-independent accuracy.
   Check construction synchronization cost in the performance gate.
3. **Threads/devices:** XLA owns execution. Do not add a fake no-op `set_threads`
   to imitate Julia's task/FFTW controls. Validate CPU/GPU behavior explicitly;
   Julia host-thread tuning and JAX compilation are different mechanisms.
4. **Indexing:** support/row indices are zero-based in Python. This is a language
   adaptation, not a changed basis or shifted interpolation grid.

## 5. Reference → precision → timing gates

### Reference inventory first

Reuse `tests/data/bspline_banded` and earlier committed Julia/SciPy references.
Their provenance records Julia `132a6e5` and numerical-file hashes including
local follow-ups; those hashes are not all identical to the merge tree. Do not
regenerate or overwrite them and do not mislabel them as exact `c5f9ca0` outputs.

Freeze **only missing coverage** from the merged Julia implementation into a
new text-fixture directory with commit/source hashes, inputs, Julia/package
versions and a generator that refuses overwrites:

- Custom bases with zero/one/multiple interior knots, repeated knots of
  multiplicity two/three, support ranges, rows, stencils and matrices.
- Nonuniform not-a-knot cases: N4/5/6/8/81 plus selected larger shapes;
  coefficients, plan outputs and value/site/query derivatives for vector and
  matrix data. Reuse existing cases where they already supply the quantity.
- Endpoint and repeated-knot one-sided derivative conventions; outside-policy
  behavior; scalar outputs and empty vector/matrix dimensions.
- Basis/coefficient accessor composition through plan construction.

Use ordinary Julia/ForwardDiff references; use existing high-precision stress
fixtures when Float64 cancellation is relevant. Do not treat dense Float64 as
an exact oracle or relax prior tolerances after seeing a disagreement. If new
reference execution needs the merged source, use a separately approved existing
reference environment/source snapshot, not a destructive update of the dirty
Julia checkout or an unapproved worktree.

### Focused acceptance before the full suite

1. Basis validation, accessors, rows/stencils/matrices and corresponding
   PyTrees compile and agree with fixtures before switching interpolator paths.
2. Local evaluation matches old dense JAX and saved Julia values and derivatives.
3. Compact plan/coefficient paths reuse factors and agree with forward/transpose
   oracles and all live-geometry derivatives.
4. Inspect lowered/JAXPR shapes for representative N4096/Q1 and large-Q cases:
   no N×N or Q×N work in compact construction/application. Check primal **and AD**.
   AD here means scalar-loss gradients and directional JVP/VJP/HVP operations,
   not an explicitly requested dense Jacobian whose output itself is quadratic.
   A memory-limit exception disappearing is not proof of linear memory.
5. Include all focused tests unconditionally in normal pytest discovery. Retain
   earlier boundary, integer, conditioning, fallback and mask regressions.
6. Run the full package suite on the supported stack and compatibility stack.
   The declared JAX minimum is 0.4.30; existing 0.4.38 evidence does not certify
   it. Resolve an actual minimum-version test or explicitly propose a compat
   change, rather than claiming untested support.
7. GPU correctness/JIT/gradient smoke coverage is required before claiming GPU
   equivalence. Production GPU timing belongs on Narval hardware, not inferred
   from laptop timings. No cluster jobs are authorized by this design alone.

Precision reports include max absolute/relative differences and locations for
each quantity, with predeclared budgets. Differentiate signs/zero crossings
from actual numerical errors; retain conditioning and cancellation diagnostics.

### Performance is a separate gate

The existing candidate's measured N81×8192 CPU results include **54.5% slower
site gradients** and **11.5% slower plan preparation** than dense JAX, despite
faster coefficients and value gradients. Those measurements remain negative
results, not an accepted universal speedup. Source: `BANDED_BSPLINE_VALIDATION.md`.

After correctness, compare the committed dense baseline, current banded/dense
candidate and compact implementation with synchronized pytest-benchmark paired
trials; use BenchmarkTools for Julia reference timings. Separate compilation,
construction and warmed reuse. Cover small N4/8, N81, N512, a large compact-only
case, multiple query counts and RHS counts including 1, 3 and 8192 where memory
permits. Measure one-shot values, both geometry/value gradients, plan creation,
repeated plan application and repeated gradients, not only the band kernel.

Proposed gate: >5% slowdown warns; repeatable >10% slowdown blocks default
promotion unless Marco explicitly accepts a documented tradeoff. Apply this
per relevant workload rather than hiding a geometry regression in an average.
Profile first; do not add custom adjoints or backend heuristics merely to guess
at the cause of the current regression.

## 6. Implementation sequence and review

1. Preserve/hash current work; complete missing Julia reference coverage.
2. Implement public basis API and local evaluation, with focused parity/AD tests.
3. Implement compact plan storage, factor reuse, coefficient helpers and dense
   compatibility access; validate all live-geometry and ownership paths.
4. Integrate exports/docs and replace dense-only guard assumptions; run full CI.
5. Run the performance matrix, address or explicitly escalate regressions,
   then perform independent review of actual code and evidence.
6. Only after acceptance and explicit publication authorization, commit/push the
   scoped work into #57 and update its scope/validation text. No merges/releases.

**Approval requested:** compact `CubicBSplinePlan` by default with explicit
computed `.operator` compatibility access; complete exported Julia basis and
coefficient functionality; the four documented JAX adaptations above. Once
approved, implementation proceeds in these tested stages rather than merely
publishing the existing partial banded port.
