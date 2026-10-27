# Paired B-spline review repairs

## Scope and preserved work

These repairs address the independent Computer Scientist (Astra / ChatGPT / max)
and Claude (Opus 5.5 / max) reviews of jaxace `6963807` and
AbstractCosmologicalEmulators.jl `c5f9ca0`. Original reports and reproduction
scripts are preserved locally under `.benchmarks/dual_review_20261026/`.

The Julia primary checkout was fast-forwarded from `5421f07` to `c5f9ca0` after
backing up its five unrelated tracked modifications. Git restored those edits
cleanly: four files remained byte-identical and the module retained precisely
the original additional `load_trained_emulator` export. Those changes are not
part of this repair. Neither original reference fixtures nor tolerances were
rewritten to make the new tests pass.

## Confirmed defects and corrections

| Finding | Correction | Focused regression |
|---|---|---|
| Julia plan aliases source/query arrays; Mooncake doubles gradients for `Plan(t,t)` | Independently snapshot constructor geometry before entering AD primitives | Shared vs copied geometry, vector/matrix coefficients, post-construction input mutation |
| Julia constructor pullback mutates incoming cotangents | Copy projected fields before accumulation | Repeated VJP, seed preservation, site/knot/query/weight contributions |
| Julia standalone stencil loses the direct `.query` cotangent | Accumulate the direct query contribution | Direct and combined query/weight losses |
| Julia integer basis arithmetic overflows before promotion | Widen integer relative differences before converting them; preserve small spans at large offsets | Overflow and translated-grid controls, rows, stencils, dense matrices |
| Julia Dual ordering changes endpoint primals or throws | Make geometric branch decisions on recursively extracted primal values | All extrapolation policies, site/query gradients, nested Duals |
| Julia detached spline clamp derivative targets an unread `sites` field | Accumulate the boundary contribution on domain knots | Raw-knot and detached-site derivatives |
| Julia dense basis materialization lacks a reverse rule | Contract row cotangents with local basis jets | Public query/geometry AD, endpoint/outside/empty queries |
| Julia SW equality can produce an out-of-band write | Strict support bounds except the two permitted domain endpoints | Invalid collocation support boundary |
| Julia transpose solve lacks RHS dimension checks | Match forward-solve validation | Short/long vectors, matrices, zero-column matrices |
| Python raw-knot clamp derivative uses different slots from its declared domain | Use knot slots `3` and `-4` | Independent dense-recursion raw-knot VJP |
| Python endpoint/site derivatives lacked an independent boundary check | Add dense-recursion/solve and one-sided SciPy comparisons | JVP/VJP at endpoints and a site, vector/matrix, clamp/zero |
| Python `throw` documentation mentioned only `jit` | Explicitly include `grad` and `jvp` tracing | Unjitted transformation rejection |
| Integer translation and mixed-signedness edges | Exact common-integer span/difference arithmetic; floating fallback when x64-off promotion cannot represent both dtypes | Translations, exact Bernstein extrema, mixed int32/uint32 subprocess |
| Dense-basis reverse indexing and clamp promotion | Enumerate flattened query rows; restore query shape; preserve promoted clamp result type | Cartesian views, scalar/matrix/tuple/0-D queries, inferred mixed/Dual types |
| Reactant zero extrapolation can generate NaNs before masking | Bound coordinates before recurrence; select zero rather than multiply invalid intermediates | Compiled Float32/Float64, scalar-series/matrix outputs, Enzyme query gradients, A/B/A inputs |

The Python private-helper clamp test now supplies a full clamped knot vector,
as production does, rather than two endpoints. Its five derivative assertions
are unchanged. No high-level boundary convention was changed.

## Evidence ledger

Logs below are local artifacts under `.benchmarks/review_fixes_20261026/`.

- `julia_before.log`: the normal public package reproduced **48 failures and
  6 errors**, with 51 passing assertions, before implementation changes. An
  earlier launch failed on direct test access to an indirect dependency; the
  focused file now imports that module through the already loaded package.
- `julia_after.log`: the initial host repair passed **105/105** assertions.
- `python_before.log`: **5 passed, 1 failed**, exposing the raw-knot domain-slot
  inconsistency independently of the package's own local recurrence.
- `python_after.log`: **27 passed**, including new regressions and unchanged
  merged-Julia API fixtures.
- `reactant_before.log`: **116 passed, 12 failed** on the actual compiled CPU
  path. Huge *finite* outside queries produced NaN zero-policy gradients for
  vector ordinates, and NaN values and gradients for matrix ordinates. This is
  now reproduced, not merely the original reviewers' unverified hypothesis.
- `julia_extended_after.log`: **123 host + 128 compiled** assertions passed.
- The independent follow-up caught a defect in the first integer repair:
  converting absolute coordinates erased an 8-unit interval near `2^62`.
  Julia now widens integer differences; Python uses unsigned modular magnitudes
  after exact integer comparisons. The unchanged floating/Dual path remains
  covered. `julia_review_v2.log` passed **129** assertions;
  `python_review_v2.log` passed **30** tests.
- Astra independently accepted the host fixes, including **421 integer/shape
  checks**, **14 additional ownership/reuse checks**, **86 Python integer
  cases**, and later **42 Julia / 13 Python** narrow edge checks. These are
  targeted independent checks, not full suites or performance measurements.
- Opus accepted the original substantive findings as repaired. Its further
  Cartesian-indexing, clamp-promotion and x64-off mixed-integer observations
  were addressed and independently rechecked by Astra. Remaining low-level
  borrowing, integer-AD and finite-device-input contracts are explicit in the
  Julia README rather than overstated as universal parity.
- `python_full_v2_062.log`: **425 passed** before the final narrow dtype guards
  and extra regression coverage. Final-version full-suite runs are separate.
- `python_final_0438.log`: **429 passed** on JAX 0.4.38, on the final numerical
  implementation. The benchmark-contract test was added after this run's
  collection; its separate result is recorded below. The unrelated local
  untracked Chebyshev test remains included in local full-suite counts.
- `python_final_062.log`: **430 passed** on JAX 0.6.2, including the final
  numerical implementation and the new benchmark-contract test.
- `julia_legacy_host_pkg.log`: **148 assertions passed through `Pkg.test()`**
  with ForwardDiff 0.10.39 and Mooncake 0.5.24. Default `Pkg.test()` still
  includes every extension; the explicit `bspline_review_host` argument is
  only a focused compatibility entry point.
- `reactant_structured_gradient.log`: **176 compiled assertions passed**,
  now checking coefficient as well as query cotangents and restoring the
  previous backend. Two earlier test-harness attempts captured traced data
  through a constant closure (zero coefficient gradients / region-isolation
  errors); the passing test supplies the complete spline and queries as
  explicit differentiated arguments. No library workaround was added for
  those harness failures.
- `gpu_final.log`: **42 selected tests passed** on JAX 0.6.2 CUDA, including
  integer extrema/translation, mixed precision, public basis APIs and JVP/VJP/
  second derivatives. CPU remained available for test-only callbacks. This is
  targeted GPU correctness evidence, not a full GPU suite or timing result.
- `docs.log`: strict MkDocs build passed. `benchmark_contract.log`: the eager
  preparation execution-contract test passed. `eager_benchmark_smoke.log`:
  all five distinct geometry cases agreed across the three real implementations
  in eager preparation; **no timings measured**. Collection confirms 312
  required benchmark trials.
- Full-suite setup failures are retained: the first Julia run was interrupted
  after follow-up edits superseded its source snapshot. A subsequent attempt
  incorrectly put a backend-selection `--eval` in `julia_args`, which also
  reached package precompilation and failed there. The replacement uses a
  process-scoped CPU environment and normal `Pkg.test` arguments. Default
  extension coverage is retained. None of those attempts is counted as a
  numerical test pass.
- `julia_full_final_v2_pkg.log`: the replacement full **`Pkg.test()` passed
  11,447/11,447 assertions**, including all extension and new regression tests
  (30m59.2s test time). Its resolved test environment used ForwardDiff 1.4.6,
  Mooncake 0.5.65, Zygote 0.7.14, Reactant 0.2.299 and Enzyme 0.13.216. These
  versions differ from the provisioned focused-test environment; both results
  are retained rather than conflated. No tests or tolerances were suppressed.
- `benchmark_contract_0438.log`: the additional benchmark-contract test also
  passed separately on JAX 0.4.38, closing the collection-time coverage gap.

## Remaining performance gate

No repaired-path speedup is claimed. The earlier thermal rejection after
19/282 trials remains a rejected campaign. The previously measured 54.5%
site-gradient and 11.5% preparation regressions remain open for their measured
workload until a valid matched comparison supersedes them. The three-way matrix
now also requires `plan_construction_eager`, including host validation and
device-to-host synchronization, without wrapping that constructor in `jit`.
The completeness checker requires all **312 trials**; the original 282-trial
campaign remains historically incomplete/rejected. A regression test verifies
that the eager operation calls its constructor without tracing, including on
every timed invocation. This repairs measurement coverage, not performance.

The suggested GPU conditional rewrite has not been applied: its benefit was a
hypothesis, not a measured result. Correctness repairs do not establish
performance acceptance or authorize a merge/release.

At 2026-10-27 04:23 UTC, after our correctness jobs had finished, the preflight
found two unrelated `blast_play` Julia jobs consuming roughly one CPU each and
CPU sensor readings up to 83°C. Timing was **not started**: this fails compute
isolation and the existing ≤55°C cooldown requirement. Those other processes
were not stopped or modified. Performance acceptance remains open until an
isolated, thermally valid complete matrix can be obtained.
