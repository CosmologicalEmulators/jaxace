# Independent reviews: JAX banded construction

Base: jaxace `368596b`, uncommitted implementation. The original arithmetic
snapshot and subsequent repairs are pinned under `.benchmarks/banded_validation/`.
Reviews were read-only; main-agent tests provide execution evidence.

## GPT-6 Astra / High / ChatGPT OAuth

The Computer Scientist, conversation
`conv-a63717e1-53af-4e10-9c45-462e13d45eb4`, verified the requested model,
provider and effort. The first report completed around 17:12 UTC, but a harness
desynchronization lost its notification. Recovery returned the existing report,
not a duplicate broad review.

**P2, confirmed and repaired:** integer geometry overflowed before conversion
to floating point. Int32 `[-2^30,-2^29,2^29,2^30]` and analogous Int64 powers
62/61 yielded `[1,-1,-1,1]` for constant-one interpolation, eagerly and under
JIT. Main reproduced both cases. `local_basis` now uses the original dense
basis promotion rule before any geometry arithmetic. Four direct regression
tests passed: both integer types, vector/matrix ordinates, eager/JIT public
interpolation, coefficients and ordinate gradients. The repaired **67/67**
focused run passed. Astra's bounded follow-up accepted the repair with all three
source/test hashes matching before and after its review. The repaired full
suite passed **371/371** on JAX 0.6.2; the repaired focused suite passed
**67/67** on JAX 0.4.38. All final execution gate exit codes are zero.

No other blocking algorithm/AD defect was identified in the original report.
The documented exception-policy difference was accepted; JAX minimum-version,
GPU and performance certification were explicitly not inferred.

## Claude Opus 5.5 / High

Session `84bb3615-f266-4fc5-a98a-2b9e8eb69238`, actual reported model
`claude-opus-5-5`; High supplied by the tool call. Its original report likewise
already existed after the harness interruption and was recovered rather than
repeated. It found no confirmed production defect by source inspection.

- **P2, plan live geometry coverage:** added site/query JVP and VJP for N4/5/8/81,
  Q1/7, clamp/zero, inside/outside; vmapped geometry; preserved host/throw behavior.
- **P2, conditioning evidence:** added raw-k, near-coincident and traced-duplicate
  cases; generated independent 256-bit native Julia ForwardDiff references after
  Float64 algorithms disagreed. Original fixtures/tolerances were not replaced.
- **P3, positive fast-path evidence:** direct bordered-solver comparisons plus
  runtime callback instrumentation prove the branch executes and an exact
  forbidden-factor perturbation rejects it. An always-general mutation fails.
- **P3, stale plan-memory docstring:** corrected.

The follow-up accepted these bounded coverage/reference changes, conditional
on execution and publication of the physical-gradient discrepancies. Those
numbers and the deliberately loose conditioning/local-gap gate are now in
`BANDED_BSPLINE_VALIDATION.md`. The 63-test pre-integer-repair run passed.

GPU conditional costs remain an unmeasured hypothesis, not a certified defect
or reason to modify arithmetic speculatively. Traced unsorted sites remain an
input-precondition violation, not a newly supported runtime-validation policy.

## Snapshot discipline

The pre-integer-repair timing matrix passed all 20 cells. The following broad
suite was interrupted when the recovered Astra report exposed the real integer
bug, so it is not claimed as final acceptance. Final gates must use the repaired
source. Rejected timing attempts and failed diagnostic assertions are retained;
no partial timing matrix is presented as a complete result. The repaired source
then passed all 20 paired timing cells; the geometry-gradient and plan-preparation
regressions are explicitly reported in `BANDED_BSPLINE_VALIDATION.md`, not
converted into an unqualified performance acceptance.

No commits, pushes, version changes or releases were made for this port.
