# Combined release dependency updates

The existing release branches incorporate the useful changes from ACE #95
(JET 0.12), ACE #99 (DataInterpolations 10), and jaxace #55
(pytest-benchmark 5.3.0). No separate dependency PR was merged or closed.

## Included changes

- ACE retains its older supported dependency versions and adds JET 0.12 and
  DataInterpolations 10. Julia 1.12 CI asserts **both actual loaded versions**;
  Julia 1.10/1.11 can resolve their supported older JET. Version assertions do
  not select a reduced test suite. All default extension tests remain enabled.
- The Python lockfile imports the two package-entry updates from #55:
  pytest-benchmark 5.3.0 and its py-cpuinfo2 10.1.1 dependency. Project metadata
  is unchanged; quadax is not reintroduced.

## Current Python CI test corrections

The mixed-precision emulator test used unseeded random weights and divided by
the output magnitude. Cancellation can make the Float64 output arbitrarily
small, invalidating that pure relative-error metric. A deterministic public-path
reproducer gives an output near 1e-10 and Float32 arithmetic error of about
1.3e-8, yet the old relative-error test reports an error near 65.

The replacement covers ordinary and deliberately near-zero outputs, retains
dtype checks, and compares each dtype with an independent NumPy Float64 network
evaluation using that dtype's represented parameters. The relative allowance
is retained; the absolute roundoff budget scales with machine epsilon and the
magnitudes of the intermediate affine sums, using tanh's 1-Lipschitz bound.
There is no production neural-network change or scientific-fixture relaxation.

The old quadrature timing tests did not synchronize their warmups and enforced
fixed 0.1/0.5-second limits on shared CI runners. Their three workloads now use
synchronized pytest-benchmark warmup and measurements, with finite-output,
shape and scalar-versus-vector numerical checks. CI saves benchmark JSON as
diagnostic evidence. Those uncontrolled runner timings are not a performance
acceptance claim; the separate controlled B-spline performance gate remains open.

## Validation

Local evidence is under `.benchmarks/dependency_merge_20261027/`.

- Poetry **2.1.4**, matching CI: `poetry check --lock` passed.
- The two changed lock entries exactly match the reviewed #55 integration.
- Focused Python checks: **20 passed** both in the existing JAX 0.6.2 environment
  and a fresh environment installed from the complete updated Poetry lockfile;
  **20 also passed** on the supported JAX 0.4.38 environment.
- Focused Julia `Pkg.test()` with both target-version assertions: **40 type-
  stability + 17 cubic-spline assertions passed**. Target dependency updates
  occur in Pkg's disposable test sandbox, not the source project.
- Full Python rerun with jaxace installed: **431 passed** on the updated locked
  environment (`python_locked_full_v2.log`). This local count includes the
  pre-existing untracked Chebyshev comparison test, which is not part of the
  PR. Benchmark timing collection was disabled during concurrent correctness
  runs; these results do not certify performance.
- After those jobs finished, the enabled benchmark/report smoke test passed
  all **3 cases**. Its JSON contains the 7/50/100-redshift workloads with 30
  measurement rounds each. This validates report plumbing, not performance.
- Full Julia `Pkg.test()` passed **11,447/11,447 assertions**, with both
  target-version assertions enabled and every default extension test retained
  (`julia_full.log`). The resolved targets were JET **0.12.3** and
  DataInterpolations **10.1.3**; the surrounding environment retained Mooncake
  0.5.62, Reactant 0.2.290 and Enzyme 0.13.209. The five pre-existing unrelated
  Julia edits remain unchanged and outside this integration.
- The first full Python run had **430 passed, 1 failed** because the new
  environment had been installed with `--no-root`: the package-version metadata
  test correctly detected missing jaxace distribution metadata. The environment
  now installs the current project, matching CI. The failed log is preserved as
  `python_locked_full.log`; no test or application code was changed for it.

## Deliberately excluded

- **SimpleChains 0.5:** registered Lux 1.31.4 still restricts it to the 0.4
  family. The upstream allowance in Lux #1748 is merged but not registered.
- **JET 0.13:** registered Mooncake 0.5.66 still restricts JET to <=0.12.x.
- **DataInterpolations 9 / Artifacts broadening / quadax:** respectively already
  supported, unnecessary for supported Julia runtimes, and removed from jaxace.

These blockers are not bypassed with development-branch pins or silent fallback
to an older version while claiming the requested version passed.
