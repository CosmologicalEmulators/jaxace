# Background simplification and origin derivatives

The October 2026 changes remove a redundant near-zero distance branch and use
an integer fourth power in the photon term. Identical single/batch normalization
branches are collapsed, and an unused private NaN helper is deleted. Active
domain masks, three individual neutrino masses, both Neff prescriptions,
lookup tables, quadrature, ODE tolerances and solver limits are unchanged.

## Intentional numerical change

At z=0 the distance value remains exactly zero, but its redshift derivative is
now c/H(0), rather than zero. The mapped quadrature weights vanish at the origin
without a special branch. Tiny nonzero redshifts below 1e-12 now return the tiny
physical distance instead of being clipped to zero. This also restores the
corresponding parameter derivatives. Forward/reverse, scalar/vector and JIT
tests cover the origin, changed near-zero inputs, curvature, individual masses
and both Neff prescriptions.

Outside that deliberately corrected region, saved values matched the baseline;
the maximum relative change in the sampled gradients from the integer power
replacement was 3.4e-16. This is a tested sample, not a universal bitwise claim.

## Validation and performance

The exact staged source snapshot passed **549 tests**, with fixtures restored
from the pinned artifact and all original reference budgets unchanged. The
origin test matrix reuses compiled derivatives across changed redshifts instead
of recompiling the same operation for every value.

Matched CPU measurements compare the same three-mass physics and cosmology,
with nontrivial w0wa and curvature, compilation excluded and every timed result
synchronized. The campaign uses pytest-benchmark in ABBA order with 50 samples
per case and explicit thermal gates. All 16 benchmark cases passed; accepted
windows stayed at 54–55 C. JAX was 0.6.2, Float64, pinned to CPU2.

| Workload | Before median range (µs) | After median range (µs) |
|---|---:|---:|
| E, 50 redshifts | 14.50–14.62 | 14.24–14.29 |
| Distance, 100 quadrature nodes | 428.91–429.35 | 383.83–384.15 |
| D and f | 226.44–226.98 | 221.54–223.39 |
| Gradient of sum(D) | 677.25–684.61 | 669.82–675.93 |

The clear gain is approximately **10.5% less distance evaluation time**. The
other changes are small; no substantial growth/gradient speedup is claimed.
These are within-JAX comparisons, not comparisons to Julia's default order 9.
The matched cosmology uses Ωcb=0.31, h=0.67, masses (0.01,0.02,0.03) eV,
w0=-0.9, wa=0.1 and Ωk=0.01, 50 redshifts over [0,5], and growth tolerances
1e-6/1e-8. Numerical equality was checked before measuring each pair, and changed
mass inputs were checked to remain dynamic.

Raw synchronized ABBA results, baseline source and the campaign harness are
retained in `.benchmarks/performance_review_20261028/` locally. No thermally
rejected Julia distance-gradient timings enter these JAX results.

The existing `test_growth_benchmarks.py` remains the general current-version
background/growth benchmark entry point. Use `--benchmark-save` and
`--benchmark-compare` across revisions, with identical input shapes, dependency
versions and CPU affinity. Do not interpret different quadrature orders or
different growth tolerances as a language-performance comparison.

Dense-output memory, alternative table layouts and import-time quadrature setup
remain profiling candidates. Reducing the solver step cap would restrict
supported tight-tolerance evaluations; it is not an accepted simplification.
