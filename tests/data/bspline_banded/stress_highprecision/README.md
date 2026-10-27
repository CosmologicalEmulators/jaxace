# Additional precision-stress references

These are independent **256-bit Julia ForwardDiff** references generated after
review exposed cancellation-sensitive cases. They do not replace or regenerate
any pre-port Julia/dense-JAX fixture in the parent directory.

`freeze_bspline_stress_inputs.py` saved the exact Float64 JAX inputs with 17-digit
round-tripping text. Julia promotes those binary Float64 values exactly to
BigFloat, without recomputing sin/cos or the grid. The raw-k case exponentiates
the saved log-k N81 grid; the N8 clustered case has a 1e-8 first spacing.

`generate_bspline_stress_reference.jl` saves coefficients and the quadratic-loss
site gradient from the original Julia implementation, using ForwardDiff's full
coefficient Jacobian. `uncancelled` is the sum of absolute *loss-Jacobian* terms.
That diagnostic alone misses cancellation already inside the collocation
contraction, so `generate_bspline_stress_scale_reference.jl` independently saves
the earlier-stage absolute collocation-contraction scale as a separate field.
It uses Julia's public basis rows and a 256-bit adjoint, not the JAX port.
Both reference generators refuse overwrites and record provenance hashes.

Coefficient tests compare both Float64 algorithms with the high-precision
reference, with an explicit matrix-conditioning allowance. Gradient tests use
dimensionless local-gap displacement coordinates, `t_i + gap_i * delta_i`, with
the gaps held fixed: an absolute gradient tolerance in physical raw-k units is
not meaningful across several decades or near-coincident sites. The original
ordinary-grid, Float64 fixture and geometry tolerances are unchanged. Physical
gradient discrepancies are reported separately in the validation document;
this test does not claim uniform 1e-12 physical-gradient accuracy.
