# Frozen banded-construction references

These text fixtures precede the JAX banded implementation. They must not be
regenerated to make a new implementation pass. The earlier spline fixtures
elsewhere in `tests/data/` are unchanged.

- Julia generator: `scripts/generate_banded_spline_reference.jl`, run using the
  existing ACE benchmark environment with ForwardDiff and the saved adaptive
  N81 grid as its sole argument. Source commit and actual numerical-file hashes
  are in `julia_metadata.toml` (the source had a callable-specialization-only
  follow-up, so a commit hash alone would not identify the exact files).
- Original dense JAX generator: `scripts/freeze_dense_bspline_reference.py`.
  Version, commit and actual source hash are in `jax_metadata.json`.
- Both generators refuse to overwrite existing outputs. They use identical
  saved inputs, not separately reconstructed floating-point grids.

Cases N=4,5,6,8,81, Float64, three RHS columns. `inputs` columns are the site,
three sampled values and three coefficient-loss weights. `query` stores the
13 evaluation sites. The scalar coefficient loss is `sum(coefficients * weights)`.
Julia value/site derivatives were generated with ForwardDiff, independently of
the reverse-mode rules. `factors` uses the seven-band layout of Julia; `transpose`
is its cached-factor transpose solution for the saved loss weights.

The production-sized N81×8192 test derives independent RHS columns by linear
scaling of a saved reference column. It exercises the actual large-batch public
path without storing tens of megabytes of redundant coefficients in git.
