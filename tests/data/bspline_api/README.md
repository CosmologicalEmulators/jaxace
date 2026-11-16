# Merged Julia public-basis references

Generated once by `scripts/generate_bspline_api_reference.jl` from the exact
`src/threads.jl` and `src/cubic_b_spline.jl` files at merged ACE `c5f9ca0`.
Both source hashes are checked before execution and recorded in `metadata.toml`.
The existing ACE environment supplies dependencies; its checked-out numerical
implementation is not used to produce these arrays.

Cases cover a polynomial basis, simple interior knots, double knots and triple
knots. Files contain full knots, queries, four-entry rows, dense matrices,
supports, clamp/zero stencils, query derivatives and geometry derivatives.
Indices are the original **one-based Julia indices**. Python tests subtract one.
Geometry parameters move repeated knots together; their multiplicities and
cotangents are recorded. Query derivatives at knots use Julia's selected span.

These augment, not replace, earlier not-a-knot solver and high-precision stress
references in `bspline_banded/`. Do not overwrite existing fixture files.

## Known Julia AD boundary discrepancy

The original `*_grad_query.txt` outputs are retained verbatim. At the right
endpoint they are zero because ForwardDiff 1.4.6 evaluates `Dual(hi,1) > hi`
as true. Consequently Julia's `basis_row` executes its outside-zero branch;
even that Dual evaluation's primal differs from the ordinary Float64 call.

The approved JAX contract retains the interior derivative at endpoint equality,
as its existing high-level interpolators already do. The **additional** files
`*_grad_query_interior.txt` implement that explicit convention using SciPy's
analytic B-spline derivative. Their generator verifies agreement with the
original Julia primal matrices and with every saved query derivative except
the identified right endpoint. Both sets are tested; no tolerance was relaxed.
`boundary_metadata.json` records the discrepancy, input hashes and provenance.
This does not modify Julia or claim identical endpoint AD behavior. With the
outside-zero extension, basis functions can be discontinuous at the domain
boundary; the supplemental reference enforces the chosen interior convention,
not a two-sided derivative there.
