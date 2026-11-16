# Mixed-precision Julia references

Generated from the exact merged ACE `c5f9ca0` numerical source by
`scripts/generate_bspline_mixed_reference.jl`. The generator refuses overwrites;
`metadata.toml` records source/generator hashes and Julia/ForwardDiff versions.

Each name records geometry, ordinate and query precision. Stored factors follow
the geometry dtype even when later ordinates are higher precision. Files include
factors, spline/plan coefficients, predictions and ordinate gradients.

The tests retain the existing Float32 budget (rtol 8e-5, atol 8e-6) when geometry
or gradient output is Float32. Full Float64 arithmetic uses rtol 2e-11 and atol
2e-12. A Float64 result from Float32 geometry is not claimed to have Float64
accuracy. Original fixtures in other directories were not regenerated.
