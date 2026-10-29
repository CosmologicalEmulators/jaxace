# Missing public-basis references from the exact merged Julia numerical source.
# Use the existing ACE benchmark environment. Never update a dirty checkout.
using AbstractCosmologicalEmulators, ForwardDiff, DelimitedFiles, SHA, TOML

length(ARGS) == 1 || error("usage: generator.jl /absolute/path/to/pinned/julia_source")
const SOURCE = abspath(only(ARGS))
const OUT = normpath(joinpath(@__DIR__, "..", "tests", "data", "bspline_api"))
ispath(OUT) && error("reference directory already exists; refusing to overwrite")
const EXPECTED = Dict(
    "src/cubic_b_spline.jl" => "824b1f806270ac87b4702f68f6f7f8f91f4ab0e2f07dc67e3c81727da25a55c5",
    "src/threads.jl" => "e0b5aeca150adaad398bc934d5e4dc3c6aae8ee9f1fde87b5947a6d5b005d3d3",
)
for (file, expected) in EXPECTED
    bytes2hex(sha256(read(joinpath(SOURCE, file)))) == expected || error("wrong source: $file")
end

module JuliaReference
using LinearAlgebra
import AbstractCosmologicalEmulators
const Adapt = AbstractCosmologicalEmulators.Adapt
const FFTW = AbstractCosmologicalEmulators.FFTW
include(joinpath(Main.SOURCE, "src", "threads.jl"))
include(joinpath(Main.SOURCE, "src", "cubic_b_spline.jl"))
end
const R = JuliaReference

function make_basis(p, multiplicities)
    knots = isempty(multiplicities) ? eltype(p)[] :
        vcat((fill(p[i+1], m) for (i, m) in enumerate(multiplicities))...)
    return R.CubicBSplineBasis(domain=(first(p), last(p)), internal_knots=knots)
end

function freeze()
    cases = (
        ("polynomial", [0.0, 1.0], Int[]),
        ("simple", [-1.0, -0.6, 0.2, 1.3, 2.0], [1, 1, 1]),
        ("double", [0.0, 0.3, 0.7, 1.0], [2, 1]),
        ("triple", [0.0, 0.35, 0.65, 1.0], [3, 2]),
    )
    mkpath(OUT)
    meta = Dict{String,Any}(
        "ace_commit" => "c5f9ca06d1cd00a6b224ea505bb36f612477004e",
        "source_sha256" => EXPECTED, "julia_version" => string(VERSION),
        "forwarddiff_version" => string(pkgversion(ForwardDiff)),
        "generator_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "dtype" => "Float64", "index_base" => 1,
        "loading" => "Exact original threads/cubic_b_spline files in isolated module; existing ACE supplies dependencies only",
        "cases" => Dict{String,Any}(),
    )
    for (name, p, multiplicities) in cases
        basis = make_basis(p, multiplicities)
        lo, hi = first(p), last(p)
        q = sort(unique(vcat(lo - 0.2, p, collect(range(lo, hi; length=9)), hi + 0.2)))
        nq, n = length(q), R.nbasis(basis)
        rows = [R.basis_row(basis, x) for x in q]
        indices = reduce(vcat, [reshape(collect(row.indices), 1, 4) for row in rows])
        weights = reduce(vcat, [reshape(collect(row.values), 1, 4) for row in rows])
        cot = [sin(0.3 * i + 0.7 * j) for i in 1:nq, j in 1:n]
        # Geometry gradients use interior queries, away from movable knot ties.
        qg = lo .+ (hi - lo) .* [0.073, 0.213, 0.487, 0.823]
        cotg = [cos(0.2 * i + 0.4 * j) for i in eachindex(qg), j in 1:n]
        fields = Dict(
            "parameters" => p, "knots" => R.knot_vector(basis), "query" => q,
            "indices" => indices, "weights" => weights,
            "matrix" => R.basis_matrix(basis, q),
            "support" => reduce(vcat, [reshape(collect(R.basis_support(basis, i)), 1, 2) for i in 1:n]),
            "query_cotangent" => cot,
            "grad_query" => ForwardDiff.gradient(x -> sum(R.basis_matrix(basis, x) .* cot), q),
            "geometry_query" => qg, "geometry_cotangent" => cotg,
            "grad_parameters" => ForwardDiff.gradient(x -> sum(R.basis_matrix(make_basis(x, multiplicities), qg) .* cotg), p),
        )
        for policy in (:clamp, :zero)
            stencil = R.basis_stencil(basis, q; extrapolation=policy)
            fields["$(policy)_indices"] = hcat(stencil.i1, stencil.i2, stencil.i3, stencil.i4)
            fields["$(policy)_weights"] = hcat(stencil.w1, stencil.w2, stencil.w3, stencil.w4)
        end
        for (field, data) in fields
            all(isfinite, data) || error("nonfinite reference: $name $field")
            writedlm(joinpath(OUT, "$(name)_$(field).txt"), data)
        end
        meta["cases"][name] = Dict("multiplicities" => multiplicities, "nbasis" => n)
        println("FROZEN ", name, " basis functions=", n, " queries=", nq)
    end
    open(io -> TOML.print(io, meta; sorted=true), joinpath(OUT, "metadata.toml"), "w")
end
freeze()
