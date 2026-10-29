# Freeze the Julia banded implementation before changing the JAX implementation.
# Run with the existing ACE benchmark project and one argument: the saved N81 grid.
# Outputs are new text fixtures, never replacements for older reference files.
using AbstractCosmologicalEmulators, ForwardDiff, DelimitedFiles, SHA, TOML
const ACE = AbstractCosmologicalEmulators
const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = joinpath(ROOT, "tests", "data", "bspline_banded")
length(ARGS) == 1 || error("usage: generator.jl /absolute/path/adaptive_design_N81.txt")
ispath(OUT) && error("reference directory already exists; never overwrite frozen outputs")
mkpath(OUT)
ace_root = dirname(dirname(pathof(ACE)))
metadata = Dict{String,Any}(
    "ace_commit" => readchomp(`git -C $ace_root rev-parse HEAD`),
    "julia_version" => string(VERSION),
    "n81_grid_sha256" => bytes2hex(sha256(read(only(ARGS)))),
    "source_sha256" => Dict(f => bytes2hex(sha256(read(joinpath(ace_root, f))))
        for f in ("src/cubic_b_spline.jl", "src/chainrules.jl", "src/threads.jl")),
    "cases" => [4, 5, 6, 8, 81],
    "dtype" => "Float64", "rhs_columns" => 3,
)
ACE.set_threads(1)
for n in metadata["cases"]
    t = n == 81 ? log.(vec(readdlm(only(ARGS), Float64))) :
        collect(range(0.0, 1.0; length=n)).^1.35
    u = hcat(sin.(1.3 .* t), cos.(0.7 .* t) .+ 0.03 .* t, exp.(0.05 .* t))
    g = hcat(cos.(0.2 .* t), sin.(0.4 .* t) .+ 0.2, ones(n))
    q = collect(range(first(t), last(t); length=13))
    plan = CubicBSplinePlan(t, q)
    coeff = bspline_coefficients(plan, u)
    grad_t = ForwardDiff.gradient(x -> sum(bspline_coefficients(CubicBSplinePlan(x, q), u) .* g), t)
    grad_u = reshape(ForwardDiff.gradient(x -> sum(bspline_coefficients(plan, reshape(x, n, 3)) .* g), vec(u)), n, 3)
    fields = ("inputs" => hcat(t, u, g), "query" => q,
        "coefficients" => coeff, "factors" => plan.factorization.bands,
        "transpose" => ACE.solve_adjoint(plan.factorization, g),
        "values" => plan(u), "grad_sites" => grad_t, "grad_values" => grad_u)
    for (name, data) in fields
        open(joinpath(OUT, "julia_n$(n)_$(name).txt"), "w") do io
            println(io, "# ACE reference; N=$n; ", metadata["ace_commit"], "; ", name)
            writedlm(io, data, '\t')
        end
    end
end
open(io -> TOML.print(io, metadata; sorted=true), joinpath(OUT, "julia_metadata.toml"), "w")
println("Frozen Julia banded references: ", OUT)
