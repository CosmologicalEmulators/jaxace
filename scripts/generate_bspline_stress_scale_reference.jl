# Additional uncancelled collocation sensitivity, not a replacement for any fixture.
using AbstractCosmologicalEmulators, ForwardDiff, DelimitedFiles, SHA, TOML
const ACE = AbstractCosmologicalEmulators

function collocation(x)
    n = length(x)
    knots = vcat(fill(first(x), 4), x[3:end-2], fill(last(x), 4))
    basis = CubicBSplineBasis(; knot_vector=knots)
    bands = zeros(eltype(x), 7, n)
    for i in eachindex(x)
        row = basis_row(basis, x[i])
        for k in 1:4
            j = row.indices[k]
            bands[4+i-j, j] = row.values[k]
        end
    end
    return bands
end

function generate_scales()
    root = normpath(joinpath(@__DIR__, "..", "tests", "data", "bspline_banded", "stress_highprecision"))
    metadata = Dict{String,Any}(
        "precision_bits" => 256, "julia_version" => string(VERSION),
        "generator_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "original_reference_metadata_sha256" => bytes2hex(sha256(read(joinpath(root, "metadata.toml")))),
    )
    metadata_path = joinpath(root, "scale_metadata.toml")
    isfile(metadata_path) && error("scale references already exist; never overwrite")
    setprecision(BigFloat, 256) do
        for name in ("raw_k", "clustered")
            path = joinpath(root, "$(name)_collocation_scale.txt")
            isfile(path) && error("refusing to overwrite $path")
            data = readdlm(joinpath(root, "$(name)_inputs.txt"), Float64)
            t, u = BigFloat.(data[:, 1]), BigFloat.(data[:, 2:3])
            n = length(t)
            plan = CubicBSplinePlan(t, BigFloat[first(t), last(t)])
            c = bspline_coefficients(plan, u)
            adjoint = ACE.solve_adjoint(plan.factorization, 2 .* c)
            jacobian = ForwardDiff.jacobian(x -> vec(collocation(x)), t)
            absolute_contraction = zeros(BigFloat, 7, n)
            for j in 1:n, i in max(1, j-3):min(n, j+3)
                absolute_contraction[4+i-j, j] = sum(abs, adjoint[i, :] .* c[j, :])
            end
            scale = transpose(abs.(jacobian)) * vec(absolute_contraction)
            writedlm(path, scale)
            println("FROZEN collocation scale ", name)
        end
    end
    open(io -> TOML.print(io, metadata; sorted=true), metadata_path, "w")
end

generate_scales()
