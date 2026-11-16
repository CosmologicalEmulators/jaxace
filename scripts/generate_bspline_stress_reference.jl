# Independent 256-bit ForwardDiff references on exactly promoted Float64 inputs.
# Existing baseline fixtures are untouched. Run with the ACE benchmark project.
using AbstractCosmologicalEmulators, ForwardDiff, DelimitedFiles, SHA, TOML

function generate_stress_reference()
    root = normpath(joinpath(@__DIR__, "..", "tests", "data", "bspline_banded", "stress_highprecision"))
    metadata_path = joinpath(root, "metadata.toml")
    isfile(metadata_path) && error("stress references already exist; never overwrite")
    ace_root = dirname(dirname(pathof(AbstractCosmologicalEmulators)))
    metadata = Dict{String,Any}(
        "precision_bits" => 256, "julia_version" => string(VERSION),
        "source_sha256" => bytes2hex(sha256(read(joinpath(ace_root, "src", "cubic_b_spline.jl")))),
        "generator_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "input_sha256" => Dict{String,String}(),
    )
    setprecision(BigFloat, 256) do
        for name in ("raw_k", "clustered")
            input = joinpath(root, "$(name)_inputs.txt")
            metadata["input_sha256"][name] = bytes2hex(sha256(read(input)))
            data = readdlm(input, Float64)
            t, u = BigFloat.(data[:, 1]), BigFloat.(data[:, 2:3])
            q = BigFloat[first(t), last(t)]
            coefficients = x -> bspline_coefficients(CubicBSplinePlan(x, q), u)
            c = coefficients(t)
            jacobian = ForwardDiff.jacobian(x -> vec(coefficients(x)), t)
            terms = 2 .* vec(c) .* jacobian
            gradient = vec(sum(terms; dims=1))
            uncancelled = vec(sum(abs, terms; dims=1))
            for (field, value) in (("coefficients", c), ("grad_sites", gradient), ("uncancelled", uncancelled))
                path = joinpath(root, "$(name)_$(field).txt")
                isfile(path) && error("refusing to overwrite $path")
                writedlm(path, value)
            end
            println("FROZEN ", name, " n=", length(t))
        end
    end
    open(io -> TOML.print(io, metadata; sorted=true), metadata_path, "w")
end

generate_stress_reference()
