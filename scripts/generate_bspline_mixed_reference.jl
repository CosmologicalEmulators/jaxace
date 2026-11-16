# Freeze mixed precision from the exact merged Julia source, without overrides.
using AbstractCosmologicalEmulators, ForwardDiff, DelimitedFiles, SHA, TOML
length(ARGS) == 1 || error("usage: generator.jl /absolute/path/to/pinned/julia_source")
const SOURCE = abspath(only(ARGS))
const OUT = normpath(joinpath(@__DIR__, "..", "tests", "data", "bspline_mixed"))
ispath(OUT) && error("references already exist; refusing overwrite")
const EXPECTED = Dict(
    "src/cubic_b_spline.jl" => "824b1f806270ac87b4702f68f6f7f8f91f4ab0e2f07dc67e3c81727da25a55c5",
    "src/threads.jl" => "e0b5aeca150adaad398bc934d5e4dc3c6aae8ee9f1fde87b5947a6d5b005d3d3",
)
for (file, hash) in EXPECTED
    bytes2hex(sha256(read(joinpath(SOURCE,file)))) == hash || error("wrong source: $file")
end
module Reference
using LinearAlgebra
import AbstractCosmologicalEmulators
const Adapt = AbstractCosmologicalEmulators.Adapt
const FFTW = AbstractCosmologicalEmulators.FFTW
include(joinpath(Main.SOURCE,"src","threads.jl"))
include(joinpath(Main.SOURCE,"src","cubic_b_spline.jl"))
end
function freeze()
    mkpath(OUT)
    metadata = Dict{String,Any}(
        "ace_commit"=>"c5f9ca06d1cd00a6b224ea505bb36f612477004e",
        "source_sha256"=>EXPECTED, "julia_version"=>string(VERSION),
        "forwarddiff_version"=>string(pkgversion(ForwardDiff)),
        "generator_sha256"=>bytes2hex(sha256(read(@__FILE__))), "cases"=>Dict(),
    )
    for (T,U,Q) in ((Float32,Float64,Float64),(Float64,Float32,Float32),
                    (Float32,Float32,Float32),(Float64,Float64,Float64))
        name = "t$(8*sizeof(T))_u$(8*sizeof(U))_q$(8*sizeof(Q))"
        t = T.([0.,.1,.2,.35,.6,.75,.88,1.])
        u = U.([sin(1.3*Float64(x))+.1*cos(.3*j) for x in t,j in 1:3])
        q = Q.([.06,.17,.41,.77,.96])
        plan = Reference.CubicBSplinePlan(t,q)
        spline = Reference.CubicBSpline(u,t)
        gradient = ForwardDiff.gradient(v->sum(abs2,plan(reshape(v,size(u)))),vec(u))
        fields = Dict("sites"=>t,"values"=>u,"query"=>q,
                      "factor"=>plan.factorization.bands,
                      "coefficients"=>Reference.bspline_coefficients(plan,u),
                      "spline_coefficients"=>Reference.bspline_coefficients(spline),
                      "prediction"=>plan(u),"grad_values"=>reshape(gradient,size(u)))
        for (field,value) in fields
            all(isfinite,value) || error("nonfinite reference: $name $field")
            writedlm(joinpath(OUT,"$(name)_$(field).txt"),value)
        end
        metadata["cases"][name] = Dict("factor_dtype"=>string(eltype(plan.factorization.bands)),
                                      "coefficient_dtype"=>string(eltype(fields["coefficients"])))
        println(name," ",metadata["cases"][name])
    end
    open(io->TOML.print(io,metadata;sorted=true),joinpath(OUT,"metadata.toml"),"w")
end
freeze()
