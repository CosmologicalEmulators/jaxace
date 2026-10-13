# Independent Julia reference for the frozen JAX scalar interpolation model.
# Run with an environment containing OrdinaryDiffEqTsit5 and ForwardDiff,
# e.g. the existing ACE benchmark project. Never called by the test suite.
using OrdinaryDiffEqTsit5, ForwardDiff, Printf, SHA
const data = joinpath(@__DIR__, "data")
const output = joinpath(data, "scalar_growth_julia_reference.txt")
isfile(output) && error("Reference already exists; do not overwrite it")
const model = joinpath(data, "scalar_growth_polynomials.txt")
rows = [parse.(Float64,split(s)) for s in eachline(model) if !startswith(s,"#")]
const coefficients = permutedims(reduce(hcat,rows))
const sites = coefficients[:,1]
function F(y)
    i = clamp(searchsortedlast(sites,y),1,length(sites)-1)
    t,u,b,c,d = coefficients[i,:]
    w = y-t
    return ((d*w+c)*w+b)*w+u
end
const gamma = 2.469e-5/.67^2
const pref = 15/pi^4*((4/11)^(1/3)*(3.044/3)^(1/4))^4*gamma
rho(a,m) = pref/a^4*F(m*a/(8.617342e-5*.71611*2.7255))
E(a,m) = sqrt(gamma/a^4+.3/a^3+(1-gamma-.3-rho(1.,m))+rho(a,m))
function growth(m,matter)
    ai=1/139
    function rhs!(du,u,m,t)
        a=exp(t); e=E(a,m); A=a*a*e
        src=(.3/a^3+(matter ? rho(a,m)-rho(a,zero(m)) : zero(m)))/e^2
        du[1]=u[2]/A
        du[2]=1.5*A*src*u[1]
    end
    u0=[oftype(m,ai),ai^3*E(ai,m)]
    sol=solve(ODEProblem(rhs!,u0,(log(ai),log(1.01)),m),Tsit5();
              reltol=1e-12,abstol=1e-14,saveat=[log(.5)])
    return sol.u[1][1]
end
open(output,"w") do io
    println(io,"# Julia ",VERSION," Tsit5 + ForwardDiff; rtol=1e-12 atol=1e-14")
    println(io,"# Frozen JAX scalar polynomial model sha256 ",bytes2hex(sha256(read(model))))
    println(io,"# Same legacy physics/interpolant, not native Julia's different tables; no physical accuracy claim for scalar extrapolation")
    println(io,"# h=.67 Omega_cb=.3 w0=-1 wa=0 Omega_k=0 Neff=3.044 z=1; D(ai)=Dprime(ai)=ai=1/139")
    println(io,"# species_m mass_eV D_raw derivative_mass")
    for matter in (false,true),m in (.06,.3,.75)
        f(v)=growth(v,matter)
        @printf(io,"%d %.2f %.16e %.16e\n",matter,m,f(m),ForwardDiff.derivative(f,m))
    end
end
println("Wrote ",output)
