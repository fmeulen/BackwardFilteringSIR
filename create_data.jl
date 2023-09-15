# The true forward realisation and observations

abstract type  Scenario end
struct Arbitrary<: Scenario end
struct Structured<: Scenario end
struct AllObserved<: Scenario end

## specify the observations 

function create_data(::Arbitrary, G, Nindividuals; seednr = 5)
    Random.seed!(seednr)
    Ztrue = rand(Float64, G.N, G.T)
    Strue = samplefrom(G, Ztrue)

    obsparents = Dict((c[1],c[2])=>(c[1],c[2]) for c in sample(CartesianIndices((G.N ,G.T)), Nindividuals, replace=false))
    Ztrue, Strue, obsparents
end

function create_data(::Structured, G; seednr = 5, tinterval=50, iinterval=3)
    Random.seed!(seednr)
    Ztrue = rand(Float64, G.N, G.T)
    Strue = samplefrom(G, Ztrue)

    tobserved = tinterval+1:tinterval:T # Observation interval and observed time steps
    iobserved = 1:iinterval:N # Individual interval and observed individuals
    obsparents = Dict((i,t) => (i,t) for i=iobserved, t=tobserved)
    
    Ztrue, Strue, obsparents
end
