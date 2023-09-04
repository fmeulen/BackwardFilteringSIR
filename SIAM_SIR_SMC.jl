@__DIR__

using Random, StaticArrays, LinearAlgebra, StatsBase, Plots, ColorSchemes, Distributions, LaTeXStrings, Unzip, Printf
include("FactoredFiltering.jl")

# Modelling the process dynamics
@enum State::UInt8 _S_=1 _I_=2 _R_=3

# Template transition model constructor
function ptemplate(i, parentstates::Tuple{Vararg{State}}; θ)
    λ, μ, ν = θ
    δ = 0.01
    #δ = 0.00
    τ = 0.1 * 1

    neighbours = setdiff(1:length(parentstates), i)
    N_i = sum(parentstates[neighbours] .== _I_)

    ψ(u) = exp(-τ*u)

    parentstates[i] == _S_ ? [(1.0-δ)*ψ(λ*N_i),  1.0-(1.0-δ)*ψ(λ*N_i),  0.      ] :
    parentstates[i] == _I_ ? [ 0.,               ψ(μ),                  1.0-ψ(μ)] :
    parentstates[i] == _R_ ? [ 1.0-ψ(ν),         0.,                    ψ(ν)    ] : println("Error")
end

# Transition model for each individual i, returns p.dist given the parentsstates
p1(parentstates; θ)     = ptemplate(1, parentstates::NTuple{3, State}; θ)
p2(parentstates; θ)     = ptemplate(2, parentstates::NTuple{4, State}; θ)
pInner(parentstates; θ) = ptemplate(3, parentstates::NTuple{5, State}; θ)
pM(parentstates; θ)     = ptemplate(3, parentstates::NTuple{4, State}; θ)
pN(parentstates; θ)     = ptemplate(3, parentstates::NTuple{3, State}; θ)

# Problem dimensions
N = 20#100
T = 501
dims = (N, T)

# The general state-space
E = [_S_, _I_, _R_]

# Map each node id to a 'type'
nodeToType = merge(Dict([1 => "1", 2 => "2", N-1 => "M", N => "N"]), Dict((c => "Inner" for c ∈ 3:N-2)))

# Map each 'type' to a transition model
typeToP = Dict(["1" => p1, "2" => p2, "Inner" => pInner, "M" => pM, "N" => pN])

# Map each 'type' to its input-space, required for transition model constuctor
typeToSupport = Dict(["1" => (E, E, E), "2" => (E, E, E, E), "Inner" => (E, E, E, E, E), "M" => (E, E, E, E), "N" => (E, E, E)])

# Parametric description of the problem dymamics
dynamics(θ) = cpds(nodeToType, typeToP, typeToSupport, θ)

# Define the initial state / root
Iinitial = 7
statespace = Dict(i => E for i in 1:N)
parents = Dict(i => intersect(i-2:i+2, 1:N) for i in 1:N)
root = vcat(fill(_I_, Iinitial), fill(_S_, N-Iinitial))

#Iinitial = 5
#root = vcat(fill(_I_, Iinitial), fill(_S_, N-2*Iinitial), fill(_I_, Iinitial))

# Parametric description of the entire forward model
SIR(θ) = FactorisedMarkovChain(statespace, parents, dynamics(θ), root, dims)

# Instantiation with the true dynamics
θ = [2.5, 0.6, 0.1]
G = SIR(θ)

# The true forward realisation
Random.seed!(4)
Ztrue = rand(Float64, dims)
Strue = samplefrom(G, Ztrue)

# Observation interval and observed time steps
tinterval = 50
tobserved = tinterval+1:tinterval:T

# Individual interval and observed individuals
iinterval = 1
iobserved = 1:iinterval:N

# Map Y observation indices to X realisation indices
# In this case the observation 'parents' are trivial, as we simply observe singular individuals
obsparents = Dict((i,t) => (i,t) for i=iobserved, t=tobserved)

# The emissions process / matrix. Many different options
O = Matrix(1.0*LinearAlgebra.I, 3, 3)
#O = [0.98 0.01 0.01; 0.01 0.98 0.01; 0.01 0.01 0.98] # observe with error
#O = [0.95 0.05; 0.95 0.05; 0.05 0.95] # observe either {S or I} or {R} with error

# Map each Observation variable index to corresponding emission process
obscpds = Dict((i,t) => O for (i,t) in keys(obsparents))

# Sample the emission process assigned to each observation variable
obsstates = Dict((i,t) => sample(weights(obscpds[(i,t)][Strue[i,t],:])) for (i,t) in keys(obsparents))
obs = (obsparents, obscpds, obsstates)

# Initialise inference
include("BoyenKollerFiltering.jl")
propagation = boyenkoller
#proof()

# Backward filtering
ms, logh = backwardfiltering(G, propagation, false, obs)

# Initialise the first guided sample
Zinit = rand(Float64, dims)
Sinit, winit = forwardguiding(G, ms, obs, Zinit, logh)

plot(heatmap(Sinit, title="initial"), heatmap(Strue, title="true"))








################


# Interesting to look at the h-transform
id = 20
ss = sum(vcat([ms[t].factoredhtransform[id] for t=2:T]'...), dims=2)
#plot(ss) # is now 1 everywhere
p_htransform = plot(vcat([ms[t].factoredhtransform[id] for t=2:T]'...)./ss,
        xlabel=L"$t$", ylabel=L"\phi_i", label=[L"\textbf{S}" L"\textbf{I}" L"\textbf{R}"], dpi=600, title="approximate h-transform for individual $id")
savefig(p_htransform, "htransform.png")
#@show  ms[501].factoredhtransform[id]




function move((Z, S, w, qZ), NR_MOVE_STEPS, ρ, N, tinterval, G, ms, obs, logh)
    # Update Z for each segment of 50 time steps individually
    blocks = (T-1)÷tinterval

    ACCZ = 0
    for i = 1:NR_MOVE_STEPS
        # Z step only
        for k = 1:blocks
            qW = randn(Float64, (N, tinterval))

            qZ′ = copy(qZ)
            qZ′[:,(k-1)*tinterval+2:k*tinterval+1] = ρ*qZ′[:,(k-1)*tinterval+2:k*tinterval+1] + √(1 - ρ^2)*qW

            Z′ = cdf.(Normal(), qZ′)
            S′, w′ = forwardguiding(G, ms, obs, Z′, logh)
            if log(rand()) < w′ - w
                qZ = qZ′
                Z = Z′
                S, w = S′, w′
                ACCZ += 1
            end
        end
        avgACCZ = 100.0*round(ACCZ/(blocks*ITER); digits=2)
   #     println("acceptance percentage for move: $avgACCZ %")
     end
     (Z=Z, S=S, w=w, qZ=qZ)
end

function inititalise_particle(G, ms, obs, dims, logh)
    Z = rand(Float64, dims)
    S, w = forwardguiding(G, ms, obs, Z, logh)
    qZ  = quantile.(Normal(), Z)
    (Z=Z, S=S, w=w, qZ=qZ)
end



function smc(ρ, NR_SMC_STEPS, NUMPARTICLES, NR_MOVE_STEPS, G, ms, obs, dims, logh, tinterval)  
    # initialise particles 
    particles = [inititalise_particle(G, ms, obs, dims, logh) for _ in 1:NUMPARTICLES]
    
    ess_threshold = round(NUMPARTICLES/2; digits=0)    
    
    for j in 1:NR_SMC_STEPS
        println(j,"\n")
        # resample step
        log_weights =  getindex.(particles, :w)
        indices = resamp(log_weights, ess_threshold) 
        println(indices)
        particles = [particles[k] for k in indices]
        # move step (pCN)
        particles = map(x -> move(particles[x], NR_MOVE_STEPS, ρ, N, tinterval, G, ms, obs, logh), 1:NUMPARTICLES)
    end
    particles
end



NR_SMC_STEPS = 50
NUMPARTICLES = 10
NR_MOVE_STEPS = 5
ρ = 0.999
particles = smc(ρ, NR_SMC_STEPS, NUMPARTICLES, NR_MOVE_STEPS, G, ms, obs, dims, logh, tinterval)    




# visualisation
sz = (500,600)

@show getindex.(particles, :w)
@show unique(getindex.(particles, :w))    

kmax = argmax(getindex.(particles, :w))
wroundmax = round(particles[kmax].w;digits=1)

kmin = argmin(getindex.(particles, :w))
wroundmin = round(particles[kmin].w;digits=1)

Savg = heatmap(mean(getindex.(particles, :S)),title="avg", size = sz, colorbar=false, yrotation=90, dps=600)
psmc = plot( heatmap(particles[kmax].S,title = "$wroundmax", size = sz, colorbar=false, yrotation=90, dps=600),
            heatmap(Strue, title="true", size = sz, colorbar=false, yrotation=90, dps=600),
            heatmap(particles[kmin].S,title = "$wroundmin", size = sz, colorbar=false, yrotation=90, dps=600),
            Savg)
savefig(psmc, "recovery_smc.png")

# Plots
# sz = (500,600)
# pinit = heatmap(Sinit, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, yrotation=90, dps=600, title="initial", size=sz)
# ptrue = heatmap(Strue, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, yrotation=90, dps=600, title="true", size=sz)
# plast = heatmap(S, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, yrotation=90, dps=600, title="last iteration", size=sz)
# pavg = heatmap(Savg, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, yrotation=90, dps=600, title="average", size=sz)

# pall = plot(pinit, ptrue, plast, pavg)
# pall
# savefig(pall, "recovery.png")

# @show w
# pinit

# construct observation ColorPalette
defaultpalette = palette(cgrad(:default, categorical=true), 3)
white = RGBA{Float64}(255, 255, 255)

observationcolors = vec(hcat(white, defaultpalette.colors.colors...))
observationpalette = ColorPalette(typeof(defaultpalette.colors)(observationcolors, "", ""))

# width of observations increased for clarity
Yobs = zero(Strue)
for ((i,t), state) in obsstates
    Yobs[i,t-3:t] .= state
end

pobs = heatmap(Yobs, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, color=observationpalette, yrotation=90, dps=600)
savefig(pobs, "obs.png")