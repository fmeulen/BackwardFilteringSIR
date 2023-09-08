@__DIR__

using Random, StaticArrays, LinearAlgebra, StatsBase, Plots, ColorSchemes, Distributions, LaTeXStrings, Unzip, Printf
include("FactoredFiltering.jl")
include("create_data.jl")
include("BoyenKollerFiltering.jl")


# Modelling the process dynamics
@enum State::UInt8 _S_=1 _I_=2 _R_=3

# Template transition model constructor
function ptemplate(i, parentstates::Tuple{Vararg{State}}; θ)
    λ, μ, ν = θ
    δ=0.001 
    τ=0.1
    
    neighbours = setdiff(1:length(parentstates), i)
    N_i = sum(parentstates[neighbours] .== _I_)

    ψ(u) = exp(-τ*u)
    #i==1 ? [0.95,  0.05,  0.0] :
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
N = 100
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

# Define the true initial state / root
Iinitial = 3
root = vcat(fill(_I_, Iinitial), fill(_S_, N-Iinitial))

statespace = Dict(i => E for i in 1:N)
parents = Dict(i => intersect(i-2:i+2, 1:N) for i in 1:N)

# Parametric description of the entire forward model
SIR(θ) = FactorisedMarkovChain(statespace, parents, dynamics(θ), root, dims)




# Instantiation with the true dynamics
θ = [1.2, 0.6, 0.03]
G = SIR(θ)

# forward simulate and extract observations from it
Ztrue, Strue, obsparents = create_data(Arbitrary(), G, N, T, 300; seednr = 5)
    


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
propagation = boyenkoller


# prior on the initial state of each individual
Πroot = [0.9, 0.05, 0.05]

# prior on the initial state of each individual
Πroot1 = [0.9, 0.05, 0.05]
Πroot2 = [1.0, 0.0, 0.0]
Πroot =  merge(Dict(i => Πroot1 for i in 1:N÷2), Dict(i => Πroot2 for i in N÷2:N))


# Backward filter
ms, logh =  backwardfiltering(G, propagation, false, obs, Πroot)

################


# Interesting to look at the h-transform
id = 20
ss = sum(vcat([ms[t].factoredhtransform[id] for t=2:T]'...), dims=2)
#plot(ss) # is now 1 everywhere


plot(vcat([ms[t].factoredhtransform[id] for t=2:T]'...)./ss,
        xlabel=L"$t$", ylabel=L"\phi_i", label=[L"\textbf{S}" L"\textbf{I}" L"\textbf{R}"], dpi=600, title="approximate h-transform for individual $id")
#@show  ms[501].factoredhtransform[id]

# above (with normalisation) and below (without normalisation) are now seemingly identical, as expected

plot(vcat([ms[t].factoredhtransform[id] for t=2:T]'...), xlabel=L"$t$", ylabel=L"\phi_i", label=[L"\textbf{S}" L"\textbf{I}" L"\textbf{R}"], dpi=600, title="approximate h-transform for individual $id")


# Update Z for each segment of 50 time steps individually






function mcmc(G, ms, obs, logh; ITER=100, BIfactor=5, ρ=0.99, tinterval=10)
    BI = ITER÷BIfactor
    # takes blocks of size tinterval
    blocks = (G.T-1)÷tinterval

    # Initialise the first guided sample
    Zinit = rand(Float64, (G.N, G.T))
    Sinit, winit = forwardguiding(G, ms, obs, Zinit, Πroot)

    # Initialise MCMC parameters
    Z = copy(Zinit); S = copy(Sinit); w = copy(winit);
    qZ = quantile.(Normal(), Z)

    Savg = zeros(G.N, G.T)
    ws = [w]

    ACCZ = 0

    Ss = [S]
#    Zs = [(Z[22,11], Z[5,4])]  # just some Zs to monitor mixing
    for i = 1:ITER
        # Z step only
        for k = 1:blocks
            qZ′ = copy(qZ)
            ind = (k-1)*tinterval+1:k*tinterval+1
            qW = randn(Float64, (N, length(ind)))
            qZ′[:,ind] = ρ*qZ′[:,ind] + √(1 - ρ^2)*qW
            Z′ = cdf.(Normal(), qZ′)
            S′, w′ = forwardguiding(G, ms, obs, Z′, logh)

            A = S′ == S # check if prev image S is identical to new image S′

            if log(rand()) < w′ - w
                qZ = qZ′
                Z = Z′
                S, w = S′, w′
                ACCZ += 1
            end
            

            if (i % 5 == 0)
                @printf("iteration: %d | Z rate: %.4f | logweight: %.4e | assert: %d\n", i, ACCZ/((i-1)*blocks + (k-1) + 1), w, A)
            end
            push!(ws, w)
        end
       
#       push!(Zs,  (Z[22,11], Z[5,4]))
        if i > BI  Savg += S end 
        if (i % 500 == 0)    push!(Ss, S)          end
    end

    (Sinit=Sinit, Slast=S, Siterates=Ss, Savg=Savg, weights=ws)
end


out = mcmc(G, ms, obs, logh;ITER=500)



# Plots
sz = (500,600)
pinit = heatmap(out.Sinit, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, yrotation=90, dps=600, title="first iteration", size=sz)
ptrue = heatmap(Strue, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, yrotation=90, dps=600, title="true", size=sz)
plast = heatmap(out.Slast, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, yrotation=90, dps=600, title="last iteration", size=sz)
pavg = heatmap(out.Savg, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, yrotation=90, dps=600, title="average", size=sz)

# construct observation ColorPalette
defaultpalette = palette(cgrad(:default, categorical=true), 3)
white = RGBA{Float64}(255, 255, 255)
white = RGBA{Float64}(16, 59, 223, 0.12)
white = RGBA(52, 162, 231, 0.23)

observationcolors = vec(hcat(white, defaultpalette.colors.colors...))
observationpalette = ColorPalette(typeof(defaultpalette.colors)(observationcolors, "", ""))

# width of observations increased for clarity
Yobs = zero(Strue)
for ((i,t), state) in obsstates
    Yobs[max(i-1,1):i, max(t-3,1):t] .= state
end

pobs = heatmap(Yobs, xlabel=L"$t$", ylabel=L"$i$", colorbar=false, 
color=observationpalette, yrotation=90, dps=600, title="observed", background_color_subplot=white)


lo = @layout [a b; c d]
pall_pobs = plot(pinit, plast, ptrue, pavg, layout=lo)#, size=(800,1600))

lo2 = @layout [a;b]
pforward = plot(pobs, ptrue, layout=lo2)

ploglik = plot(out.weights, label="", ylabel="loglikelihood", xlabel="MCMC update step")


savefig(pforward, "true_and_observed.png")
savefig(pall_pobs,  "true_and_outmcmc.png")
savefig(ploglik,  "trace_loglik.png")