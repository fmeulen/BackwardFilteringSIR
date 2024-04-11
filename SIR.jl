@__DIR__

using Random, StaticArrays, LinearAlgebra, StatsBase, Plots, ColorSchemes, Distributions, LaTeXStrings, Unzip, Printf

# Problem dimensions
N = 100
T = 500
size_neighbourhood = 1

include("FactoredFiltering.jl")
include("create_data.jl")
include("BoyenKollerFiltering.jl")
include("setup.jl")
include("mcmc.jl")


# Define the true initial state / root
Iinitial = 2
root = vcat(fill(_S_, N÷3-Iinitial), fill(_I_, Iinitial÷2), fill(_S_, N-N÷3), fill(_I_, Iinitial-Iinitial÷2))

δ = 0.0 # artificial par to become infected without infected neighbours
τ = 0.1 # discretisation time step of SIR model

# Parametric description of the entire forward model (I wish to pass δ and τ as well)
SIR(θ, δ, τ) = FactorisedMarkovChain(statespace, parents, dynamics(θ, δ, τ), root, (N, T))

# Instantiation with the true dynamics
θ = [1.2, 0.6, 0.03] # now in paper
θ = 5.0*[1.2, 0.1, 0.03] # i would not make lambda too large, basically makes P(infected given 1 or 2 I neighbours) = 0.999

G = SIR(θ, δ, τ)

# forward simulate and extract observations from it
#Nobs = 300
#Ztrue, Strue, obsparents = create_data(Arbitrary(), G, Nobs; seednr = 15)

Ztrue, Strue, obsparents = create_data(Structured(), G; seednr = 5, tinterval=2, iinterval=2)

plot(heatmap(Ztrue), heatmap(Strue))

# The emissions process / matrix. Many different options
#O = [1.0 0.0; 0.0 1.0 ; 1.0 0.0] # observe infected
#O = [.9 .1; 0.1 .9 ; .9 0.1] # observe infected
#O = Matrix(1.0*LinearAlgebra.I, 3, 3)
#O = [0.98 0.01 0.01; 0.01 0.98 0.01; 0.01 0.01 0.98] # observe with error
#O = [0.95 0.05; 0.95 0.05; 0.05 0.95] # observe either {S or I} or {R} with error
Id = Matrix(1.0*LinearAlgebra.I, 3, 3)
δobs = 0.5
O = (1.0-δobs) * Id + δobs/2 * (ones(3,3) - Id)  # observe with error


# Map each Observation variable index to corresponding emission process
obscpds = Dict((i,t) => O for (i,t) in keys(obsparents))

# Sample the emission process assigned to each observation variable
obsstates = Dict((i,t) => sample(weights(obscpds[(i,t)][Strue[i,t],:])) for (i,t) in keys(obsparents))
obs = (obsparents, obscpds, obsstates)

# Initialise inference
propagation = boyenkoller

# prior on the initial state of each individual
# Πroot1 = [0.9, 0.05, 0.05]
# Πroot2 = [1.0, 0.0, 0.0]
# Πroot =  merge(Dict(i => Πroot1 for i in 1:N÷2), Dict(i => Πroot2 for i in N÷2:N))
Πroot =  Dict(i => [0.98, 0.02, 0.00] for i in 1:N)

# Backward filter
ms, logh =  backwardfiltering(G, propagation, false, obs, Πroot, size_neighbourhood)

################
# Interesting to look at the h-transform, visualise this for individual `id`
id = 41 #40
pB = plot(vcat([ms[t].factoredhtransform[id] for t=2:T]'...), xlabel=L"$t$", ylabel=L"g_t",
            label=[L"\textbf{S}" L"\textbf{I}" L"\textbf{R}"], dpi=600,
            title="guiding vectors for individual $id")

# when do we observe individual id?
ℴ = values(obsparents) # (individal, time) combinations of observations
Iind = findall(first.(ℴ).==id)
tℴ = last.(ℴ)[Iind]  # these are the times
vline!(pB, tℴ, lwd=2, color="black", label="obs. time")
Strue[id, tℴ]  # these are the states at observation times


plot(pB)
savefig(pB, "htransform.png")

################ run mcmc ################

out = mcmc(G, ms, obs, Πroot;ITER=100, ρ=0.5 )

################ visualisation ################

# Plots
sz = (700,600)
pinit = heatmap(out.Sinit, xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="first iteration", size=sz)
ptrue = heatmap(Strue, xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="true", size=sz)
plast = heatmap(out.Slast, xlabel="",ylabel="",colorbar=false, yrotation=90, dps=600, title="last iteration", size=sz)
pavg = heatmap(out.Savg, xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="average", size=sz)

ii = 1:50
ptrue_zoomed = heatmap(Strue[:,ii],xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="true (zoomed)", size=sz)
pavg_zoomed = heatmap(out.Savg[:,ii], xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="average (zoomed)", size=sz)
pinit_zoomed = heatmap(out.Sinit[:,ii],xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="initial (zoomed)", size=sz)

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

pobs = heatmap(Yobs, xlabel="time", ylabel="individual", colorbar=false, color=observationpalette, yrotation=90, dps=600, title="observed", background_color_subplot=white)



lo = @layout [a b; c d; e f g]
pall_pobs = plot(pinit, plast, ptrue, pavg, pinit_zoomed,
            ptrue_zoomed, pavg_zoomed, layout=lo)#, xlabel="time", ylabel="individual")#, size=(800,1600))

lo2 = @layout [a;b]
pforward = plot(ptrue, pobs,  layout=lo2)

ploglik = plot(out.weights, label="", ylabel="loglikelihood", xlabel="MCMC update step", linewidth=2.0, size = (700,400))

savefig(pforward, "true_and_observed.png")
savefig(pall_pobs,  "true_and_outmcmc.png")
savefig(ploglik,  "trace_loglik.png")

plot(heatmap(Ztrue, title="Ztrue"), heatmap(out.Zinit, title="Z first iteration"), heatmap(out.Zlast, title="Z last iteration"))
