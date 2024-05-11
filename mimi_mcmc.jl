
# mini mcmc programme
@__DIR__

using Random, StaticArrays, LinearAlgebra, StatsBase, Plots, ColorSchemes, Distributions, LaTeXStrings, Unzip, Printf

#Random.seed!(110)  # interesting case
Random.seed!(10)

# Problem dimensions
N = 50
T = 100
size_neighbourhood = 2

include("FactoredFiltering.jl")
include("create_data.jl")
include("BoyenKollerFiltering.jl")
include("setup.jl")
include("mcmc.jl")


# Define the true initial state / root
 Iinitial = 2
 root = vcat(fill(_S_, N÷3-Iinitial), fill(_I_, Iinitial÷2), fill(_S_, N-N÷3), fill(_I_, Iinitial-Iinitial÷2))

# have one infected in the middle
#root = vcat(fill(_S_, N÷2), [_I_], fill(_S_, N-N÷2-1))


δ = 0.0 # artificial par to become infected without infected neighbours
τ = 0.1 # discretisation time step of SIR model

# Parametric description of the entire forward model (I wish to pass δ and τ as well)
SIR(θ, δ, τ) = FactorisedMarkovChain(statespace, parents, dynamics(θ, δ, τ), root, (N, T))

# Instantiation with the true dynamics
θ = [1.2, 0.6, 0.03] # now in paper
θ = 5.0*[1.2, 0.1, 0.03] # i would not make lambda too large, basically makes P(infected given 1 or 2 I neighbours) = 0.999

G = SIR(θ, δ, τ)

# forward simulate and extract observations from it
Nobs = 30# 50#00
Ztrue, Strue, obsparents = create_data(Arbitrary(), G, Nobs; seednr = 15)

# observe only in the middle
#obsparents = Dict([(N÷2 + 1, T÷2) => (N÷2 + 1, T÷2),(N÷2 + 1, T÷2 + 1) => (N÷2 + 1, T÷2 + 1)])
#obsparents = Dict([(n, T÷2) => (n, T÷2) for n in 1:N])

#Ztrue, Strue, obsparents = create_data(Structured(), G; seednr = 5, tinterval=2, iinterval=2)

plot(heatmap(Ztrue), heatmap(Strue))

# The emissions process / matrix. Many different options
Id = Matrix(1.0*LinearAlgebra.I, 3, 3)
δobs = 0.0001
O = (1.0-δobs) * Id + δobs/2 * (ones(3,3) - Id)  # observe with error

# Map each Observation variable index to corresponding emission process
obscpds = Dict((i,t) => O for (i,t) in keys(obsparents))

# no error on middle observation
#obscpds = Dict((i,t) => Matrix(1.0*LinearAlgebra.I, 3, 3) for (i,t) in keys(obsparents))


# Sample the emission process assigned to each observation variable
 obsstates = Dict((i,t) => sample(weights(obscpds[(i,t)][Strue[i,t],:])) for (i,t) in keys(obsparents))

obs = (obsparents, obscpds, obsstates)

# Initialise inference
propagation = boyenkoller

# prior on the initial state of each individual
Πroot =  Dict(i => [0.98, 0.02, 0.00] for i in 1:N) # so most are susceptbile, 2% chance of being infected
#Πroot[N÷2 + 1] = [0.0, 1.0, 0.0]


# Backward filter
ms, logh =  backwardfiltering(G, propagation, false, obs, Πroot, size_neighbourhood)
𝒢 = forwardguiding(G, ms, obs, Πroot)


partitions = partition_into_blocks_close(G.N, 10)

# initialisation mcmc
U = rand(G.N, G.T)
S, w = 𝒢(U)
extracols = vcat( [0x01], [0x02], fill(0x03,G.N-2) )  # just to have the colouring consistent
heatmap(hcat(S, extracols), title="$w")

Ss= [S]
ws = [w]
ids = [0]

B = 1000
δrw = 0.2 # propose uᵒ = u + Uniform(-δrw, δrw)
#ind = 1:5  # only at times in "ind" we update the innovations
for ind in partitions[1]
    for i in 1:B
        global wᵒ
        Uᵒ = u_update(U, δrw, ind)
        Sᵒ, wᵒ = 𝒢(Uᵒ)

        if log(rand()) < wᵒ - w && (Sᵒ !== S)
            U.= Uᵒ
            w = wᵒ
            S.= Sᵒ
            @show "accepted"
            # only save accepted
            push!(Ss, copy(S))
            push!(ws, w)
            push!(ids, i)
        end
    mod(i,10)==0 && print(i)  
    end
end

sz = (700,700)
p0 = heatmap(Strue, title="true")
lo = @layout [a; c]
anim = @animate for i in 1:length(ids)
    #k1, k2 = ids[i-1], ids[i]
    k2 = ids[i]
    #p1 = heatmap(hcat(Ss[i-1], extracols), title="iteration $k1")
    p2 = heatmap(hcat(Ss[i], extracols), title="iteration $k2")
    plot(p2, p0, layout=lo, size=sz)
end

# top panel: heatmap at iteration i-1, middle panel: heatmpa at iteration i
gif(anim, "test_animation.gif", fps=1)


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


pobs = heatmap(Yobs, xlabel="time", ylabel="individual", colorbar=true, color=observationpalette, yrotation=90, dps=600, title="observed", background_color_subplot=white)


lo = @layout [a; b; c]

plot(heatmap(hcat(Ss[end], extracols), title="iteration $(ids[end])"), p0, pobs, layout=lo, size=sz)