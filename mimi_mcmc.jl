
# mini mcmc programme
@__DIR__

using Random, StaticArrays, LinearAlgebra, StatsBase, Plots, ColorSchemes, Distributions, LaTeXStrings, Unzip, Printf

Random.seed!(110)

# Problem dimensions
N = 50
T = 100
size_neighbourhood = 1

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
Nobs = 50#00
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

# initialisation mcmc
U = rand(G.N, G.T)
S, w = 𝒢(U)
extracols = vcat( [0x01], [0x02], fill(0x03,G.N-2) )  # just to have the colouring consistent
heatmap(hcat(S, extracols), title="$w")

Ss= [S]
ws = [w]

B =100
δrw = 0.2 # propose uᵒ = u + Uniform(-δrw, δrw)
ind = 1:50  # only at times in "ind" we update the innovations
for _ in 1:B
    thesame = true
    global wᵒ
    nrattempts = 0
    while thesame  # sample new innovations until the picture changes
        Uᵒ = u_update(U, δrw, ind)
        Sᵒ, wᵒ = 𝒢(Uᵒ)
        thesame = (Sᵒ == S) 
    end
    nrattempts += 1
    @show nrattempts

    if log(rand()) < wᵒ - w
        U.= Uᵒ
        w = wᵒ
        S.= Sᵒ
        @show "accepted"
    end
    push!(Ss, copy(S))
    push!(ws, w)
end

sz = (700,700)
p0 = heatmap(Strue, title="true")
lo = @layout [a; b; c]
anim = @animate for i in 2:B
    p1 = heatmap(hcat(Ss[i-1], extracols), title="iteration $(i-1)")
    p2 = heatmap(hcat(Ss[i], extracols), title="iteration $i")
    plot(p1, p2, p0, layout=lo, size=sz)
end

# top panel: heatmap at iteration i-1, middle panel: heatmpa at iteration i
gif(anim, "test_animation.gif", fps=5)

