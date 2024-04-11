@__DIR__
# try like https://www.freecodecamp.org/news/deep-learning-with-julia/

using Images, Flux
using Random, StaticArrays, LinearAlgebra, StatsBase, Plots, ColorSchemes, Distributions, LaTeXStrings, Unzip, Printf

# Problem dimensions
N = 20#100
T = 200#501

include("FactoredFiltering.jl")
include("create_data.jl")
include("BoyenKollerFiltering.jl")
include("setup.jl")
include("mcmc.jl")

# Define the true initial state / root
Iinitial = 2
root = vcat(fill(_S_, N÷3-Iinitial), fill(_I_, Iinitial÷2), fill(_S_, N-N÷3), fill(_I_, Iinitial-Iinitial÷2))

# Parametric description of the entire forward model
SIR(θ) = FactorisedMarkovChain(statespace, parents, dynamics(θ), root, (N, T))

# construct observation ColorPalette
defaultpalette = palette(cgrad(:default, categorical=true), 3)
white = RGBA{Float64}(255, 255, 255)
white = RGBA{Float64}(16, 59, 223, 0.12)
white = RGBA(52, 162, 231, 0.23)
observationcolors = vec(hcat(white, defaultpalette.colors.colors...))
observationpalette = ColorPalette(typeof(defaultpalette.colors)(observationcolors, "", ""))

# Instantiation with the true dynamics
θ = 5.0*[1.2, 0.1, 0.03]

G = SIR(θ)

# forward simulate and extract observations from it
Nobs = Int.(round(N*T*0.2;digits=0))
#O = [1.0 0.0; 0.0 1.0 ; 1.0 0.0] # observe infected
O = Matrix(1.0*LinearAlgebra.I, 3, 3)
obscpds = Dict((i,t) => O for (i,t) in keys(obsparents))
obsstates = Dict((i,t) => sample(weights(obscpds[(i,t)][Strue[i,t],:])) for (i,t) in keys(obsparents))





Ntraining = 2
for j in 1:Ntraining
    Ztrue, Strue, obsparents = create_data(Arbitrary(), G, Nobs; seednr = j)
    Yobs = zero(Strue)
    for ((i,t), state) in obsstates
        #Yobs[max(i-1,1):i, max(t-3,1):t] .= state
        Yobs[i, t] = state
    end

    pobs = heatmap(Yobs,  colorbar=false, color=observationpalette, yrotation=90,  background_color_subplot=white)
    ptrue = heatmap(Strue, colorbar=false)
    plot(pobs, ptrue)
end

############################ check if we can get it right #######################

obs = (obsparents, obscpds, obsstates)
# Initialise inference
propagation = boyenkoller

Πroot =  Dict(i => [1.0-2.0/N, 2.0/N, 0.00] for i in 1:N)  # apriori expect two infections at time zero

# Backward filter
ms, logh =  backwardfiltering(G, propagation, false, obs, Πroot)
out = mcmc(G, ms, obs, Πroot;ITER=1000, ρ=0.5, BIfactor=3)

sz = (700,600)
pinit = heatmap(out.Sinit, xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="first iteration", size=sz)
ptrue = heatmap(Strue, xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="true", size=sz)
plast = heatmap(out.Slast, xlabel="",ylabel="",colorbar=false, yrotation=90, dps=600, title="last iteration", size=sz)
pavg = heatmap(out.Savg, xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="average", size=sz)

ii = 1:50
ptrue_zoomed = heatmap(Strue[:,ii],xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="true (zoomed)", size=sz)
pavg_zoomed = heatmap(out.Savg[:,ii], xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="average (zoomed)", size=sz)
pinit_zoomed = heatmap(out.Sinit[:,ii],xlabel="",ylabel="", colorbar=false, yrotation=90, dps=600, title="first iter. (zoomed)", size=sz)

lo = @layout [a b; c d; e f g]
pall_pobs = plot(pinit, plast, ptrue, pavg, pinit_zoomed, 
            ptrue_zoomed, pavg_zoomed, layout=lo)#, xlab




savefig(pobs, "observed.png")
savefig(ptrue, "full.png")

img = load("full.png")
size(img)

plot(Gray.(img))

data = Float32.(channelview(img))

Npix = prod(size(img))


model = Chain(
    Flux.flatten,
    Dense(Npix=>Npix,relu),
    Dense(Npix=>Npix,sigmoid),
    #softmax
)

#############

using Plots
using Statistics#Auxiliary functions for generating our data
using Flux

function generate_real_data(n)
    x1 = rand(1,n) .- 0.5
    x2 = (x1 .* x1)*3 .+ randn(1,n)*0.1
    return vcat(x1,x2)
end

function generate_fake_data(n)
    θ  = 2*π*rand(1,n)
    r  = rand(1,n)/3
    x1 = @. r*cos(θ)
    x2 = @. r*sin(θ)+0.5
    return vcat(x1,x2)
end# Creating our data

train_size = 5000
real = generate_real_data(train_size)
fake = generate_fake_data(train_size)# Visualizing
scatter(real[1,1:500],real[2,1:500])
scatter!(fake[1,1:500],fake[2,1:500])


function NeuralNetwork()
    return Chain(
            Dense(2, 25,relu),
            Dense(25,1,x->σ.(x))
            )
end

# Organizing the data in batches
X    = hcat(real,fake)
Y    = vcat(ones(train_size),zeros(train_size))
data = Flux.Data.DataLoader((X, Y'), batchsize=100,shuffle=true);

# Defining our model, optimization algorithm and loss function
m    = NeuralNetwork()
opt = Descent(0.05)
loss(x, y) = sum(Flux.Losses.binarycrossentropy(m(x), y))

# Training Method 1
ps = Flux.params(m)
epochs = 20
for i in 1:epochs
    Flux.train!(loss, ps, data, opt)
end
println(mean(m(real)),mean(m(fake))) # Print model prediction# Visualizing the model predictions
scatter(real[1,1:100],real[2,1:100],zcolor=m(real)')
scatter!(fake[1,1:100],fake[2,1:100],zcolor=m(fake)',legend=false)