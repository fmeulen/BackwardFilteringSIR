
# Modelling the process dynamics
@enum State::UInt8 _S_=1 _I_=2 _R_=3

# Template transition model constructor
function ptemplate(i, parentstates::Tuple{Vararg{State}}, θ, δ, τ)
    λ, μ, ν = θ



    neighbours = setdiff(1:length(parentstates), i)
    N_i = sum(parentstates[neighbours] .== _I_)

    ψ(u) = exp(-τ*u)

    parentstates[i] == _S_ ? [(1.0-δ)*ψ(λ*N_i),  1.0-(1.0-δ)*ψ(λ*N_i),  0.      ] :
    parentstates[i] == _I_ ? [ 0.,               ψ(μ),                  1.0-ψ(μ)] :
    parentstates[i] == _R_ ? [ 1.0-ψ(ν),         0.,                    ψ(ν)    ] : println("Error")
end

#=
# Transition model for each individual i, returns p.dist given the parentsstates
p1(parentstates; θ)     = ptemplate(1, parentstates::NTuple{3, State}, θ, δ, τ)
p2(parentstates; θ)     = ptemplate(2, parentstates::NTuple{4, State}, θ, δ, τ)
pInner(parentstates; θ) = ptemplate(3, parentstates::NTuple{5, State}, θ, δ, τ)
pM(parentstates; θ)     = ptemplate(3, parentstates::NTuple{4, State}, θ, δ, τ)
pN(parentstates; θ)     = ptemplate(3, parentstates::NTuple{3, State}, θ, δ, τ)


# The general state-space
E = [_S_, _I_, _R_]

statespace = Dict(i => E for i in 1:N)
parents = Dict(i => intersect(i-2:i+2, 1:N) for i in 1:N)


# Map each node id to a 'type'
nodeToType = merge(Dict([1 => "1", 2 => "2", N-1 => "M", N => "N"]), Dict((c => "Inner" for c ∈ 3:N-2)))

# Map each 'type' to a transition model
typeToP = Dict(["1" => p1, "2" => p2, "Inner" => pInner, "M" => pM, "N" => pN])

# Map each 'type' to its input-space, required for transition model constuctor
typeToSupport = Dict(["1" => (E, E, E), "2" => (E, E, E, E), "Inner" => (E, E, E, E, E), "M" => (E, E, E, E), "N" => (E, E, E)])

# Parametric description of the problem dymamics
dynamics(θ) = cpds(nodeToType, typeToP, typeToSupport, θ)
=#


## april 2024 - generalise neighbourhood

## node To Type

#size_neighbourhood = 1

# The general state-space
E = [_S_, _I_, _R_]
statespace = Dict(i => E for i in 1:N)

parents = Dict(i => intersect(i-size_neighbourhood:i+size_neighbourhood, 1:N) for i in 1:N)

nodeToTypestart  = Dict(i => string(i) for i = 1:size_neighbourhood)
nodeToTypemiddle = Dict(c => "Inner" for c ∈ 1+size_neighbourhood:N-size_neighbourhood)
nodeToTypeend    = Dict(i => string(i) for i = N+1-size_neighbourhood:N)

nodeToType = merge(nodeToTypestart, nodeToTypemiddle, nodeToTypeend)

## type To P

function pLeft(parentstates::NTuple{s, State}, θ, δ, τ) where s
    ptemplate(s-size_neighbourhood, parentstates, θ, δ, τ)
end

function pInner(parentstates::NTuple{1+2*size_neighbourhood}, θ, δ, τ)
    ptemplate(1+size_neighbourhood, parentstates, θ, δ, τ)
end

function pRight(parentstates::NTuple{s, State}, θ, δ, τ) where s
    ptemplate(1+size_neighbourhood, parentstates, θ, δ, τ)
end

typeToPstart = Dict(string(i) => pLeft for i = 1:size_neighbourhood)
typeToPinner = Dict("Inner" => pInner)
typeToPend   = Dict(string(i) => pRight for i=N-size_neighbourhood+1:N)

typeToP = merge(typeToPstart, typeToPinner, typeToPend)


## type To Support

typeToSupportstart  = Dict([string(i) => ntuple(j -> E, i+size_neighbourhood) for i=1:size_neighbourhood])
typeToSupportmiddle = Dict(["Inner" => ntuple(j -> E, 1+2*size_neighbourhood)])
typeToSupportend    = Dict([string(i) => ntuple(j -> E, N-i+1+size_neighbourhood) for i = N+1-size_neighbourhood:N])

typeToSupport = merge(typeToSupportstart, typeToSupportmiddle, typeToSupportend)

## done

dynamics(θ, δ, τ) = cpds(nodeToType, typeToP, typeToSupport, θ, δ, τ)
