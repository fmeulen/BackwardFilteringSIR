
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
