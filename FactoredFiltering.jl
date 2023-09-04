abstract type AbstractBayesNet end

struct Message{T}
    factoredhtransform::Dict{Int, Vector{T}}
    approximatepullback::Dict{Int, Vector{T}}
end

struct FactorisedMarkovChain{T} <: AbstractBayesNet
    statespace::Dict{Int, <: Vector{<: Enum}}
    parents::Dict{Int, UnitRange{Int}}
    cpd::Dict{Int, Dict{Vector{UInt8}, Vector{T}}}
    cpdmatrix::Dict{Int, SArray{S, T, 2, L} where {S <: Tuple, L}}
    root::Vector{UInt8}
    N::Int
    T::Int
end

function FactorisedMarkovChain(statespace, parents, (cpd, cpdmatrix), root, (N, T))
    FactorisedMarkovChain(statespace, parents, cpd, cpdmatrix, UInt8.(root), N, T)
end

discretesample(p::Vector, z) = z < p[1] ? UInt8(1) : z < p[1] + p[2] ? UInt8(2) : UInt8(3)

function samplefrom(FMC::FactorisedMarkovChain, Z::Matrix)
    samples = Matrix{UInt8}(undef, FMC.N, FMC.T)
    samples[:,1] = FMC.root
    for t in 2:FMC.T
        for i in 1:FMC.N
            p = FMC.cpd[i][@view samples[FMC.parents[i],t-1]]
            samples[i,t] = discretesample(p, Z[i,t])
        end
    end
    samples
end

function inplacemultiplication!(a::AbstractVector, b::AbstractVector)
    for i in eachindex(a)
        @inbounds a[i]=a[i]*b[i]
    end
end

function backwardfiltering(FMC::FactorisedMarkovChain{T}, kernel::Function, approximations, observations, Πroot) where T
    # we first initialise the (faactored) htransforms for all time steps

    htransforms = Dict([t => Dict([i => Vector{T}(ones(length(E))) for i in 1:FMC.N]) for t in 1:FMC.T])
    messages = Dict{Int, Message{T}}()

    obsparents, obscpds, obsstate = observations

    # then we write the observations into the correct htransforms
    # this seems a bit hack-y now but made sense at the time (pullback from individual nodes, obs, etc..)

    for ((i, t), obs) in pairs(obsstate)
        chtransform = obscpds[(i,t)][:,obs]
        inplacemultiplication!(htransforms[t][i], chtransform)
    end

    # Normalise guiding term at time step T

    for i in 1:FMC.N
        htransforms[FMC.T][i] = htransforms[FMC.T][i] / sum(htransforms[FMC.T][i])
    end


    for t in FMC.T:-1:2
        # first computea all the pullback factors. these are passed to a kernel function which produces
        # some projection of the pullback onto the space of marginaals. in this case we use a junction tree algo / boyen koller / fully factorised EP

        factoredpullback = Dict([i => Vector(FMC.cpdmatrix[i]*htransforms[t][i]) for i in 1:FMC.N])

        # this 'approximations' variable was originally the 'reference measures' with which we perform backward marginalisation
        # now it is (ab)used to pass the observation-pullbacks to the junction tree algo -> very hack-y...
        # just always set approxiations to false

        if approximations == false
            approximatepullback = kernel(factoredpullback, htransforms[t-1], FMC.N, t, 2)
        else
            approximatepullback = kernel(factoredpullback, approximations[t-1], FMC.N, t, 2)
        end

        # note: the kernel function handles normalisation and multiplication with observation term

        for i in 1:FMC.N
            #inplacemultiplication!(htransforms[t-1][i], approximatepullback[i])
            htransforms[t-1][i] = approximatepullback[i]
        end

        messages[t] = Message(htransforms[t], approximatepullback)
    end
        
    for (key, value) in htransforms[2]
            a = htransforms[2][key] .* Πroot        
            htransforms[1][key] = a/sum(a)
    end
     
    messages[1] = Message(htransforms[1] , htransforms[2])

    logh = (x0) -> sum(log(htransforms[1][i][x0[i]]) for i=1:FMC.N) # should not be FMC.root but x0
#    logh = sum(log(htransforms[1][i][FMC.root[i]]) for i=1:FMC.N) # should not be FMC.root but x0
    messages, logh
end

function cpds(nodeToType, typeToP, typeToSupport, θ::Vector{T}) where T
    typeToCpd = Dict{String, Dict{Vector{UInt8}, Vector{T}}}()
    typeToK = Dict{String, SArray{S, T, 2, L} where {S <: Tuple, L}}()

    nodeToCpd = Dict{Int, Dict{Vector{UInt8}, Vector{T}}}()
    nodeToK = Dict{Int, SArray{S, T, 2, L} where {S <: Tuple, L}}()

    for (type, p) in typeToP
        # should build the dict and matrix in one loop // make this a view
        tempcpd = Dict([UInt8.(collect(parentsstate)) => p(parentsstate; θ=θ) for parentsstate in Iterators.product(typeToSupport[type]...)])
        typeToCpd[type] = tempcpd
        typeToK[type] = SMatrix{length(E)^length(typeToSupport[type]), length(E), T}(reduce(hcat, [tempcpd[UInt8.(collect(parentsstate))] for parentsstate in Iterators.product(typeToSupport[type]...)])')
    end
    # re: memory save: assigning nodetoK keys to typetoK values does NOT copy K, only reference
    for (node, type) in nodeToType                                               # potential memory save
        nodeToCpd[node] = typeToCpd[type]                                        # nodeToCpd(node) = typeToCpd[nodeToType[node]]
        nodeToK[node] = typeToK[type]                                            # nodeToK(node) = typeToK[nodeToType[node]]
    end

    nodeToCpd, nodeToK
end

function cpds(nodeToType, typeToP, typeToSupport, θ)
    cpds(nodeToType, typeToP, typeToSupport, [θ])
end

function forwardguiding(FMC::FactorisedMarkovChain{T}, messages::Dict{Int, Message{T}}, observations, Z::Matrix, logh) where T
    samples = Matrix{UInt8}(undef, FMC.N, FMC.T)
    logweight = 0.
    samples[:,1] = FMC.root

    # for i in 1:FMC.N
    #     inplacemultiplication!(p, messages[1].factoredhtransform[i])
    #     samples[i,t] = discretesample(p, sum(p)*Z[i,t])

    #     logweight += log(weight)
    # end 

    obsparents, obscpds, obsstates = observations

    
    for t in 2:FMC.T
        #message = messages[t] faster ?
        for i in 1:FMC.N
            parentsamples = @view samples[FMC.parents[i],t-1]
            p = copy(FMC.cpd[i][parentsamples])  # forward prob for individual i at at time t, given state of parents

            # remember/note that when we SAMPLE the states in time step t
            # that we compute the WEIGHT for the preceding time step

            weight = dot(p, messages[t].factoredhtransform[i]) / messages[t].approximatepullback[i][samples[i,t-1]]

            if haskey(obsstates, (i,t-1))
                likelihood = obscpds[(i,t-1)][:,obsstates[(i,t-1)]] # get the correct column vector from the markov kernel
                contribution = likelihood[samples[i,t-1]]  # get the correct value from the likelihood function
                weight *= contribution                     # of course if we observe without error the contribution == 1
            end

            inplacemultiplication!(p, messages[t].factoredhtransform[i])
            samples[i,t] = discretesample(p, sum(p)*Z[i,t])

            logweight += log(weight)
        end
    end

    # compute the weight for time step T, is not done in the loop because we dont sample T+1

    weightT = 1.
    for i = 1:FMC.N
        if haskey(obsstates, (i,FMC.T))
            likelihood = obscpds[(i,FMC.T)][:,obsstates[(i,FMC.T)]]
            contribution = likelihood[samples[i,FMC.T]]
            weightT *= contribution
        end
    end
    logweight += log(weightT)


    samples, logh + logweight
end
