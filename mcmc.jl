f(x) = (1. + sin(x))/2.0
#f(x) = cdf(Normal(),x)


function pcn(Z, ρ, ind)
    Zᵒ = copy(Z)
    N = size(Z)[1]
    W = randn(Float64, (N, length(ind)))
    Zᵒ[:,ind] = ρ*Z[:,ind] + √(1 - ρ^2)*W
    Uᵒ = f.(Zᵒ) #cdf.(Normal(), Zᵒ)
    Uᵒ, Zᵒ
end

function u_update(U, δ, ind)
    Uᵒ = copy(U)
    N = size(U)[1]
    dims = (N, length(ind))
    Uᵒ[:,ind] = mod.(U[:,ind] + rand(Uniform(-δ,δ), dims),1)
    Uᵒ
end

function mcmc(G, ms, obs, Πroot; ITER=100, BIfactor=4, ρ=0.99, NUMBLOCKS=10)
    BI = ITER÷BIfactor
    partitions = partition_into_blocks_close(G.T, NUMBLOCKS)

    𝒢 = forwardguiding(G, ms, obs, Πroot)

    # Initialise the first guided sample
    Z = randn(G.N, G.T)
    U = cdf.(Normal(), Z)
    S, w = 𝒢(U)
    Sinit = copy(S) # to save
    Zinit = copy(Z)

    Savg = zeros(G.N, G.T)
    ws = [w]
    Ss = [S]

    ACCZ = 0
    k = 0

    for i = 1:ITER
        for ind in partitions
            k +=1 

            Uᵒ, Zᵒ = pcn(Z, ρ, ind) 
       
            Sᵒ, wᵒ = 𝒢(Uᵒ)

            A = Sᵒ == S # check if prev image S is identical to new image S′

            if log(rand()) < wᵒ - w
                U .= Uᵒ
                Z .= Zᵒ
                S, w = Sᵒ, wᵒ
                ACCZ += 1
            end

            if (i % 50 == 0)
                @printf("iteration: %d %d | Z rate: %.4f | logweight: %.4e | assert: %d\n", i, k,  ACCZ/k, w, A)
            end
            push!(ws, w)
        end

#       push!(Zs,  (Z[22,11], Z[5,4]))
        if i > BI  Savg += S end
        if (i % 100 == 0)    push!(Ss, S)          end
    end

    (Sinit=Sinit, Slast=S, Siterates=Ss, Savg=Savg, weights=ws, Zinit=Zinit, Zlast=Z)
end



function partition_into_blocks_close(N::Int, n::Int)
    # Check if n is smaller than N
    if n >= N
        throw(ArgumentError("n must be smaller than N"))
    end
    
    # Initialize partitions array
    partitions = Array{Vector{Int}, 1}(undef, n)
    
    # Initialize blocks
    for i in 1:n
        partitions[i] = Int[]
    end
    
    # Calculate the number of elements per block
    elements_per_block = div(N, n)
    
    # Distribute the elements evenly
    current_block = 1
    current_element = 1
    for _ in 1:n
        while length(partitions[current_block]) < elements_per_block && current_element <= N
            push!(partitions[current_block], current_element)
            current_element += 1
        end
        current_block += 1
    end
    
    # Distribute the remaining elements
    while current_element <= N
        push!(partitions[mod(current_element, n) == 0 ? n : mod(current_element, n)], current_element)
        current_element += 1
    end
    
    return partitions
end



function mcmcU(G, ms, obs, Πroot; ITER=100, BIfactor=4, δ=0.1, NUMBLOCKS=10)
    BI = ITER÷BIfactor
    partitions = partition_into_blocks_close(G.T, NUMBLOCKS)

    𝒢 = forwardguiding(G, ms, obs, Πroot)

    # Initialise the first guided sample
    U = rand(G.N, G.T)
    S, w = 𝒢(U)
    Sinit = copy(S) # to save
    Uinit = copy(U)

    Savg = zeros(G.N, G.T)
    ws = [w]
    Ss = [S]

    ACCZ = 0
    k = 0

    for i = 1:ITER
        for ind in partitions
            k +=1 

            Uᵒ = u_update(U, δ, ind)
            Sᵒ, wᵒ = 𝒢(Uᵒ)

            A = Sᵒ == S # check if prev image S is identical to new image S′

            if log(rand()) < wᵒ - w
                U .= Uᵒ
                S, w = Sᵒ, wᵒ
                ACCZ += 1
            end

            if (i % 5 == 0)
                @printf("iteration: %d | Z rate: %.4f | logweight: %.4e | assert: %d\n", k,  ACCZ/k, w, A)
            end
            push!(ws, w)
        end

#       push!(Zs,  (Z[22,11], Z[5,4]))
        if i > BI  Savg += S end
        if (i % 100 == 0)    push!(Ss, S)          end
    end

    (Sinit=Sinit, Slast=S, Siterates=Ss, Savg=Savg, weights=ws, Zinit=Uinit, Zlast=U)
end