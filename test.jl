# pcn versus cyclic uniform updates
B = 10000

u = 0.1
z = quantile(Normal(), u)
w = randn(B)
ρ =0.5
zᵒ = [ρ*z + √(1-ρ^2) * w_ for w_ in w]
uᵒ = cdf.(Normal(), zᵒ)
histogram(zᵒ)
histogram(uᵒ)
#
u = 0.91
δ = 0.25
uᵒ = mod.(u .+ rand(Uniform(-δ, δ), B),1)
histogram(uᵒ)


function partition_into_blocks(N::Int, n::Int)
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
    
    # Assign numbers to blocks
    for num in 1:N
        block_index = mod(num, n) == 0 ? n : mod(num, n)
        push!(partitions[block_index], num)
    end
    
    return partitions
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


N = 101
n = 5
partitions = partition_into_blocks_close(N, n)
for (i, block) in enumerate(partitions)
    println("Block $i: $block")
end
