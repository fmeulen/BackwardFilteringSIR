function move((U, S, w), NR_MOVE_STEPS, δ,  NUMBLOCKS, G, ms, obs, Πroot)
    # Update Z for each segment of 50 time steps individually
    partitions = partition_into_blocks_close(G.T, NUMBLOCKS)
    N = G.N
    𝒢 = forwardguiding(G, ms, obs, Πroot)
    ws = [w]

    ACCZ = 0
    for i = 1:NR_MOVE_STEPS
        for ind in partitions
            Uᵒ = u_update(U, δ, ind)
            Sᵒ, wᵒ = 𝒢(Uᵒ)
            if log(rand()) < wᵒ - w
                U .= Uᵒ
                S, w = Sᵒ, wᵒ
                ACCZ += 1
            end
            push!(ws, w)
        end
    end
    
     (U=U, S=S, w=w)
end

function inititalise_particle(G, ms, obs, Πroot)
    dims = (G.N, G.T)
    U = rand(Float64, dims)
    S, w = forwardguiding(G, ms, obs, U, Πroot)
    (U=U, S=S, w=w)
end




function smc(δ, NR_SMC_STEPS, NUMPARTICLES, NR_MOVE_STEPS, G, ms, obs, NUMBLOCKS)  
    # initialise particles 
    particles = [inititalise_particle(G, ms, obs, Πroot) for _ in 1:NUMPARTICLES]
    
    ess_threshold = round(NUMPARTICLES/2; digits=0)    
    lls = []

    for j in 1:NR_SMC_STEPS
        println(j,"\n")
        # resample step
        log_weights =  getindex.(particles, :w)
        indices = resamp(log_weights, ess_threshold) 
        println(indices)
        particles = [particles[k] for k in indices]
        # move step (pCN)
        particles = map(x -> move(particles[x], NR_MOVE_STEPS, δ, NUMBLOCKS, G, ms, obs, Πroot), 1:NUMPARTICLES)
        push!(lls, [particles[i].w for i in eachindex(particles)])
    end
    particles, lls
end

