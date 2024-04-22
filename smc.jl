function move((Z, S, w, qZ), NR_MOVE_STEPS, ρ,  tinterval, G, ms, obs, Πroot)
    # Update Z for each segment of 50 time steps individually
    blocks = (T-1)÷tinterval
    N = G.N

    ACCZ = 0
    for i = 1:NR_MOVE_STEPS
        # Z step only
        for k = 1:blocks
            qW = randn(Float64, (N, tinterval))

            qZ′ = copy(qZ)
            qZ′[:,(k-1)*tinterval+2:k*tinterval+1] = ρ*qZ′[:,(k-1)*tinterval+2:k*tinterval+1] + √(1 - ρ^2)*qW

            Z′ = cdf.(Normal(), qZ′)
            S′, w′ = forwardguiding(G, ms, obs, Z′,Πroot)
            
            if log(rand()) < w′ - w
                qZ = qZ′
                Z = Z′
                S, w = S′, w′
                ACCZ += 1
            end
        end
#        avgACCZ = 100.0*round(ACCZ/(blocks*ITER); digits=2)
   #     println("acceptance percentage for move: $avgACCZ %")
     end
     (Z=Z, S=S, w=w, qZ=qZ)
end

function inititalise_particle(G, ms, obs, Πroot)
    dims = (G.N, G.T)
    Z = rand(Float64, dims)
    S, w = forwardguiding(G, ms, obs, Z, Πroot)
    qZ  = quantile.(Normal(), Z)
    (Z=Z, S=S, w=w, qZ=qZ)
end




function smc(ρ, NR_SMC_STEPS, NUMPARTICLES, NR_MOVE_STEPS, G, ms, obs, tinterval)  
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
        particles = map(x -> move(particles[x], NR_MOVE_STEPS, ρ,tinterval, G, ms, obs, Πroot), 1:NUMPARTICLES)
        push!(lls, [particles[i].w for i in eachindex(particles)])
    end
    particles, lls
end

