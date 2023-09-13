function mcmc(G, ms, obs, Πroot; ITER=100, BIfactor=3, ρ=0.99, tinterval=10)
    BI = ITER÷BIfactor
    # takes blocks of size tinterval
    blocks = (G.T-1)÷tinterval

    𝒢 = forwardguiding(G, ms, obs, Πroot)

    # Initialise the first guided sample
    Zinit = rand(Float64, (G.N, G.T))
    Sinit, winit = 𝒢(Zinit)

    # Initialise MCMC parameters
    Z = copy(Zinit); S = copy(Sinit); w = copy(winit);
    qZ = quantile.(Normal(), Z)

    Savg = zeros(G.N, G.T)
    ws = [w]

    ACCZ = 0

    Ss = [S]
#    Zs = [(Z[22,11], Z[5,4])]  # just some Zs to monitor mixing
    for i = 1:ITER
        # Z step only
        for k = 1:blocks
            qZ′ = copy(qZ)
            ind = (k-1)*tinterval+1:k*tinterval+1
            qW = randn(Float64, (N, length(ind)))
            qZ′[:,ind] = ρ*qZ′[:,ind] + √(1 - ρ^2)*qW
            Z′ = cdf.(Normal(), qZ′)
            S′, w′ = 𝒢(Z′)

            A = S′ == S # check if prev image S is identical to new image S′

            if log(rand()) < w′ - w
                qZ = qZ′
                Z = Z′
                S, w = S′, w′
                ACCZ += 1
            end
            

            if (i % 5 == 0)
                @printf("iteration: %d %d | Z rate: %.4f | logweight: %.4e | assert: %d\n", i, k,  ACCZ/((i-1)*blocks + (k-1) + 1), w, A)
            end
            push!(ws, w)
        end
       
#       push!(Zs,  (Z[22,11], Z[5,4]))
        if i > BI  Savg += S end 
        if (i % 500 == 0)    push!(Ss, S)          end
    end

    (Sinit=Sinit, Slast=S, Siterates=Ss, Savg=Savg, weights=ws, Zinit=Zinit, Zlast=Z)
end
