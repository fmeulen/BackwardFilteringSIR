function boyenkoller(factoredpullback::T, factoredhtransform::T, n, t, k) where T
    # A simple implementation of junction tree algorithm for {i-2:i+2} - {i-1:i+3} - etc j-tree
    l = length(E)
    ϕs = T() # holds results

    # i = 1
    μ1i = copy(factoredpullback[1])
    obs = repeat(factoredhtransform[1], outer=l^k)
    μ1i .*= obs
    #println(length(obs) == length(μ1i))

    μ1i = μ1i / sum(μ1i)
    leftμs = [μ1i]

    for i=2:k
        μ1i = repeat(μ1i, outer=l) .* factoredpullback[i]
        obs = repeat(factoredhtransform[i], inner=l^(i-1), outer=l^k)
        μ1i .*= obs
        #println(length(obs) == length(μ1i))

        μ1i = μ1i / sum(μ1i)
        push!(leftμs, μ1i)
    end

    for i=k+1:n-k
        temp = repeat(μ1i, outer=l) .* factoredpullback[i]
        obs = repeat(factoredhtransform[i], inner=l^k, outer=l^k)
        temp .*= obs
        #println(length(obs) == length(temp))

        μ1i = vec(sum(reshape(temp, ntuple(c -> l, 2*k+1)), dims=1))
        μ1i = μ1i / sum(μ1i)
        push!(leftμs, μ1i)
    end

    for i=n-k+1:n-1
        temp = μ1i .* factoredpullback[i]
        obs = repeat(factoredhtransform[i], inner=l^k, outer=l^(n-i))
        temp .*= obs
        #println(length(obs) == length(temp))

        μ1i = vec(sum(reshape(temp, ntuple(c -> l, k+n-i+1)), dims=1))
        μ1i = μ1i / sum(μ1i)
        push!(leftμs, μ1i)
    end

    # i = n
    μin = copy(factoredpullback[n])
    obs = repeat(factoredhtransform[n], inner=l^k)
    μin .*= obs
    #println(length(obs) == length(μin))

    μin = μin / sum(μin)
    rightμs = [μin]

    for i=n-1:-1:n-k+1
        μin = factoredpullback[i] .* repeat(μin, inner=l)
        obs = repeat(factoredhtransform[i], inner=l^k, outer=l^(n-i))
        μin .*= obs
        #println(length(obs) == length(μin))

        μin = μin / sum(μin)
        pushfirst!(rightμs, μin)
    end

    for i=n-k:-1:k+1
        temp = factoredpullback[i] .* repeat(μin, inner=l)
        obs = repeat(factoredhtransform[i], inner=l^k, outer=l^k)
        temp .*= obs
        #println(length(obs) == length(temp))

        μin = vec(sum(reshape(temp, ntuple(c -> l, 2*k+1)), dims=2*k+1))
        μin = μin / sum(μin)
        pushfirst!(rightμs, μin)
    end

    for i=k:-1:2
        temp = factoredpullback[i] .* μin
        obs = repeat(factoredhtransform[i], inner=l^(i-1), outer=l^k)
        temp .*= obs
        #println(length(obs) == length(temp))

        μin = vec(sum(reshape(temp, ntuple(c -> l, k+i)), dims=k+i))
        μin = μin / sum(μin)
        pushfirst!(rightμs, μin)
    end

    pushfirst!(rightμs, zeros(1))

    ϕ1 = vec(sum(reshape(factoredpullback[1] .* rightμs[2], ntuple(c -> l, k+1)), dims=setdiff(1:k+1, 1)))
    ϕ1 = ϕ1 .* factoredhtransform[1]
    ϕs[1] = ϕ1/sum(ϕ1)

    for i=2:k
        ϕi = vec(sum(reshape(repeat(leftμs[i-1], outer=l) .* factoredpullback[i] .* rightμs[i+1], ntuple(c -> l, k+i)), dims=setdiff(1:k+i, i)))
        ϕi = ϕi .* factoredhtransform[i]
        ϕs[i] = ϕi/sum(ϕi)
    end

    for i=k+1:n-k
        ϕi = vec(sum(reshape(repeat(leftμs[i-1], outer=l) .* factoredpullback[i] .* repeat(rightμs[i+1], inner=l), ntuple(c -> l, 2*k+1)), dims=setdiff(1:2*k+1, k+1)))
        ϕi = ϕi .* factoredhtransform[i]
        ϕs[i] = ϕi/sum(ϕi)
    end

    for i=n-k+1:n-1
        ϕi = vec(sum(reshape(leftμs[i-1] .* factoredpullback[i] .* repeat(rightμs[i+1], inner=l), ntuple(c -> l, k+n-i+1)), dims=setdiff(1:k+n-i+1, k+1)))
        ϕi = ϕi .* factoredhtransform[i]
        ϕs[i] = ϕi/sum(ϕi)
    end

    ϕn = vec(sum(reshape(leftμs[n-1] .* factoredpullback[n], ntuple(c -> l, k+1)), dims=setdiff(1:k+1, k+1)))
    ϕn = ϕn .* factoredhtransform[n]
    ϕs[n] = ϕn/sum(ϕn)

    ϕs
end



## correctness proof
function proof()
    # initialise some random objects and take the product of ALL pullbacks, then normalise
    # compare result to boyenkoller function, which does this in a smart way by exploiting j-tree

    n = 8
    k = 2

    factoredhtf = [rand(3) for i=1:n]
    #factoredhtf = [1:3 .== 1, 1:3 .== 1, 1:3 .== 1, 1:3 .== 2, 1:3 .== 2, 1:3 .== 3, 1:3 .== 3, 1:3 .== 3]

    factoredtransitionmodel = [[rand(3^(k + i), 3) for i=1:k]..., [rand(3^(2 * k + 1), 3) for i=k+1:n-k]..., [rand(3^(k + n - i + 1), 3) for i=n-k+1:n]...]
    #factoredtransitionmodel = [G.cpdmatrix[i] for i in [1, 2, 3, 4, N-3, N-2, N-1, N]]

    factpullback = Dict([i => Vector(factoredtransitionmodel[i] * factoredhtf[i]) for i = 1:n])
    factobs = Dict([i => rand(3) for i = 1:n])
    #factobs = Dict([i => ones(3) for i = 1:n])

    f = boyenkoller(factpullback, factobs, n, 0, k)

    bigp = ones(3^n)
    for i=1:k
        temp = repeat(factpullback[i], outer=3^(n-(k+i)))
        bigp .*= temp
    end
    for i=k+1:n-k
        temp = repeat(factpullback[i], outer=3^(n-(k+i)), inner=3^(i-(k+1)))
        bigp .*= temp
    end
    for i=n-k+1:n
        temp = repeat(factpullback[i], inner=3^(i-(k+1)))
        bigp .*= temp
    end
    for i=1:n
        temp = repeat(factobs[i], inner=3^(i-1), outer=3^(n-i))
        bigp .*= temp
    end


    ftemp = reshape(bigp / sum(bigp), ntuple(c -> 3, n))
    f2 = [vec(sum(ftemp, dims=setdiff(1:n, i))) for i=1:n]

    #s = [f[i] .* factobs[i] for i=1:n]
    s = f
    s = Dict([i => s[i] / sum(s[i]) for i=1:n])

    for i=1:n
        println(f2[i] .- s[i])
    end
end

## direct marginalise, extra manual proof

#leftdims(i) = i in 1:k ? collect(1:i-1) : collect(1:k)

#rightdims(i) = i in 1:k ? collect(i+1:i+k) :
#               i in n-k+1:n ? collect(k+1+1:k+(n-i)+1) :
#                              collect(k+1+1:2*k+1)

#otherdims(i) = union(leftdims(i), rightdims(i))

#log3(n) = Int(round(log(n) / log(3)))

#f5 = typeof(f)()
#for (i, pb) in factpullback
#    temp = vec(sum( reshape(pb, ntuple(c -> 3, log3(length(pb)))) , dims=otherdims(i)))
#    f5[i] = temp / sum(temp)
#end
