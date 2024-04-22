
# Function to exponentiate and normalize log-weights
function resamp(lw, ess_threshold)
    w = exp.(lw .- maximum(lw))
    nw = w/sum(w)
    ess = 1 / sum(nw.^2)
    #println(ess)
    if ess < ess_threshold
        println("ess equals: $ess and is below threshold, resampling takes place\n")
        return SMC.resample(nw)
    else
        return 1:length(lw)
    end
end


# lw = [log(1.2), log(0.3), log(10.1), log(0.4)]
# resamp(lw, .2)