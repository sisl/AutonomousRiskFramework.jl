"""
Importance weighted risk metrics
"""
@with_kw struct IWRiskMetrics
    Z # cost data
    w # importance weights
    α # probability threshold

    𝒫 # emperical CDF
    𝒞 # conditional distribution

    mean # expected value
    var # Value at Risk
    cvar # Conditional Value at Risk
    worst # worst case
end

function IWRiskMetrics(Z,w,α)
    # If no failures, no cost distribution.
    if length(Z) == 0
        Z = [Inf]
        w = [1.0]
    end
    𝒫 = ecdf(Z, w)
    𝒞, w_conditional = conditional_distr(𝒫, Z, α, w)
    𝔼 = mean(Z, weights(w))
    var = VaR(𝒞, w_conditional)
    cvar = CVaR(𝒞, w_conditional)
    return IWRiskMetrics(Z=Z, w=w, α=α, 𝒫=𝒫, 𝒞=𝒞, mean=𝔼, var=var, cvar=cvar, worst=worst_case(Z, w))
end

"""
Conditional distribution from importance weighted samples
"""
function conditional_distr(𝒫,Z,α,w)
    idx = filter(i->1-𝒫(Z[i]) ≤ α, 1:length(Z))
    return Z[idx], w[idx] 
end

"""
Importance weighted Empirical CDF
"""
function StatsBase.ecdf(X, w)
    perm = sortperm(X)
    Xs = X[perm]
    ws = w[perm]
    n = length(X)
    tot_w = sum(w)

    ef(x) = sum(ws[1:searchsortedlast(Xs, x)]) / tot_w

    return ef
end


"""
Importance weighted Value-at-Risk and Conditional Value-at-Risk
"""
CVaR(𝒞, w) = mean(𝒞, weights(w))
VaR(𝒞) = minimum(𝒞)
VaR(𝒞, w) = VaR(𝒞)

"""
Wrapper for worst-case value with weighting
"""
worst_case(Z) = maximum(Z)
worst_case(Z, w) = worst_case(Z)