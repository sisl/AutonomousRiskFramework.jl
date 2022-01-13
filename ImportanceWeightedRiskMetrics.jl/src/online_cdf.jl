"""
Running cdf estimator
"""
mutable struct RunningCDFEstimator
    Xs::Vector{Float64}
    partial_Ws::Vector{Float64}
    last_i::Int
end

RunningCDFEstimator() = RunningCDFEstimator([], [], 0)

function RunningCDFEstimator(X::Vector{Float64}, W::Vector{Float64})
    perm = sortperm(X)
    Xs = X[perm]
    Ws = W[perm]
    partial_Ws = [sum(Ws[1:i]) for i=1:length(Ws)]
    sumW = sum(W)
    n = length(X)

    return RunningCDFEstimator(Xs, partial_Ws, n)
end

"""
CDF function
"""
cdf(est::RunningCDFEstimator, x) = est.partial_Ws[searchsortedlast(est.Xs, x)] / est.partial_Ws[est.last_i]

"""
Quantile function
"""
quantile(est::RunningCDFEstimator, α::Float64) = est.Xs[searchsortedfirst(est.partial_Ws, (1 - α)*est.partial_Ws[est.last_i])]

"""
Update function
"""
function update!(est::RunningCDFEstimator, x, w)
    if est.last_i == 0
        push!(est.Xs, x)
        push!(est.partial_Ws, w)
    elseif x < first(est.Xs)
        pushfirst!(est.Xs, x)
        pushfirst!(est.partial_Ws, 0.0)
        est.partial_Ws = est.partial_Ws .+ w
    elseif x ≥ last(est.Xs)
        push!(est.Xs, x)
        push!(est.partial_Ws, est.partial_Ws[est.last_i] + w)
    else
        new_idx = searchsortedlast(est.Xs, x) + 1
        splice!(est.Xs, new_idx:new_idx-1, [x])
        splice!(est.partial_Ws, new_idx:new_idx-1, [est.partial_Ws[new_idx-1]])
        @show new_idx, est.Xs, est.partial_Ws
        est.partial_Ws[new_idx:end] = est.partial_Ws[new_idx:end] .+ w
    end
    est.last_i += 1
end
