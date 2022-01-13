"""
Running quantile estimator
"""
mutable struct RunningQuantileEstimator
    q::Vector{Float64}
    n::Vector{Int}
    n_prime::Vector{Float64}
    dn::Vector{Float64}
    last_i::Int
    α::Float64
end

RunningQuantileEstimator(α::Float64) = RunningQuantileEstimator(zeros(5),
                                                                        [i for i=1:5],
                                                                        [1, 1 + 2α, 1+4α, 3+2α, 5],
                                                                        [0, α/2, α, (1+α)/2, 1],
                                                                        0, α)

function quantile(est::RunningQuantileEstimator)
    if est.last_i < 5
        return nothing
    else
        return est.q[3]
    end
end

"""
interpolation functions
"""
function parabolic_update(i::Int, q::Vector{Float64}, n, d)
    return q[i] + d/(n[i+1]-n[i-1])*((n[i]-n[i-1]+d)*(q[i+1]-q[i])/(n[i+1]-n[i]) + (n[i+1]-n[i]-d)*(q[i]-q[i-1])/(n[i]-n[i-1]))
end

function linear_update(i::Int, q::Vector{Float64}, n, d)
    d = Int(d)
    return q[i] + d/(n[i+d]-n[i])*(q[i+d]-q[i])
end

"""
P-square algorithm adapted from (Jain and Chlamtac)
"""
function update_quantile!(est::RunningQuantileEstimator, x::Float64)
    q = est.q
    n = est.n
    n_prime = est.n_prime
    dn = est.dn

    if x < q[1]
        q[1] = x
        k = 1
    elseif q[1] ≤ x < q[2]
        k = 1
    elseif q[2] ≤ x < q[3]
        k = 2
    elseif q[3] ≤ x < q[4]
        k = 3
    elseif q[4] ≤ x < q[5]
        k = 4
    else
        q[5] = x
        k = 4
    end
    for i=k:5
        n[i] = n[i] + 1
    end
    n_prime = n_prime .+ dn
    for i=2:4
        d = n_prime[i] - n[i]
        if ((d ≥ 1) && (n[i+1] - n[i] > 1))||((d ≤ -1) && (n[i-1] - n[i] < -1))
            d = sign(d)
            q_prime = parabolic_update(i, q, n, d)
            if q[i-1] < q_prime < q[i+1]
                q[i] = q_prime
            else
                q[i] = linear_update(i, q, n, d)
            end
            n[i] = n[i] + d
        end
    end
    est.q = q
    est.n = n
    est.n_prime = n_prime
    est.last_i = est.last_i + 1

end
