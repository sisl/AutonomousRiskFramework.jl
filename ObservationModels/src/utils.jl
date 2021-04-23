"""
Creates a copy of the vehicle state with the new specified noise
"""
update_veh_noise(s::NoisyPedState, noise::Noise) = NoisyPedState(s.veh_state, noise)
update_veh_noise(s::BlinkerState, noise::Noise) = BlinkerState(s.veh_state, s.blinker, s.goals, noise)

"""
Utility functions for dictionary of distributions
"""
function Base.rand(rng::AbstractRNG, d::Dict{Symbol, Vector{Sampleable}})
    Dict(k => rand.(Ref(rng), d[k]) for k in keys(d))
end

function Distributions.fit(d::Dict{Symbol, Vector{Sampleable}}, samples, weights; add_entropy = (x) -> x)
    N = length(samples)
    new_d = Dict{Symbol, Vector{Sampleable}}()
    for s in keys(d)
        dtype = typeof(d[s][1])
        m = length(d[s])
        new_d[s] = [add_entropy(fit(dtype, [samples[j][s][i] for j=1:N], weights)) for i=1:m]
    end
    new_d
end

function Distributions.logpdf(d::Dict{Symbol, Vector{Sampleable}}, x, i)
    sum([logpdf(d[k][i], x[k][i]) for k in keys(d)])
end

function Distributions.logpdf(d::Dict{Symbol, Vector{Sampleable}}, x)
    sum([logpdf(d, x, i) for i=1:length(first(x)[2])])
end

"""
Probability of gaussian distribution
"""
function gaussian_distribution(y, μ, σ)
    return 1 ./ ((sqrt(2π).*σ)).*exp.(-0.5((y .- μ)./σ).^2)
end;

function log_gaussian_distribution(y, μ, σ)
    term1 = 0.5((y .- μ)./σ).^2
    term1 = clamp(term1, 1e-3, 1e10)
    return -0.5*log(2π).-log(σ).-term1
end;