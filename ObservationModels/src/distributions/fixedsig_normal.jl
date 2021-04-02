"""
Normal distribution object with fixed standard deviation
"""
struct Fsig_Normal{T<:Real} <: ContinuousUnivariateDistribution
    mu::Float64
    Fsig_Normal(mu::T) where {T<:Real} = new{T}(Float64(mu))
end

Fsig_Normal(mu) = Normal(Float64(mu), 5.0)
Distributions.pdf(d::Fsig_Normal, x::Float64) = Distributions.pdf(Normal(d.mu, 5.0), x)
Base.rand(d::Fsig_Normal) = rand(Normal(d.mu, 5.0))
Base.rand(rng::AbstractRNG, d::Fsig_Normal) = rand(rng, Normal(d.mu, 5.0))
Distributions.sampler(d::Fsig_Normal) = Distributions.sampler(Normal(d.mu, 5.0))
Distributions.logpdf(d::Fsig_Normal, x::Real) = Distributions.logpdf(Normal(d.mu, 5.0), x)
Distributions.cdf(d::Fsig_Normal, x::Real) = Distributions.cdf(Normal(d.mu, 5.0), x)
Distributions.quantile(d::Fsig_Normal, q::Real) = Distributions.quantile(Normal(d.mu, 5.0), q)
Base.minimum(d::Fsig_Normal) = -Inf
Base.maximum(d::Fsig_Normal) = Inf
Distributions.insupport(d::Fsig_Normal, x::Real) = Distributions.insupport(Normal(d.mu, 5.0), x)
Distributions.mean(d::Fsig_Normal) = Distributions.mean(Normal(d.mu, 5.0))
Distributions.var(d::Fsig_Normal) = Distributions.var(Normal(d.mu, 5.0))

function Distributions.fit(::Type{<:Fsig_Normal}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    norm_fit = Distributions.fit(Normal{T}, x, w)
    Fsig_Normal(norm_fit.μ)
end