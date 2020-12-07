struct INormal_GMM{T<:Real, C<:Categorical} <: ContinuousUnivariateDistribution
    mu::T
    sigma::T
    gmm_mu::Vector{T}
    gmm_sigma::Vector{T}
    gmm_prior::C
    INormal_GMM(mu::T, sigma::T, gmm_mu::AbstractVector{T}, gmm_sigma::AbstractVector{T}, gmm_prior::C) where {T<:Real,C<:Categorical} = new{T,C}(T(mu), T(sigma), gmm_mu, gmm_sigma, gmm_prior)
end

INormal_GMM(mu, sigma) = INormal_GMM(Float64(mu), Float64(sigma), [Float64(mu), Float64(mu)], [sigma/2.0, sigma*2.0], Categorical(0.5, 0.5))
Distributions.pdf(d::INormal_GMM, x::Float64) = Distributions.pdf(Normal(d.mu, d.sigma), x)
Base.rand(d::INormal_GMM) = rand(UnivariateGMM(d.gmm_mu, d.gmm_sigma, d.gmm_prior))
Base.rand(rng::AbstractRNG, d::INormal_GMM) = rand(rng, UnivariateGMM(d.gmm_mu, d.gmm_sigma, d.gmm_prior))
Distributions.sampler(d::INormal_GMM) = Distributions.sampler(UnivariateGMM(d.gmm_mu, d.gmm_sigma, d.gmm_prior))
Distributions.logpdf(d::INormal_GMM, x::Real) = Distributions.logpdf(Normal(d.mu, d.sigma), x) 
Distributions.cdf(d::INormal_GMM, x::Real) = Distributions.cdf(Normal(d.mu, d.sigma), x)
Distributions.quantile(d::INormal_GMM, q::Real) = Distributions.quantile(Normal(d.mu, d.sigma), q)
Base.minimum(d::INormal_GMM) = -Inf
Base.maximum(d::INormal_GMM) = Inf
Distributions.insupport(d::INormal_GMM, x::Real) = Distributions.insupport(Normal(d.mu, d.sigma), x)
Distributions.mean(d::INormal_GMM) = Distributions.mean(Normal(d.mu, d.sigma))
Distributions.var(d::INormal_GMM) = Distributions.var(Normal(d.mu, d.sigma))

function Distributions.fit(::Type{<:INormal_GMM}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    norm_fit = Distributions.fit(Normal{T}, x, w)
    INormal_GMM(norm_fit.μ, norm_fit.σ)
end