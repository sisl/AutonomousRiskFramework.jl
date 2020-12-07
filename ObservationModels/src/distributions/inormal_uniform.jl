struct INormal_Uniform{T<:Real} <: ContinuousUnivariateDistribution
    mu::Float64
    sigma::Float64
    mu_sup::Float64
    sup::Float64
    INormal_Uniform(mu::T, sigma::T, mu_sup::T, sup::T) where {T<:Real} = new{T}(Float64(mu), Float64(sigma), Float64(mu_sup), Float64(sup))
end

INormal_Uniform(mu, sigma, mu_sup) = INormal_Uniform(Float64(mu), Float64(sigma), Float64(mu_sup), 10.0)
INormal_Uniform(mu, sigma) = INormal_Uniform(Float64(mu), Float64(sigma), Float64(mu))
Distributions.pdf(d::INormal_Uniform, x::Float64) = Distributions.pdf(Normal(d.mu, d.sigma), x)
Base.rand(d::INormal_Uniform) = rand(Uniform(d.mu_sup-d.sup/2, d.mu_sup+d.sup/2))
Base.rand(rng::AbstractRNG, d::INormal_Uniform) = rand(rng, Uniform(d.mu_sup-d.sup/2, d.mu_sup+d.sup/2))
Distributions.sampler(d::INormal_Uniform) = Distributions.sampler(Uniform(d.mu_sup-d.sup/2, d.mu_sup+d.sup/2))
Distributions.logpdf(d::INormal_Uniform, x::Real) = Distributions.logpdf(Normal(d.mu, d.sigma), x) 
Distributions.cdf(d::INormal_Uniform, x::Real) = Distributions.cdf(Normal(d.mu, d.sigma), x)
Distributions.quantile(d::INormal_Uniform, q::Real) = Distributions.quantile(Normal(d.mu, d.sigma), q)
Base.minimum(d::INormal_Uniform) = d.mu-d.sup/2
Base.maximum(d::INormal_Uniform) = d.mu+d.sup/2
Distributions.insupport(d::INormal_Uniform, x::Real) = Distributions.insupport(Normal(d.mu, d.sigma), x)
Distributions.mean(d::INormal_Uniform) = Distributions.mean(Normal(d.mu, d.sigma))
Distributions.var(d::INormal_Uniform) = Distributions.var(Normal(d.mu, d.sigma))

function Distributions.fit(::Type{<:INormal_Uniform}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    norm_fit = Distributions.fit(Normal{T}, x, w)
    INormal_Uniform(norm_fit.μ, norm_fit.σ)
end