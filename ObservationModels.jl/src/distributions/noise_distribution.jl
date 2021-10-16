"""
Distribution utilities over `Noise` objects 
"""
Distributions.pdf(d::T, x::Noise) where T <: ContinuousMultivariateDistribution = Distributions.pdf(d, [x.pos.x, x.pos.y, x.vel])
Distributions.logpdf(d::T, x::Noise) where T <: ContinuousMultivariateDistribution = Distributions.logpdf(d, [x.pos.x, x.pos.y, x.vel])

Distributions.pdf(d::T, x::Vector{Noise}) where T <: ContinuousMultivariateDistribution = [Distributions.pdf(d, x[i]) for i in 1:length(x)]

Distributions.logpdf(d::T, x::Vector{Noise}) where T <: ContinuousMultivariateDistribution = [Distributions.logpdf(d, x[i]) for i in 1:length(x)]

function Distributions.fit(dtype::Type{<:ContinuousMultivariateDistribution}, x::Vector{Noise}, w::AbstractArray{Float64}) 
    n = length(x)
    x_matrix = Array{Float64}(undef, 3, n)
    for i in 1:n
        x_matrix[1, i] = x[i].pos.x + 1e-7*randn()
        x_matrix[2, i] = x[i].pos.y + 1e-7*randn()
        x_matrix[3, i] = x[i].vel + 1e-7*randn()
    end
    Distributions.fit(dtype, x_matrix, w)  
end

function Distributions.fit(dtype::Type{<:ContinuousMultivariateDistribution}, x::Vector{Noise})
    Distributions.fit(dtype, x, ones(length(x)))  
end