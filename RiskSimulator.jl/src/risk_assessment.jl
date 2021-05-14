##############################################################################
# Risk Assessment
##############################################################################
using Distributions
using Plots
using PGFPlotsX
using LaTeXStrings
using StatsBase
using Statistics
using LinearAlgebra
using Parameters
using Markdown

@with_kw struct RiskMetrics
    Z # cost data
    α # probability threshold

    𝒫 # emperical CDF
    𝒞 # conditional distribution

    mean # expected value
    var # Value at Risk
    cvar # Conditional Value at Risk
    worst # worst case
end


function RiskMetrics(Z,α)
    𝒫 = ecdf(Z)
    𝒞 = conditional_distr(𝒫, Z, α)
    𝔼 = mean(Z)
    var = VaR(𝒞)
    cvar = CVaR(𝒞)
    return RiskMetrics(Z=Z, α=α, 𝒫=𝒫, 𝒞=𝒞, mean=𝔼, var=VaR(𝒞), cvar=CVaR(𝒞), worst=worst_case(Z))
end

conditional_distr(𝒫,Z,α) = filter(z->1-𝒫(z) ≤ α, Z)

VaR(𝒫,Z,α) = minimum(conditional_distr(𝒫,Z,α))
VaR(𝒞) = minimum(𝒞)

worst_case(Z) = maximum(Z)

CVaR(𝒫,Z,α) = mean(conditional_distr(𝒫,Z,α))
CVaR(𝒞) = mean(𝒞)


function risk_assessment(𝒟, α=0.2)
    metrics = RiskMetrics(cost_data(𝒟), α)
    return metrics
end


"""
Return the cost data (Z) of the failures or `nonfailures` (i.e., rate/severity).
"""
function cost_data(𝒟; nonfailures=false)
    return [d[1][end] for d in filter(d->nonfailures ? !d[2] : d[2], 𝒟)]
end


"""
Return the distance data (𝐝) of the failures or `nonfailures`.
"""
function distance_data(𝒟; nonfailures=false)
    return [d[1][end-1] for d in filter(d->nonfailures ? !d[2] : d[2], 𝒟)]
end


"""
Display risk metrics in a LaTeX enviroment.
Useful in Pluto.jl notebooks.
"""
function latex_metrics(metrics::RiskMetrics)
    # Note indenting is important here to render correctly.
    return Markdown.parse(string("
\$\$\\begin{align}",
"\\alpha &=", metrics.α, "\\\\",
"\\mathbb{E}[Z] &=", round(metrics.mean, digits=3), "\\\\",
"\\operatorname{VaR}_{", metrics.α, "}(Z) &=", round(metrics.var, digits=3), "\\\\",
"\\operatorname{CVaR}_{", metrics.α, "}(Z) &=", round(metrics.cvar, digits=3), "\\\\",
"\\text{worst case} &=", round(metrics.worst, digits=3), "\\\\",
"\\end{align}\$\$"))
end
