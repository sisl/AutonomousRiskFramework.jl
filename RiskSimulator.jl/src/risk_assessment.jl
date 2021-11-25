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
    # If no failures, no cost distribution.
    Z = length(Z) == 0 ? [Inf] : Z
    𝒫 = ecdf(Z)
    𝒞 = conditional_distr(𝒫, Z, α)
    𝔼 = mean(Z)
    var = VaR(𝒞)
    cvar = CVaR(𝒞)
    return RiskMetrics(Z=Z, α=α, 𝒫=𝒫, 𝒞=𝒞, mean=𝔼, var=VaR(𝒞), cvar=CVaR(𝒞), worst=worst_case(Z))
end

function RiskMetricsModeled(Z, α, ℱ; length=1000)
    𝒫 = z->cdf(ℱ, z)
    𝒞 = conditional_distr_model(𝒫, α, ℱ; length=length)
    𝔼 = mean(ℱ)
    var = VaR(𝒞)
    cvar = CVaR(𝒞)
    return RiskMetrics(Z=Z, α=α, 𝒫=𝒫, 𝒞=𝒞, mean=𝔼, var=VaR(𝒞), cvar=CVaR(𝒞), worst=worst_case(Z))
end

conditional_distr(𝒫,Z,α) = filter(z->1-𝒫(z) ≤ α, Z)
conditional_distr_model(𝒫,α,ℱ;length=1000) = filter(z->1-𝒫(z) ≤ α, rand(ℱ, length))

VaR(𝒫,Z,α) = minimum(conditional_distr(𝒫,Z,α))
VaR(𝒞) = minimum(𝒞)

worst_case(Z) = maximum(Z)

CVaR(𝒫,Z,α) = mean(conditional_distr(𝒫,Z,α))
CVaR(𝒞) = mean(𝒞)


metrics(planner, α=0.2) = metrics(planner.mdp.dataset, α)
function metrics(𝒟::Vector, α=0.2)
    Z = cost_data(𝒟)
    return RiskMetrics(Z, α)
end


combine_datasets(planners) = vcat(map(planner->planner.mdp.dataset, planners)...)


"""
Combine datasets from different runs then collect risk metrics.
"""
function collect_metrics(planners, α)
    dataset = combine_datasets(planners)
    metrics = metrics(dataset, α)
    return metrics
end


"""
Return the cost data (Z) of the failures or `nonfailures` (i.e., rate/severity).

See POMDPStressTesting.jl/src/AST.jl for how 𝒟 is constructed. At a high level,
𝒟 is a vector of tuples of a vector of size 3 containing AST actions,
distances, and rates, and a boolean indicating an event.
"""
function cost_data(𝒟; nonfailures=false, terminal_only=true)
    if typeof(𝒟[1][1]) <: Vector{Vector{Any}}
        # [end] in the 𝐱 data, and [2:end] to remove the first rate value (0 - first distance)
        costs = [abs.(d[1][end][2:end]) for d in filter(d -> nonfailures ? !d[2] : d[2], 𝒟)]
        # when we collect data for FULL trajectory (not just at the terminal state)
        if terminal_only
            filter!(!isempty, costs)
            return convert(Vector{Real}, vcat(last.(costs)...))
        else
            return convert(Vector{Real}, vcat(costs...))
        end
    else
        return [abs.(d[1][end]) for d in filter(d -> nonfailures ? !d[2] : d[2], 𝒟)]
    end
end


"""
Return the distance data (𝐝) of the failures or `nonfailures`.
"""
function distance_data(𝒟; nonfailures=false, terminal_only=true)
    if typeof(𝒟[1][1]) <: Vector{Vector{Any}}
        # [end] in the 𝐱 data, and [2:end] to match the removal of the first rate value (0 - first distance)
        distances = [d[1][end-1][2:end] for d in filter(d->nonfailures ? !d[2] : d[2], 𝒟)]
        # when we collect data for FULL trajectory (not just at the terminal state)
        if terminal_only
            return convert(Vector{Real}, vcat(last.(distances)...))
        else
            return convert(Vector{Real}, vcat(distances...))
        end
    else
        return [d[1][end-1] for d in filter(d -> nonfailures ? !d[2] : d[2], 𝒟)]
    end
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


"""
Weight used to normalize maximum likelihood in polar plot/AUC.
"""
function inverse_max_likelihood(failure_metrics_vector_set)
    return 1/maximum(map(fmv->maximum(map(fm->exp(fm.highest_loglikelihood), fmv)), failure_metrics_vector_set)) # 1/max(p)
end


function risk_statistic(rms::Vector{RiskMetrics}, func)
    Z = vcat(map(m->m.Z, rms)...)
    α = first(rms).α
    𝒫 = ecdf(Z)
    𝒞 = conditional_distr(𝒫, Z, α)

    Z_mean = func(filter(!isinf, [m.mean for m in rms]))
    var = func(filter(!isinf, [m.var for m in rms]))
    cvar = func(filter(!isinf, [m.cvar for m in rms]))
    worst = func(filter(!isinf, [m.worst for m in rms]))

    return RiskMetrics(Z=Z, α=α, 𝒫=𝒫, 𝒞=𝒞, mean=Z_mean, var=var, cvar=cvar, worst=worst)
end

Statistics.mean(rms::Vector{RiskMetrics}) = risk_statistic(rms, mean)
Statistics.std(rms::Vector{RiskMetrics}) = risk_statistic(rms, std)
