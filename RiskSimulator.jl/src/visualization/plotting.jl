using ColorSchemes
using Convex
using Distributions
using Plots
using PGFPlotsX
using POMDPStressTesting # for: failure_metrics
using LaTeXStrings
using SCS
using StatsBase
using Statistics
using LinearAlgebra
using Parameters

include("../risk_assessment.jl")

global FCOLOR = "#d62728"
global NFCOLOR = "#2ca02c"


function use_latex_fonts(using_plots_jl=true)
    # Use LaTeX fonts for rendering PyPlots
    mpl = using_plots_jl ? Plots.PyPlot.matplotlib : matplotlib
    mpl.rc("font", family=["serif"])
    mpl.rc("font", serif=["Helvetica"])
    mpl.rc("text", usetex=true)
end


function plot_cost_distribution(Z)
    viridis_green = "#238a8dff"
    return histogram(Z,
        color=viridis_green,
        label=nothing,
        alpha=0.5,
        reuse=false,
        xlabel=L"\operatorname{cost}",
        ylabel=L"\operatorname{probability}",
        normalize=:probability, size=(600, 300))
end

zero_ylims(adjust=0.001) = ylims!(0, ylims()[2]+adjust)


# TODO: Calculate mean_y, etc from Z (or metrics.𝒫)
function plot_risk(metrics; mean_y=0.036, var_y=0.02, cvar_y=0.01, α_y=0.017)
    gr() # TODO: Remove.

    Z = metrics.Z
    # P = normalize(fit(Histogram, metrics.Z)) # pdf
    plot_cost_distribution(Z)
    font_size = 11

    # Expected cost value
    𝔼 = metrics.mean
    plot!([𝔼, 𝔼], [0, mean_y], color="black", linewidth=2, label=nothing)
    annotate!([(𝔼, mean_y*1.04, text(L"\mathbb{E}[{\operatorname{cost}}]", font_size))])

    # Value at Risk (VaR)
    var = metrics.var
    plot!([var, var], [0, var_y], color="black", linewidth=2, label=nothing)
    annotate!([(var, var_y*1.08, text(L"\operatorname{VaR}", font_size))])

    # Worst case
    worst = metrics.worst
    plot!([worst, worst], [0, var_y], color="black", linewidth=2, label=nothing)
    annotate!([(worst, var_y*1.08, text(L"\operatorname{worst\ case}", font_size))])

    # Conditional Value at Risk (CVaR)
    cvar = metrics.cvar
    plot!([cvar, cvar], [0, cvar_y], color="black", linewidth=2, label=nothing)
    annotate!([(cvar, cvar_y*1.15, text(L"\operatorname{CVaR}", font_size))])

    # α failure probability threshold
    α_mid = (worst+var)/2
    plot!([α_mid, worst*0.985], [α_y, α_y], color="gray", linewidth=2, label=nothing, arrow=(:closed, 0.5))
    plot!([α_mid, var*1.015], [α_y, α_y], color="gray", linewidth=2, label=nothing, arrow=(:closed, 0.5))
    annotate!([(α_mid, α_y*1.1, text(L"\operatorname{top}\ (1-\alpha)\ \operatorname{quantile}", font_size-3))])

    zero_ylims()
    xlims!(xlims()[1], worst+0.1worst)
end


function plot_failure_distribution(𝒟, key1=:xpos_sut, key2=:ypos_sut)
    failure_samples_x = []
    failure_samples_y = []
    failure_samples_z = []
    nonfailure_samples_x = []
    nonfailure_samples_y = []
    nonfailure_samples_z = []

    USE_FIRST = false

    for d in 𝒟 # d = (𝐱, y)
        trajectory = d[1][1:end-1] # remove closure rate from 𝐱
        isevent = d[2]
        T = USE_FIRST ? 1 : length(trajectory)
        for t in 1:T
            # sample_x = first(trajectory).sample[key1] # TODO: for now, just look at the noise RIGHT BEFORE the failure.
            sample_x = trajectory[t].sample[key1]
            sample_x_value = sample_x.value
            sample_x_logprob = sample_x.logprob

            # sample_y = first(trajectory).sample[key2] # TODO: for now, just look at the noise RIGHT BEFORE the failure.
            sample_y = trajectory[t].sample[key2]
            sample_y_value = sample_y.value
            sample_y_logprob = sample_y.logprob
            if isevent
                push!(failure_samples_x, sample_x_value)
                push!(failure_samples_y, sample_y_value)
                push!(failure_samples_z, exp(sample_x_logprob + sample_y_logprob))
            else
                push!(nonfailure_samples_x, sample_x_value)
                push!(nonfailure_samples_y, sample_y_value)
                push!(nonfailure_samples_z, exp(sample_x_logprob + sample_y_logprob))
            end
        end
    end

    # histogram(failure_samples_x, label="y=1", alpha=0.5)
    # histogram!(nonfailure_samples_x, label="y=0", alpha=0.5)

    # histogram2d(failure_samples_x, failure_samples_y, label="y=1", alpha=0.5)
    # histogram2d!(nonfailure_samples_x, nonfailure_samples_y, label="y=0", alpha=0.5)

    # contourf(failure_samples_x, failure_samples_y, failure_samples_z, color=:viridis)
    # https://discourse.julialang.org/t/how-to-plot-a-multivariate-normal-distribution/35486/7
    # scatter(failure_samples_x, failure_samples_y, markersize=1000failure_samples_z)

    @show size(failure_samples_x)
    @show size(nonfailure_samples_x)

    scatter(failure_samples_x, failure_samples_y, label="y=1", alpha=0.2, markersize=10)
    scatter!(nonfailure_samples_x, nonfailure_samples_y, label="y=0", alpha=0.2)

    # return failure_samples_x, failure_samples_y, failure_samples_z
end


# Severity of failure
function plot_closure_rate_distribution(𝒟, show_nonfailures=true; reuse=true)
    # Full distribution of closure rates (i.e., failure or not)
    # histogram([d[1][end] for d in 𝒟], normalize=:probability)
    gr()

    failure_cr = cost_data(𝒟)
    nonfailure_cr = cost_data(𝒟, nonfailures=true)

    if (show_nonfailures && !isempty(nonfailure_cr))
        histogram(nonfailure_cr, alpha=0.5, label="non-failure", normalize=:pdf, color=NFCOLOR)
    end
    histogram!(failure_cr, alpha=0.5, label="failure", normalize=:pdf, reuse=reuse, color=FCOLOR)

    # Fit gamma distributions (only on failures, b/c rate will be non-negative)
    G_fail = fit(Gamma, failure_cr)
    p = plot!(x->pdf(G_fail,x), xlim=xlims(), label="fit (failure)", color="crimson", linewidth=2)

    ylabel!("frequency")
    xlabel!("closure rate (severity)")
    zero_ylims()
    return p
end


"""
Estimate the probabiltiy of failure using the beta distribution.
""" # TODO: multiple SUT on the same plot.
function probability_of_failure_beta(planner)
    fm = failure_metrics(planner)
    n = fm.num_failures
    m = fm.num_terminals - n
    P = Beta(1 + n, 1 + m)
    μ = mean(P)
    σ = std(P)
    plot(x->pdf(P,x), xlim=(μ-8σ, μ+8σ), label="p(fail)")
end


function multi_plot(planner)
    𝒟 = planner.mdp.dataset
    closure_rate_distribution(𝒟; reuse=false)
    # probability_of_failure_beta(planner)
    metrics = RiskMetrics(cost_data(𝒟), 0.2)
    risk_plot(metrics; mean_y=0.3, var_y=0.22, cvar_y=0.15, α_y=0.2)
end


## Polar area plots
copyend(Y) = vcat(Y, Y[1]) # repeat first.

function collect_risk_metrics(metricsvec::Vector; weights=ones(4), label_spacing_for_risk_only=true)
    # Fix xticklabel spacing
    metricnames = ["mean(Z)", "VaR(Z)", "CVaR(Z)", "worst(Z)"]
    M = []
    for met in metricsvec
        subset_met = [met.mean, met.var, met.cvar, met.worst] .* weights
        push!(M, subset_met)
    end
    return (M, metricnames)
end

function collect_overall_metrics(plannervec::Vector; weights=ones(4+3), α=α)
    planner_metrics, risk_metric_names = collect_risk_metrics([RiskMetrics(cost_data(planner.mdp.dataset), α) for planner in plannervec]; weights=weights[1:4])
    for i in 1:length(plannervec)
        planner = plannervec[i]
        local_fm = failure_metrics(planner)
        # How easily were failures found? Higher percentage of episodes, worse (using 1st failure)
        ease_of_failure = (local_fm.num_terminals - local_fm.first_failure) / local_fm.num_terminals
        push!(planner_metrics[i], local_fm.failure_rate/100 * weights[5], ease_of_failure * weights[6], local_fm.highest_loglikelihood * weights[7])
    end
    M = planner_metrics
    metricnames = vcat(risk_metric_names, "fail-rate", "ease(fail)", "max(log p)")
    return M, metricnames
end


"""
Polar area plot for RiskMetrics only.
"""
function plot_risk_metrics(metricsvec::Vector, labels::Vector; weights=ones(4)) # Vector{<:RiskMetrics}
    M, metricnames = collect_risk_metrics(metricsvec; weights=weights, label_spacing_for_risk_only=false)
    p = plot_metrics(M, metricnames, labels)
    # Note, `PyPlot.title` necessary when passed back Plots.backend_object
    Plots.PyPlot.title("Risk metrics", fontdict=Dict("fontsize"=>18), pad=15)
    return p
end

"""
Polar area plot for combined RiskMetrics and FailureMetrics.
"""
function plot_overall_metrics(plannervec::Vector, labels::Vector; weights=ones(4+3), α)
    M, metricnames = collect_overall_metrics(plannervec; weights=weights, α=α)
    p = plot_metrics(M, metricnames, labels)
    # Note, `PyPlot.title` necessary when passed back Plots.backend_object
    Plots.PyPlot.title("Risk and failure metrics", fontdict=Dict("fontsize"=>18), pad=15)
    return p
end


function plot_metrics(metricvec::Vector, metricnames::Vector, labels::Vector)
    pyplot() # polar plots only work well with PyPlot
    use_latex_fonts() # publication worthy.
    n = length(metricvec)
    local p
    for i in 1:n
        Y = metricvec[i]
        p = metric_area_plot(Y, i, n, metricnames, labels[i]; first=(i==1))
    end

    # Adjust the polar xtick labels
    obj = Plots.backend_object(p)
    obj.axes[1].tick_params(pad=11)
    return obj
end


function metric_area_plot(Y, i, n, metricnames, label; first=false)
    X = range(0, stop=2π, length=length(Y)+1)
    color = get(ColorSchemes.viridis, 1 - i/n) # reverse

    _plot = first ? plot : plot!
    return _plot(X, Y, projection=:polar, c=color, m=color,
        fillrange=[0], fillalpha=0.5, label=label,
        xticks=(X, vcat(metricnames, "")), legend=:outertopright)
end


risk_area(M::Vector; weights=ones(4)) = [localarea(m) for m in collect_risk_metrics(M; weights=weights)[1]]
overall_area(M::Vector; weights=ones(4+3), α=α) = [localarea(m) for m in collect_overall_metrics(M; weights=weights, α=α)[1]]

function localarea(m::Vector)
    A = 0
    for i in 1:length(m)-1
        A += (m[i]+m[i+1])/2
    end
    return A
end

"""
Plot rate vs. distance with fitted multivariate Gaussians, and separete univariate Gaussians.
- `gdatype`: :lda or :qda
- `boundary`: :binary or :gradient
"""
function plot_multivariate_distance_and_rate(𝒟; gda=false, gdatype=:qda, share_cov=false, k=1, svm=true, return_predict=false, boundary=:binary, subplots=true)
    pp = Plots.PyPlot

    # Use LaTeX fonts for rendering PyPlots
    use_latex_fonts()

    if subplots
        pp.figure(figsize=[6.4, 4.8].*1.75)
    else
        pp.figure(figsize=[6.4, 4.8])
    end
    Z_fail = cost_data(𝒟)
    Z_nonfail = cost_data(𝒟; nonfailures=true)
    d_fail = distance_data(𝒟)
    d_nonfail = distance_data(𝒟; nonfailures=true)

    if subplots
        pp.subplot(2,2,3)
    end

    # fit multivariate Gaussians
    pos_data = [Z_fail d_fail]'
    neg_data = [Z_nonfail d_nonfail]'
    mv_fail = fit_mle(MvNormal, pos_data)
    mv_nonfail = fit_mle(MvNormal, neg_data)

    if gda
        if gdatype == :lda
            # LDA with shared covariances
            if k == 1 # which class k shared their covariance?
                mv_fail = MvNormal(mv_fail.μ, mv_nonfail.Σ) # shared covariance
            else
                mv_nonfail = MvNormal(mv_nonfail.μ, mv_fail.Σ) # shared covariance
            end
        end
         # TODO. Calculate these.
        dbX = range(-0.1, stop=0.8, length=1000)
        dbY = range(3, stop=8, length=1000)
        π₁ = π₂ = 0.5 # priors
        μ₁ = mv_fail.μ
        μ₂ = mv_nonfail.μ
        Σ₁ = mv_fail.Σ
        Σ₂ = mv_nonfail.Σ
        if gdatype == :qda
            if boundary == :binary
                # QDA: in the form for class k=1
                predictₖ = (x, μₖ, Σₖ, πₖ) -> -1/2*log(det(Σₖ)) - 1/2*(x - μₖ)'inv(Σₖ)*(x - μₖ) + log(πₖ)
                predict1 = x -> predictₖ(x, μ₁, Σ₁, π₁)
                predict2 = x -> predictₖ(x, μ₂, Σ₂, π₂)
                predict = x -> predict1(x) > predict2(x) ? 0 : 1
            elseif boundary == :gradient
                # QDA: zero-decision bounday
                predict = x -> (x - μ₁)'inv(Σ₁)*(x - μ₁) + log(det(Σ₁)) - (x - μ₂)'inv(Σ₂)*(x - μ₂) - log(det(Σ₂))
            else
                error("No `boundary` of $boundary")
            end
        elseif gdatype == :lda
            # LDA (with shared Σs)
            Σ = Σ₁ # doesn't matter which k we choose, covariance is copied/duplicated above.
            if boundary == :binary
                predictₖ = (x, μₖ, πₖ) -> x'inv(Σ)*μₖ - 1/2*μₖ'inv(Σ)*μₖ + log(πₖ)
                predict1 = x -> predictₖ(x, μ₁, π₁)
                predict2 = x -> predictₖ(x, μ₂, π₂)
                predict = x -> predict1(x) > predict2(x) ? 0 : 1
            elseif boundary == :gradient
                predict = x -> (x - μ₁)'inv(Σ)*(x - μ₁) - (x - μ₂)'inv(Σ)*(x - μ₂)
            else
                error("No `boundary` of $boundary")
            end
        else
            error("No `gdatype` of $gdatype")
        end
        dbZ = [predict([x,y]) for y in dbY, x in dbX] # Note x-y "for" ordering
        vmin = minimum(dbZ)
        vmax = maximum(dbZ)
        TwoSlopeNorm = pp.matplotlib.colors.TwoSlopeNorm
        if boundary == :binary
            # Decision boundary binary, thus is relative to each k class prediction.
            norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2, vmax=vmax)
        elseif boundary == :gradient
            # Decision boundary is at zero.
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        end
        # Colormap so that:
        #   red   = class 1 = failure
        #   green = class 2 = non-failure
        pp.contourf(dbX, dbY, dbZ, 100, cmap="RdYlGn", vmin=vmin, vmax=vmax, norm=norm)
    end

    pp.scatter(Z_nonfail, d_nonfail, label="non-failure", alpha=0.5,color=NFCOLOR, s=10, edgecolor="black")
    pp.scatter(Z_fail, d_fail, label="failure", alpha=0.5, color=FCOLOR, s=10, edgecolor="black")

    pp.xlabel("rate")
    pp.ylabel("distance")

    current_xlim = pp.xlim()
    current_ylim = pp.ylim()

    fX = range(-2, stop=maximum(Z_fail)*1.1, length=1000)
    fY = range(-2, stop=(gdatype==:lda) ? maximum(d_nonfail)*1.1 : maximum(d_fail)*1.1 , length=1000)
    fZ = [pdf(mv_fail, [x,y]) for y in fY, x in fX] # Note x-y "for" ordering

    pp.contour(fX, fY, fZ, alpha=0.75, cmap="plasma")

    nfX = range(-2, stop=maximum(Z_nonfail)*1.1, length=1000)
    nfY = range(-2, stop=maximum(d_nonfail)*1.1, length=1000)
    nfZ = [pdf(mv_nonfail, [x,y]) for y in nfY, x in nfX] # Note x-y "for" ordering

    pp.contour(nfX, nfY, nfZ, alpha=0.75, cmap="viridis")

    pp.xlim(current_xlim)
    pp.ylim(current_ylim)

    # 1D Gaussians
    if subplots
        pp.subplot(4,2,3)
        pp.hist(Z_fail, color=FCOLOR, density=true, alpha=0.5)
        pp.hist(Z_nonfail, color=NFCOLOR, density=true, alpha=0.5)
        normal_Z_fail = fit_mle(Normal, Z_fail) # NOTE: Gamma?
        normal_Z_nonfail = fit_mle(Normal, Z_nonfail)
        Z_current_xlim = pp.xlim()
        Z_current_ylim = pp.ylim()
        pp.plot(fY, [pdf(normal_Z_fail, x) for x in fY], color=FCOLOR)
        pp.plot(nfY, [pdf(normal_Z_nonfail, x) for x in nfY], color=NFCOLOR)
        pp.xlim(Z_current_xlim)
        pp.ylim(Z_current_ylim)
        pp.xticks([])

        pp.subplot(2,4,7)
        pp.hist(d_fail, orientation="horizontal", color=FCOLOR, density=true, alpha=0.5)
        pp.hist(d_nonfail, orientation="horizontal", color=NFCOLOR, density=true, alpha=0.5)
        normal_d_fail = fit_mle(Normal, d_fail) # NOTE: Gamma?
        normal_d_nonfail = fit_mle(Normal, d_nonfail)
        d_current_xlim = pp.xlim()
        d_current_ylim = pp.ylim()
        base = pp.gca().transData
        rot = pp.matplotlib.transforms.Affine2D().rotate_deg(90)
        pp.plot(fY, [-pdf(normal_d_fail, x) for x in fY], transform=rot+base, color=FCOLOR) # notice negative pdf to flip transformation
        pp.plot(nfY, [-pdf(normal_d_nonfail, x) for x in nfY], transform=rot+base, color=NFCOLOR) # notice negative pdf to flip transformation
        pp.xlim(d_current_xlim)
        pp.ylim(d_current_ylim)
        pp.yticks([])
    end

    # Boundary line calculated using support vector machines (SVMs)
    if svm
        @info "Running SVM..."
        svm_x = range(-0.1, stop=0.8, length=1000) # TODO. Calculate.

        w, b = compute_svm(pos_data, neg_data)
        svm_boundary = (x,w,b) -> (-w[1] * x .+ b)/w[2] # line of the decision boundary
        svm_y = svm_boundary(svm_x, w, b)
        svm_classify = x -> sign((-w'*x + b)/w[2]) > 0 ? 1 : 0

        if subplots
            pp.subplot(2,2,3)
        end
        pp.plot(svm_x, svm_y, label="SVM boundary", color="black")
    end

    analyze_fit(predict, svm_classify, pos_data, neg_data)

    pp.legend()
    pp.subplots_adjust(wspace=0.08, hspace=0.1)

    pp.savefig("d-Z-distribution-subplots.png")
    fig = pp.gcf()

    return return_predict ? (fig, predict) : fig
end



function analyze_fit(predict, svm_classify, pos_data, neg_data; boundary=:linear)
    svm_true_positives = sum([svm_classify(x) for x in eachcol(pos_data)]) / size(pos_data)[2]
    svm_true_negatives = sum([1-svm_classify(x) for x in eachcol(neg_data)]) / size(neg_data)[2]
    @show round(svm_true_positives, digits=4)
    @show round(svm_true_negatives, digits=4)

    qda_true_positives = sum([predict(x) > 0 ? 0 : 1 for x in eachcol(pos_data)]) / size(pos_data)[2]
    qda_true_negatives = sum([predict(x) <= 0 ? 0 : 1 for x in eachcol(neg_data)]) / size(neg_data)[2]
    @show round(qda_true_positives, digits=4)
    @show round(qda_true_negatives, digits=4)
end


# https://jump.dev/Convex.jl/v0.13.2/examples/general_examples/svm/
function compute_svm(pos_data, neg_data, solver=() -> SCS.Optimizer(verbose=0))
    # Create variables for the separating hyperplane w'*x = b.
    n = 2 # dimensionality of data
    C = 10 # inverse regularization parameter in the objective  w = Variable(n)
    w = Variable(n)
    b = Variable()
    # Form the objective.
    obj = sumsquares(w) + C*sum(max(1+b-w'*pos_data, 0)) + C*sum(max(1-b+w'*neg_data, 0))
    # Form and solve problem.
    problem = minimize(obj)
    solve!(problem, solver)
    return evaluate(w), evaluate(b)
end