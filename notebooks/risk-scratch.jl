using Distributed
@everywhere using GaussianMixtures

HTML("<br>"^50)

planner.mdp.metrics.logprob[planner.mdp.metrics.event];
sim = ast.AutoRiskSim(ast.AutomotiveSimulator.PrincetonDriver(v_des=10.0));
sim.sut.model.idm;

if false
    model_fit = GMM(2,1)
    em!(model_fit, reshape(Z, length(Z), 1))
    model_fit.μ
end


logpdf(action_trace) # TODO. different than `highest_loglikelihood_of_failure`



# GDA/LDA/QDA
# begin
#   π₁ = π₂ = 0.5 # prior
#   μ₁ = mv_fail.μ
#   μ₂ = mv_nonfail.μ
#   # Σ = mv_nonfail.Σ # TODO. Not shared!
#   Σ₁ = mv_fail.Σ
#   Σ₂ = mv_nonfail.Σ

#   # a₀ = -1/2 * (μ₁ + μ₂)' * inv(Σ)*(μ₁ - μ₂)
#   # aⱼ = inv(Σ)*(μ₁ - μ₂)
#   # predict(x) = a₀ + aⱼ⋅x
#   # predict(x) = 2(inv(Σ)*(μ₁ - μ₂))'x + (μ₂ - μ₁)'inv(Σ)*(μ₂ - μ₁)
#   # predict(x) = x'inv(Σ₁ - Σ₂)*x + 2*(inv(Σ₂)*μ₂ - inv(Σ₁)*μ₁)'x + 
#       # (μ₁'inv(Σ₁)*μ₁ - μ₂'inv(Σ₂)*μ₂) + log(det(Σ₁) / det(Σ₂)) + 2*log(0.5/0.5)
#   # predict(x) = x'*inv(Σ₁ - Σ₂)*x + 2*(inv(Σ₂)*μ₂- inv(Σ₁)*μ₁)'*x +
#   #   (μ₁'*inv(Σ₁)*μ₁ - μ₂'*inv(Σ₂)*μ₂) + log(det(Σ₁)/det(Σ₂))
#   predict(x) = -1/2 * x'inv(Σ₂)*x + x'inv(Σ₂)*μ₂ - 1/2*μ₂'inv(Σ₂)*μ₂ - 1/2*log(det(Σ₂)) + log(π₂)
#   predict_class(x) = predict(x) > 0 ? :failure : :nonfailure
# end

# begin
#   # dbZ = [[x,y]' * inv(mv_nonfail.Σ)*(mv_fail.μ - mv_nonfail.μ) for y in nfY, x in nfX] # Note x-y "for" ordering
#   dbX = range(0, stop=0.08, length=1000)
#   dbY = range(3, stop=8, length=1000)
#   dbZ = [predict([x,y]) for y in dbY, x in dbX] # Note x-y "for" ordering
# end;

# plotting.Plots.PyPlot.figure(); contourf(dbX, dbY, dbZ, 100, cmap="RdYlGn_r"); gcf()

# https://stats.stackexchange.com/questions/71489/three-versions-of-discriminant-analysis-differences-and-how-to-use-them
# https://stackoverflow.com/questions/25500541/matplotlib-bwr-colormap-always-centered-on-zero


#   # predict(x) = -1/2 * x'inv(Σ₂)*x + x'inv(Σ₂)*μ₂ -
#                # 1/2*μ₂'inv(Σ₂)*μ₂ - 1/2*log(det(Σ₂)) + log(π₂)
#   predict(x) = x'*inv(Σ₁ - Σ₂)*x + 2*(inv(Σ₂)*μ₂- inv(Σ₁)*μ₁)'*x +
#       (μ₁'*inv(Σ₁)*μ₁ - μ₂'*inv(Σ₂)*μ₂) + log(det(Σ₁)/det(Σ₂))
#   # predict(x) = 2*(inv(Σ)*(μ₂ - μ₁))'x + (μ₁ - μ₂)'inv(Σ)*(μ₁ - μ₂) + 2*log(π₂/π₁)
    
#   # LDA (with different Σs)
#   # predict(x) = 2*(inv(Σ)*(μ₂ - μ₁))'x + (μ₁ - μ₂)'inv(Σ)*(μ₁ - μ₂)
#   # predict(x) = μ₂'inv(Σ₂)*x - 1/2*μ₂'inv(Σ₂)*μ₂ + log(π₂)
#   # a₀ = -1/2 * (μ₁ + μ₂)' * inv(Σ)*(μ₁ - μ₂)
#   # aⱼ = inv(Σ)*(μ₁ - μ₂)
#   # predict(x) = a₀ + aⱼ⋅x



#LDA
            # WRONG?
            # Σ = Σ₂
            # predict = x -> 2*(inv(Σ)*(μ₂ - μ₁))'x + (μ₁ - μ₂)'inv(Σ)*(μ₁ - μ₂) + 2*log(π₂/π₁)


#QDA.
            # predictₖ = (x, μₖ, Σₖ, πₖ) -> x'inv(Σ₁ - Σ₂)*x + 2*(inv(Σ₂)*μ₂ - inv(Σ₁)*μ₁)'*x + (μ₁'inv(Σ₁)*μ₁ - μ₂'inv(Σ₂)*μ₂) + log(det(Σ₁)/det(Σ₂)) + 2*log(π₂/π₁)
            # predict = (x) -> x'inv(Σ₁ - Σ₂)*x + 2*(inv(Σ₂)*μ₂ - inv(Σ₁)*μ₁)'*x + (μ₁'inv(Σ₁)*μ₁ - μ₂'inv(Σ₂)*μ₂) + log(det(Σ₁)/det(Σ₂)) + 2*log(π₂/π₁)
            # predictₖ = (x, μₖ, Σₖ, πₖ) -> -1/2*log(det(Σₖ)) - 1/2*(x - μₖ)'inv(Σₖ)*(x - μₖ) + log(πₖ)
            # predict = x -> -1/2*x'inv(Σ₁)*x + x'*inv(Σ₁)*μ₁ - 1/2*μ₁'inv(Σ₁)*μ₁ - 1/2*log(det(Σ₁)) + log(π₁)
