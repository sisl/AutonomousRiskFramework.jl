using ImportanceWeightedRiskMetrics
using TreeImportanceSampling
using MCTS
using D3Trees

include("mdp_carla.jl")

mdp = CARLAScenarioMDP()
Random.seed!(mdp.seed) # Determinism
s0 = rand(initialstate(mdp))
# s0 = ScenarioState("Scenario6", Dict(
#         :cloudiness             => 0.0,
#         :precipitation_deposits => 0.0,
#         :wetness                => 0.0,
#         :fog_density            => 0.0,
#         :wind_intensity         => 0.0,
#         :precipitation          => 0.0,
#         :sun_altitude_angle     => 90.0,
#         :sun_azimuth_angle      => 360.0,
#         :fog_distance           => 0.0))
# s0 = ScenarioState("Scenario6", nothing)
# rollout(mdp, s0)
# reward(mdp, s0, nothing)



function disturbance(m::CARLAScenarioMDP, s::ScenarioState)
    xs = POMDPs.actions(m, s)
    ps = normalize(ones(length(xs)), 1)
    px = GenericDiscreteNonParametric(xs, ps)
    return px
end

tree_mdp = TreeMDP(mdp, 1.0, [], [], disturbance, "sum")
N = 1000 # number of samples drawn from the tree (NOTE: D3Tree will have N + number of children nodes)
c = 0.0 # exploration bonus (NOTE: keep at 0)
α = 0.1 # VaR/CVaR risk parameter
planner = TreeImportanceSampling.mcts_isdpw(tree_mdp; N, c, α)
tree_in_info = true
planner.solver.tree_in_info = tree_in_info # NOTE!
β = 0.01 # for picking equal to Monte Carlo strategy (if β=1 then exactly MC)
γ = 0.01 # for better estimate of VaR (γ=1 would give minimum variance estimate of VaR)
a, info = action_info(planner, TreeImportanceSampling.TreeState(s0); tree_in_info=tree_in_info, β, γ)

tis_output = (planner.mdp.costs, [], planner.mdp.IS_weights, info[:tree])
## look at tis_output[1] (cost), tis_output[2] (log. weights)
costs = tis_output[1]
log_weights = tis_output[2]
mcts_data_tree = tis_output[end]



#######################################
# Visualize tree if available
########################################
visualize_tree = true
if visualize_tree
    tree = mcts_data_tree.dpw_tree;
    t = D3Tree(tree, init_expand=1);
    for i in 1:length(t.text)
        if t.text[i][1:4] == "JLD2"
            t.text[i] = split(t.text[i], "\n")[end-1]
        end
    end
    inchrome(t)
end


#######################################
# Estimated CDF and Conditional CDF
########################################
# if visualize_tree
#     cdf_est = mcts_data_tree.cdf_est;
#     conditional_cdf_est = mcts_data_tree.conditional_cdf_est;
#     println(ImportanceWeightedRiskMetrics.quantile(cdf_est, 1e-1))
#     println(ImportanceWeightedRiskMetrics.quantile(cdf_est, 1e-2))
#     println(ImportanceWeightedRiskMetrics.quantile(cdf_est, 1e-3))
#     println(ImportanceWeightedRiskMetrics.quantile(cdf_est, 1e-4))
#     println(ImportanceWeightedRiskMetrics.quantile(cdf_est, 1e-5))
# end
