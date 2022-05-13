using ImportanceWeightedRiskMetrics
using TreeImportanceSampling
using MCTS
using D3Trees
using BSON

include("mdp_carla.jl")

function disturbance(m::CARLAScenarioMDP, s::ScenarioState)
    xs = POMDPs.actions(m, s)
    return xs
end

function save_bson(planner, filename)
    @time BSON.@save filename planner
end

mdp = CARLAScenarioMDP(use_neat=true, apply_gnss_noise=false)
Random.seed!(mdp.seed) # Determinism

RESUME = false

tree_mdp = TreeMDP(mdp, 1.0, [], [], disturbance, "sum")
planner_filename = "planner.bson"
s0_tree_filename = "s0_tree.bson"

if RESUME
    planner = BSON.load(planner_filename)[:planner]
else
    N = 100 # number of samples drawn from the tree (NOTE: D3Tree will have N + number of children nodes)
    c = 0.0 # exploration bonus (NOTE: keep at 0)
    α = 0.1 # VaR/CVaR risk parameter
    s0_tree = TreeImportanceSampling.TreeState(rand(initialstate(mdp)))
    planner = TreeImportanceSampling.mcts_isdpw(tree_mdp; N, c, α)
end

tree_in_info = true
planner.solver.tree_in_info = tree_in_info # NOTE!
β = 0.01 # for picking equal to Monte Carlo strategy (if β=1 then exactly MC)
γ = 0.01 # for better estimate of VaR (γ=1 would give minimum variance estimate of VaR)

try
    global a
    global info
    global tis_output
    global costs
    global log_weights
    global mcts_data_tree

    a, info = action_info(planner, s0_tree; tree_in_info=tree_in_info, β, γ)
    
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
catch err
    @warn err
    save_bson(planner, planner_filename)
    save_bson(s0_tree, s0_tree_filename)
    # change N
end
