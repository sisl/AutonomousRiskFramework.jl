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

function show_tree(planner::ISDPWPlanner)
    tree = planner.tree
    show_tree(tree)
end

function show_tree(tree)
    t = D3Tree(tree, init_expand=1);
    for i in 1:length(t.text)
        if t.text[i][1:4] == "JLD2"
            t.text[i] = split(t.text[i], "\n")[end-1]
        end
    end
    inchrome(t)
end


mdp = CARLAScenarioMDP(agent=NEAT)
Random.seed!(mdp.seed) # Determinism

RESUME = false
RUN_DETERMINISTIC_TEST = false

if RUN_DETERMINISTIC_TEST
    @info "Running deterministic test given a specific initial state and seed."
    mdp.seed = 12648476
    mdp.run_separate_process = true
    Random.seed!(mdp.seed) # Determinism
    s0 = ScenarioState("Scenario4", Dict(
        :cloudiness             => 0.0,
        :precipitation_deposits => 0.0,
        :wetness                => 0.0,
        :fog_density            => 0.0,
        :wind_intensity         => 0.0,
        :precipitation          => 0.0,
        :sun_altitude_angle     => 90.0,
        :sun_azimuth_angle      => 120.0,
        :fog_distance           => 0.0))
    rollout(mdp, s0)
    reward(mdp, s0, nothing)
else
    tree_mdp = TreeMDP(mdp, 1.0, [], [], disturbance, "sum")
    planner_filename = "planner.bson"
    s0_tree_filename = "s0_tree.bson"

    N = 100 # number of samples drawn from the tree
    if RESUME
        planner = BSON.load(planner_filename)[:planner]
        planner.solver.n_iterations = N - length(planner.mdp.costs) # remaining runs.
    else
        c = 0.0 # exploration bonus (NOTE: keep at 0)
        α = 0.1 # VaR/CVaR risk parameter
        s0 = rand(initialstate(mdp))
        s0_tree = TreeImportanceSampling.TreeState(s0)
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
            show_tree(planner)
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
end
