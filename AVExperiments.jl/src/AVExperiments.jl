module AVExperiments

# Set up experiments with configurations
# Save results
    # costs
    # tree
    # planner
    # info!
    # (naming schemes)
    # sub-directories
# Generate figures
    # histogram of costs
    # re-weighted histogram
    # anything from info?
    # polar plots?
# [] TODO: monitor carla.exe, kick off `carla-start` if missing.
# [] TODO: periodically save planner/costs a la "Weights and Biases" monitoring

using Reexport
# @reexport using JLD
@reexport using BSON
@reexport using ImportanceWeightedRiskMetrics
@reexport using TreeImportanceSampling
@reexport using Random
@reexport using MCTS
@reexport using D3Trees
using POMDPPolicies
using ProgressMeter
using POMDPSimulators

include("mdp_carla.jl")
include("monitor.jl")

export
    NEAT,
    WorldOnRails,
    GNSS,
    ExperimentConfig,
    ExperimentResults,
    show_tree,
    CARLAScenarioMDP,
    GenericDiscreteNonParametric,
    Weather,
    Agent,
    run_carla_experiment,
    pyreload,
    generate_dirname!,
    load_data,
    get_costs


function disturbance(m::CARLAScenarioMDP, s::ScenarioState)
    xs = POMDPs.actions(m, s)
    return xs
end


show_tree(planner::ISDPWPlanner) = show_tree(planner.tree.dpw_tree)
function show_tree(tree)
    t = D3Tree(tree, init_expand=1);
    for i in 1:length(t.text)
        if t.text[i][1:4] == "JLD2"
            t.text[i] = split(t.text[i], "\n")[end-1]
        end
    end
    inchrome(t)
end


# save_data(data, filename) = save(filename, "data", data)
# load_data(filename) = load(filename)[:data]

save_data(data, filename) = BSON.@save(filename, data)
load_data(filename) = BSON.load(filename, @__MODULE__)[:data]



@with_kw mutable struct ExperimentConfig
    seed        = 0xC0FFEE  # RNG seed for determinism
    agent       = NEAT      # AV policy/agent to run. Options: [NEAT, WorldOnRails, GNSS]
    N           = 100       # Number of scenario selection iterations
    dir         = "results" # Directory to save results
    use_tree_is = true      # Use tree importance sampling (IS) for scenario selection (SS) [`false` will use Monte Carlo SS]
    leaf_noise  = true      # Apply adversarial noise disturbances at the leaf nodes
    resume      = false     # Resume previous run?
    additional  = true      # Resume experiment by running an additional N iterations (as opposed to "finishing" the remaining `N-length(results)`)
    rethrow     = false     # Choose to rethrow the errors or simply provide warning.
    retry       = true      # Restart the run if an error was encountered.
    monitor     = @task start_carla_monitor() # Task to monitor that CARLA is still running.
    render_carla = true     # Show CARLA rendered display.
    iterations_per_process = 3 # Number of runs to make in separate Julia process (due to CARLA memory leak).
    save_frequency = 5 # After X iterations, save results.
end


@with_kw mutable struct ExperimentResults
    planner
    costs
    info
end


function generate_dirname!(config::ExperimentConfig)
    dir = "results"
    if config.agent == WorldOnRails
        dir = "$(dir)_wor"
    elseif config.agent == NEAT
        dir = "$(dir)_neat"
    elseif config.agent == GNSS
        dir = "$(dir)_gnss"
    end

    if config.use_tree_is
        dir = "$(dir)_SS-IS"
    else
        dir = "$(dir)_SS-MC"
    end

    if config.leaf_noise
        dir = "$(dir)_leaf-MC"
    else
        dir = "$(dir)_leaf-none"
    end

    config.dir = dir
end


get_costs(results::Vector) = map(res->res.hist[end].r, results)
get_costs(planner::ISDPWPlanner) = planner.mdp.costs


function run_carla_experiment(config::ExperimentConfig)
    # Monitor that CARLA executable is still alive.
    if !istaskstarted(config.monitor)
        schedule(config.monitor) # Done asynchronously.
    end

    mdp = CARLAScenarioMDP(seed=config.seed,
                           agent=config.agent,
                           leaf_noise=config.leaf_noise,
                           render_carla=config.render_carla,
                           iterations_per_process=config.iterations_per_process)
    Random.seed!(mdp.seed) # Determinism

    !isdir(config.dir) && mkdir(config.dir)
    planner_filename = joinpath(config.dir, "planner.bson")

    N = config.N # number of samples drawn from the tree

    if config.use_tree_is
        tree_mdp = TreeMDP(mdp, 1.0, [], [], disturbance, "sum")
        s0 = rand(initialstate(mdp))
        s0_tree = TreeImportanceSampling.TreeState(s0)
        if config.resume
            @info "Resuming: $planner_filename"
            planner = load_data(planner_filename)
            if config.additional
                planner.solver.n_iterations = N
            else
                planner.solver.n_iterations = N - length(planner.mdp.costs)
            end
            @info "Resuming for N = $(planner.solver.n_iterations) runs."
        else
            c = 0.0 # exploration bonus (NOTE: keep at 0)
            α = mdp.α # VaR/CVaR risk parameter
            planner = TreeImportanceSampling.mcts_isdpw(tree_mdp; N, c, α)
        end

        tree_in_info = true
        planner.solver.tree_in_info = tree_in_info
        β = 0.01 # for picking equal to Monte Carlo strategy (if β=1 then exactly MC)
        γ = 0.01 # for better estimate of VaR (γ=1 would give minimum variance estimate of VaR)

        try
            save_callback = planner->save_data(planner, planner_filename)
            a, info = action_info(planner, s0_tree; tree_in_info=tree_in_info, save_frequency=config.save_frequency, save_callback=save_callback, β, γ)
            
            tis_output = (planner.mdp.costs, [], planner.mdp.IS_weights, info[:tree])
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
        catch err
            if config.rethrow
                rethrow(err)
            else
                @warn err
                if config.retry
                    @info "Retrying AV experiment!"
                    config.N = config.N - length(planner.mdp.costs)
                    config.resume = true
                    run_carla_experiment(config) # Retry if there was an error.
                end
            end
        end
        save_data(planner, planner_filename)
        return ExperimentResults(planner, planner.mdp.costs, [])
    else
        # Use Monte Carlo scenario selection instead of tree-IS.
        s0 = rand(initialstate(mdp))
        policy = RandomPolicy(mdp)
        results_filename = joinpath(config.dir, "random_scenario_results.bson")
        if config.resume
            @info "Resuming: $results_filename"
            results = load_data(results_filename)
            if !config.additional
                N = N - length(results)
            end
            @info "Resuming for N = $N runs."
        else
            results = []
        end

        if length(results) != 0
            Random.seed!(mdp.seed + length(results)) # Change seed to where we left off.
        end

        try
            @showprogress for i in 1:N
                res = POMDPSimulators.simulate(HistoryRecorder(), mdp, policy, s0)
                push!(results, res)
                if i % config.save_frequency == 0
                    save_data(results, results_filename)
                end
            end
        catch err
            if config.rethrow
                rethrow(err)
            else
                @warn err
                if config.retry
                    @info "Retrying AV experiment!"
                    config.N = config.N - length(results)
                    config.resume = true
                    run_carla_experiment(config) # Retry if there was an error.
                end
            end
        end

        save_data(results, results_filename)
        costs = get_costs(results)
        return ExperimentResults(policy, costs, results)
    end
end


end # module
