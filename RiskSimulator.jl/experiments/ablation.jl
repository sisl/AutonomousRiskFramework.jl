# Ablation studies
using Distributed

num_threads = 4
if nprocs() < num_threads
    addprocs(num_threads - nprocs())
end


@everywhere using Revise
@everywhere using RiskSimulator
@everywhere using POMDPStressTesting
@everywhere using AutomotiveSimulator
@everywhere using AutomotiveVisualization
@everywhere using AdversarialDriving
@everywhere using Random
@everywhere using Distributions
@everywhere using JLD

@everywhere begin

AutomotiveVisualization.colortheme["background"] = colorant"white";
AutomotiveVisualization.set_render_mode(:fancy);

SEED = 1000
Random.seed!(SEED);

SEEDS = 1:2 # TODO: 10.

function change_noise_disturbance!(sim)
    σ0 = 1e-300

    # Scenario specific noise
    if SC == CROSSING 
        σ = 1
        σᵥ = 1/10
    elseif SC == T_HEAD_ON
        σ = 10
        σᵥ = 4
    elseif SC == T_LEFT
        σ = 10
        σᵥ = 1
    elseif SC == STOPPING
        σ = 2
        σᵥ = σ/100
    elseif SC == MERGING
        σ = 2
        σᵥ = 1
    elseif SC == CROSSWALK
        σ = 2
        σᵥ = 1/10
    end
    
    sim.xposition_noise_veh = Normal(0, σ)
    sim.yposition_noise_veh = Normal(0, σ)
    sim.velocity_noise_veh = Normal(0, σᵥ)

    sim.xposition_noise_sut = Normal(0, σ)
    sim.yposition_noise_sut = Normal(0, σ)
    sim.velocity_noise_sut = Normal(0, σᵥ)
end

@show SCENARIO

SC = STOPPING
scenario = get_scenario(SC)
scenario_string = get_scenario_string(SC)

# state_proxy = :distance # :distance, :rate, :actual, :none
# which_solver = :mcts
# use_nn_obs_model = false
# adjust_noise = true
# learned_solver = :ppo
# use_learned_rollout = false


system = IntelligentDriverModel(v_des=12.0)
system2 = PrincetonDriver(v_des=12.0)


function run_learned_rollout_phase(system,
                                   scenario,
                                   seed,
                                   use_nn_obs_model,
                                   state_proxy,
                                   learned_solver,
                                   adjust_noise)
    @info "Running learned rollout phase..."
    learned_planner = setup_ast(sut=system, scenario=scenario, seed=seed,
        nnobs=use_nn_obs_model, state_proxy=state_proxy, which_solver=learned_solver,
        noise_adjustment=adjust_noise ? change_noise_disturbance! : nothing)
    search!(learned_planner)
    learned_fail_metrics = failure_metrics(learned_planner)
    @show learned_fail_metrics
    learned_rollout = (mdp, s, d) -> ppo_rollout(mdp, s, d, learned_planner)
    return learned_rollout
end


function run_sut(system,
                  scenario,
                  seeds,
                  include_rate_reward,
                  use_nn_obs_model,
                  state_proxy,
                  which_solver,
                  adjust_noise,
                  use_learned_rollout)

    if use_learned_rollout
        learned_rollout = run_learned_rollout_phase(system, scenario, first(seeds), use_nn_obs_model, state_proxy, :ppo, adjust_noise)
    end

    failure_metrics_vector::Vector{FailureMetrics} = []
    planner = nothing
    for seed in seeds
        planner = setup_ast(sut=system, scenario=scenario, seed=seed,
            nnobs=use_nn_obs_model, state_proxy=state_proxy,
            which_solver=which_solver,
            noise_adjustment=adjust_noise ? change_noise_disturbance! : nothing,
            rollout=use_learned_rollout ? learned_rollout : RiskSimulator.AST.rollout,
            use_potential_based_shaping=include_rate_reward
        )

        # Run AST.
        search!(planner)
        fail_metrics = failure_metrics(planner)
        push!(failure_metrics_vector, fail_metrics)

    end
    # Save last planner.
    @info "Saving..."
    # save_planner(planner, system, scenario, seeds, use_nn_obs_model, state_proxy, which_solver, adjust_noise, use_learned_rollout)
    @info mean(failure_metrics_vector)

    # TODO: Print to file.
    latex = RiskSimulator.POMDPStressTesting.latex_metrics(mean(failure_metrics_vector), std(failure_metrics_vector))
    println(latex)
    return latex
end


function save_planner(planner,
                      system,
                      scenario,
                      seed,
                      use_nn_obs_model,
                      state_proxy,
                      which_solver,
                      adjust_noise,
                      use_learned_rollout)

    # metrics
    filename = join([typeof(system), SC, "seed$seed", "obs-$use_nn_obs_model", "state-proxy-$state_proxy", which_solver, "adjnoise-$adjust_noise", "rollout-$use_learned_rollout"], ".")
    save("metrics.$filename.jld", "metrics", planner.mdp.metrics)

    # dataset
    save("dataset.$filename.jld", "metrics", planner.mdp.dataset)
end

end