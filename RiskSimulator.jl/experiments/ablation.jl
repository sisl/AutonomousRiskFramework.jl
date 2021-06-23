# Ablation studies
using Revise
using RiskSimulator
using IntelligentDriving
using POMDPStressTesting
using AutomotiveSimulator
using AutomotiveVisualization
using AdversarialDriving
using Random
using Distributions
using JLD


AutomotiveVisualization.colortheme["background"] = colorant"white";
AutomotiveVisualization.set_render_mode(:fancy);

SEED = 1000
Random.seed!(SEED);

PLANNERS = Dict()

function change_noise_disturbance!(sim, scenario_key)
    σ0 = 1e-300

    # Scenario specific noise
    if scenario_key == CROSSING 
        σ = 10
        σᵥ = 2
    elseif scenario_key == T_HEAD_ON
        σ = 10
        σᵥ = 2
    elseif scenario_key == T_LEFT
        σ = 10
        σᵥ = 2
    elseif scenario_key == STOPPING
        σ = 2
        σᵥ = 1e-4
    elseif scenario_key == MERGING
        σ = 3
        σᵥ = 1
    elseif scenario_key == CROSSWALK
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

# scenario_string = get_scenario_string(SC)

# Get fresh copies of the SUTs
SUT = () -> IntelligentDriverModel(v_des=12.0)
SUT2 = () -> PrincetonDriver(v_des=12.0)
SUT3 = () -> begin
    obj_fn = IntelligentDriving.track_reference_avoid_others_obj_fn
    m = MPCDriver(obj_fn, "lon_lane")
    set_desired_speed!(m, 12.0)
    return m
end


function run_learned_rollout_phase(system,
                                   scenario_key,
                                   seed,
                                   use_nn_obs_model,
                                   state_proxy,
                                   learned_solver,
                                   adjust_noise)
    @info "Running learned rollout phase..."
    scenario = get_scenario(scenario_key)
    learned_planner = setup_ast(sut=system, scenario=scenario, seed=seed,
        nnobs=use_nn_obs_model, state_proxy=state_proxy, which_solver=learned_solver,
        noise_adjustment=adjust_noise ? sim->change_noise_disturbance!(sim,scenario_key) : nothing)
    search!(learned_planner)
    learned_fail_metrics = failure_metrics(learned_planner)
    @show learned_fail_metrics
    learned_rollout = (mdp, s, d) -> ppo_rollout(mdp, s, d, learned_planner)
    return learned_rollout
end


function run_sut(system,
                  scenario_key,
                  seeds,
                  include_rate_reward,
                  use_nn_obs_model,
                  state_proxy,
                  which_solver,
                  adjust_noise,
                  use_learned_rollout)

    if use_learned_rollout
        learned_rollout = run_learned_rollout_phase(system, scenario_key, first(seeds), use_nn_obs_model, state_proxy, :ppo, adjust_noise)
    end

    scenario = get_scenario(scenario_key)

    failure_metrics_vector::Vector{FailureMetrics} = []
    planner = nothing
    for seed in seeds
        planner = setup_ast(sut=system, scenario=scenario, seed=seed,
            nnobs=use_nn_obs_model, state_proxy=state_proxy,
            which_solver=which_solver,
            noise_adjustment=adjust_noise ? sim->change_noise_disturbance!(sim,scenario_key) : nothing,
            rollout=use_learned_rollout ? learned_rollout : RiskSimulator.AST.rollout,
            use_potential_based_shaping=include_rate_reward
        )

        # Run AST.
        search!(planner)
        fail_metrics = failure_metrics(planner)
        @show fail_metrics
        push!(failure_metrics_vector, fail_metrics)

        # Save global planner.
        @info "Saving..."
        save_planner(planner, system, scenario_key, seed, use_nn_obs_model, state_proxy, which_solver, adjust_noise, use_learned_rollout)
    end
    # # Save last planner.
    # @info "Saving..."
    # save_planner(planner, system, scenario_key, seeds, use_nn_obs_model, state_proxy, which_solver, adjust_noise, use_learned_rollout)
    @info mean(failure_metrics_vector)

    latex = RiskSimulator.POMDPStressTesting.latex_metrics(mean(failure_metrics_vector), std(failure_metrics_vector))
    println(latex)
    return latex
end


function save_planner(planner,
                      system,
                      scenario_key,
                      seed,
                      use_nn_obs_model,
                      state_proxy,
                      which_solver,
                      adjust_noise,
                      use_learned_rollout)

    global PLANNERS
    planner_key = join([typeof(system), scenario_key, "seed$seed"], ".")
    PLANNERS[planner_key] = planner

    # metrics
    # filename = join([typeof(system), scenario_key, "seed$seed", "obs-$use_nn_obs_model", "state-proxy-$state_proxy", which_solver, "adjnoise-$adjust_noise", "rollout-$use_learned_rollout"], ".")
    # save("metrics.$filename.jld", "metrics", planner.mdp.metrics)

    # dataset
    # save("dataset.$filename.jld", "metrics", planner.mdp.dataset)
end
