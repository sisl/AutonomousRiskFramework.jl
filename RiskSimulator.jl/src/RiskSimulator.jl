module RiskSimulator # Naming? TODO: RiskAssessment?

# using IntelligentDriving
using ObservationModels
using AutomotiveSimulator
using AutomotiveVisualization
using AdversarialDriving
using POMDPStressTesting
using POMDPs, POMDPPolicies, POMDPSimulators
using Distributions
using Parameters
using Random
using WebIO
using Blink
using InteractiveUtils
using Interact

export Simulator,
       Vehicle,
       Agent,
       Sensor,
       Scenario,
       simulate,
       evaluate,

       UrbanIDM,
       noisy_scene!,
       # roadway, # TODO.
       buildingmap, # TODO.
       cross_left_to_right,
       cross_bottom_to_top,
       t_left_to_right,
       t_right_to_turn,
       t_bottom_to_turn_left,
       hw_behind,
       hw_stopping,
       cw_left_to_right,
       cw_pedestrian_walking_up,
       hw_straight,
       hw_merging,
       x_intersection,
       t_intersection,
       multi_lane_roadway,
       crosswalk_roadway,
       merging_roadway,
       get_XIDM_template,
       get_scenario,
       get_scenario_string,
       SCENARIO,
       CROSSING,
       T_HEAD_ON,
       T_LEFT,
       STOPPING,
       MERGING,
       CROSSWALK,
       scenario_t_head_on_turn,
       scenario_t_left_turn,
       scenario_hw_stopping,
       scenario_crossing,
       scenario_pedestrian_crosswalk,
       scenario_hw_merging,
       scenario_hw_merging_no_obs,

       AutoRiskSim,
       AutoRiskParams,
       setup_ast,

       training_phase,

       visualize_most_likely_failure,

       RiskMetrics,
       conditional_distr,
       VaR,
       CVaR,
       worst_case,
       risk_assessment,
       combine_datasets,
       collect_metrics,
       cost_data,
       distance_data,
       latex_metrics,
       inverse_max_likelihood,

       use_latex_fonts,
       plot_cost_distribution,
       plot_risk,
       plot_combined_cost,
       plot_failure_distribution,
       plot_closure_rate_distribution,
       probability_of_failure_beta,
       multi_plot,
       copyend,
       collect_risk_metrics,
       collect_overall_metrics,
       plot_risk_metrics,
       plot_polar_risk,
       plot_metrics,
       metric_area_plot,
       risk_area,
       overall_area,
       plot_multivariate_distance_and_rate,
       analyze_fit,
       compute_svm,

       ppo_rollout,

       # AutomotiveSimulator.jl
       IntelligentDriverModel,
       PrincetonDriver,

       # POMDPStressTesting.jl
       search!,
       get_action,
       failure_metrics,
       most_likely_failure,
       FailureMetrics

Random.seed!(4) # TODO: reconcile with AST.jl and MDN training.

include("systems/urban_idm.jl")
include("scenario.jl")
include("ast.jl")
include("training_phase.jl")
include("visualization/interactive.jl")
include("visualization/plotting.jl")
include("rollouts/ppo_rollout.jl")
include("risk_assessment.jl")


#=
struct SafetyValidationTask
    scenario
    disturbances
    system
end

@with_kw mutable struct Vehicle
    agent = nothing
    dynamics = nothing
end


@with_kw mutable struct Simulator
    vehicles::Vector{Vehicle} = Vehicle[]
    scenario::Scenario = Scenario()

    Simulator(vehicles::Vector{Vehicle}) = new(vehicles, Scenario())
    Simulator(vehicles::Vector{Vehicle}, scenario::Scenario) = new(vehicles, scenario)
end


"""
Full simulation with output data for plotting and analysis.
"""
function simulate(sim::Simulator)
    @info "Running simulation:\n$sim"
end


"""
Streamlined evaluation for aggregate metrics.
"""
function evaluate(sim::Simulator)
    @info "Running evaluation:\n$sim"
end
=#

end # module
