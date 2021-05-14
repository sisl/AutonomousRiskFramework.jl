module RiskSimulator # Naming? TODO: RiskAssessment?

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
       roadway, # TODO.
       buildingmap, # TODO.

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
       cost_data,
       distance_data,
       latex_metrics,

       use_latex_fonts,
       plot_cost_distribution,
       plot_risk,
       plot_failure_distribution,
       plot_closure_rate_distribution,
       probability_of_failure_beta,
       multi_plot,
       copyend,
       collect_risk_metrics,
       collect_overall_metrics,
       plot_risk_metrics,
       plot_overall_metrics,
       plot_metrics,
       metric_area_plot,
       risk_area,
       overall_area,
       plot_multivariate_distance_rate,
       analyze_fit,
       compute_svm,

       # AutomotiveSimulator.jl
       IntelligentDriverModel,
       PrincetonDriver,

       # POMDPStressTesting.jl
       search!,
       failure_metrics

Random.seed!(4) # TODO: reconcile with AST.jl and MDN training.

include("systems/urban_idm.jl")
include("scenario.jl")
include("ast.jl")
include("training_phase.jl")
include("visualization/interactive.jl")
include("visualization/plotting.jl")
include("risk_assessment.jl")


struct SafetyValidationTask
    scenario
    disturbances
    system
end

@with_kw mutable struct Vehicle
    agent = nothing
    dynamics = nothing
end

@with_kw mutable struct Scenario
    file = nothing
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

end # module
