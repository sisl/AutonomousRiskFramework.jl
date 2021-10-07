using Revise
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

Random.seed!(4) # TODO: reconcile with AST.jl and MDN training.

include("../../RiskSimulator.jl/src/systems/urban_idm.jl")
include("../../RiskSimulator.jl/src/scenario.jl")
include("../../RiskSimulator.jl/src/ast.jl")
include("../../RiskSimulator.jl/src/training_phase.jl")
include("../../RiskSimulator.jl/src/visualization/interactive.jl")
include("../../RiskSimulator.jl/src/risk_assessment.jl")


##############################################################################
# (1) Observation Model Training Phase
# (2) Standard AST Phase.
# TODO: (3) Learned rollout phase.
##############################################################################
# net, net_params = training_phase()
# use_princeton = false
# system = use_princeton ? PrincetonDriver(v_des=12.0) : IntelligentDriverModel(v_des=12.0)
# planner = setup_ast(net, net_params; sut=system, seed=10);
# action_trace = search!(planner)
# visualize_most_likely_failure(planner, buildingmap)
