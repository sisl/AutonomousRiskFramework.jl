using Pkg
Pkg.activate("ast_obs")
using Distributed
using Markdown
using InteractiveUtils
using Interact
using SharedArrays
using LinearAlgebra
using Plots
using Statistics
using Distributions, Parameters, Random, Latexify
using AutomotiveSimulator, AutomotiveVisualization
using POMDPs, POMDPPolicies, POMDPSimulators
using WebIO
using Blink
using AdversarialDriving
using POMDPStressTesting
using CrossEntropyMethod
using Revise
using ObservationModels
# addprocs(4, topology=:master_worker)
# @everywhere using Pkg
# @everywhere Pkg.activate("ast_obs")
# @everywhere using AutomotiveSimulator
# @everywhere using POMDPStressTesting

Base.rand(rng::AbstractRNG, s::Scene) = s

include("notebooks/obs_run/viz.jl")

include("notebooks/obs_run/initialize.jl")

include("notebooks/obs_run/sim_setup.jl")

include("notebooks/obs_run/run_ast.jl")

include("notebooks/obs_run/eval_plan.jl")