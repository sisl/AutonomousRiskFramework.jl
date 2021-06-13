# Ablation studies
using Revise
using RiskSimulator
using AutomotiveSimulator
using AutomotiveVisualization
using AdversarialDriving
using Random
using Distributions

AutomotiveVisualization.colortheme["background"] = colorant"white";
AutomotiveVisualization.set_render_mode(:fancy);

SEED = 1000
Random.seed!(SEED);

SEED; net, net_params = training_phase();

function change_noise_disturbance!(planner)
    σ0 = 1e-300
    σ = 5
    σᵥ = σ

    planner.mdp.sim.xposition_noise_veh = Normal(0,σ)
    planner.mdp.sim.yposition_noise_veh = Normal(0,σ)
    planner.mdp.sim.velocity_noise_veh = Normal(0,σᵥ)

    planner.mdp.sim.xposition_noise_sut = Normal(0,σ)
    planner.mdp.sim.yposition_noise_sut = Normal(0,σ)
    planner.mdp.sim.velocity_noise_sut = Normal(0,σᵥ)
end

@info SCENARIO

SC = CROSSWALK
scenario = get_scenario(SC)
scenario_string = get_scenario_string(SC)

# System 1: IDM
system = IntelligentDriverModel(v_des=12.0)
planner = setup_ast(net, net_params;
    sut=system, scenario=scenario, seed=SEED, nnobs=false)

if true
    change_noise_disturbance!(planner)
end

# Run AST.
search!(planner)
fail_metrics = failure_metrics(planner)
RiskSimulator.POMDPStressTesting.latex_metrics(fail_metrics)


# System 2: Princeton
system2 = PrincetonDriver(v_des=12.0)
planner2 = setup_ast(net, net_params;
    sut=system2, scenario=scenario, seed=SEED, nnobs=false)

if true
    change_noise_disturbance!(planner2)
end

# Run AST.
search!(planner2)
fail_metrics2 = failure_metrics(planner2)
RiskSimulator.POMDPStressTesting.latex_metrics(fail_metrics2)


## Risk assessment
α = 0.2 # Risk tolerance.
𝒟 = planner.mdp.dataset

# Plot cost distribution.
metrics = risk_assessment(𝒟, α)
p_risk = plot_risk(metrics; mean_y=3.33, var_y=3.25, cvar_y=2.1, α_y=2.8)


𝐰 = ones(7)
𝐰[end] = 1e7/2
areas = overall_area([planner, planner2], weights=𝐰, α=α)
area_idm = round(areas[1], digits=5)
area_princeton = round(areas[2], digits=5)
p_metrics = plot_overall_metrics([planner, planner2],
    ["IDM ($area_idm)", "Princeton ($area_princeton)"]; 
    weights=𝐰, α=α, title="Overall Risk:\n$scenario_string")


# TODO. `roadway`
function AutomotiveVisualization.add_renderable!(rendermodel::RenderModel, veh::Entity{BlinkerState, VehicleDef, Int64})
    reg_veh = Entity(veh.state.veh_state, veh.def, veh.id)
    add_renderable!(rendermodel, FancyCar(car=reg_veh))

    noisy_veh = Entity(noisy_entity(veh, scenario.roadway).state.veh_state, veh.def, veh.id)
    ghost_color = weighted_color_mean(0.3, colorant"blue", colorant"white")
    add_renderable!(rendermodel, FancyCar(car=noisy_veh, color=ghost_color))

    li = laneid(veh)
    bo = BlinkerOverlay(on = blinker(veh), veh = reg_veh, right=Tint_signal_right[li])
    add_renderable!(rendermodel, bo)
    return rendermodel
end


viz = false
viz && fail_metrics.num_failures > 0 && visualize_most_likely_failure(planner)