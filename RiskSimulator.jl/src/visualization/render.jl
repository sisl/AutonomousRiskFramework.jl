using AdversarialDriving
using AutomotiveVisualization
using POMDPPolicies
using POMDPSimulators
using Reel
using RiskSimulator


# Creates a GIF of a simulation run with the given scenario type and initial conditions.
# Examples:
#   render_gif("animation.gif", MERGING, 20.48, 26.91, 34.72, 14.46)
#   render_gif("animation.gif", STOPPING, 3.20, 12.50, 36.72, 1.83)
function render_gif(filename::String, scenario_type::SCENARIO, s_sut::Float64, s_adv::Float64, v_sut::Float64, v_adv::Float64)
    scenario = get_scenario(scenario_type, s_sut=s_sut, s_adv=s_adv, v_sut=v_sut, v_adv=v_adv)
    roadway = scenario.roadway

    mdp = AdversarialDrivingMDP(scenario.sut, [scenario.adversary], roadway, 0.1)

    disturbances = scenario.disturbances

    hist = POMDPSimulators.simulate(HistoryRecorder(), mdp, FunctionPolicy((s) -> disturbances))
    scenes = state_hist(hist)

    timestep = 0.5
    nticks = length(scenes)
    animation = roll(fps=1.0/timestep, duration=nticks*timestep) do t, dt
        i = Int(floor(t/dt)) + 1
        AutomotiveVisualization.render([roadway, scenes[i]], canvas_height=240)
    end

    write(filename, animation)
end
