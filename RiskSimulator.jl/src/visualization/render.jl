using AdversarialDriving
using AutomotiveVisualization
using POMDPPolicies
using POMDPSimulators
using Reel
using RiskSimulator


# Creates a GIF of a simulation run with the given scenario type and initial conditions.
# Example: render_failure("animation.gif", STOPPING, s_sut=3.20, s_adv=12.50, v_sut=36.72, v_adv=1.83)
function render_gif(filename, scenario_type::SCENARIO, s_sut, s_adv, v_sut, v_adv)
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
        AutomotiveVisualization.render([roadway, scenes[i]], canvas_height=120)
    end

    write(filename, animation)
end
