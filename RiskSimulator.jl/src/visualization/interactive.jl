##############################################################################
"""
Evaluate Plan (Interactive)
"""

function visualize_most_likely_failure(planner, buildingmap)
    failure_action_set = filter(d->d[2], planner.mdp.dataset)

    # [end-1] to remove closure rate from 𝐱 data
    # displayed_action_trace = convert(Vector{ASTAction}, failure_action_set[1][1][1:end-2]) # TODO: could be empty.
    displayed_action_trace = most_likely_failure(planner)
    playback_trace = playback(planner, displayed_action_trace, sim->sim.state, return_trace=true)

    man = @manipulate for t=slider(1:length(playback_trace), value=1., label="t")
        AutomotiveVisualization.render([planner.mdp.sim.problem.roadway, buildingmap, playback_trace[min(t, length(playback_trace))]])
    end;

    win = Blink.Window()
    body!(win, man)
end
