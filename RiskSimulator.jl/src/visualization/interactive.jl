##################################################
# Evaluate plan (interactive)
##################################################

function visualize_most_likely_failure(planner)
    displayed_action_trace = most_likely_failure(planner)
    playback_trace = playback(planner, displayed_action_trace, sim->sim.state, return_trace=true, verbose=false)

    visualize_failure(planner, playback_trace)
end


function visualize_failure(planner, playback_trace)
    man = @manipulate for t=slider(1:length(playback_trace), value=1., label="t")
        AutomotiveVisualization.render([planner.mdp.sim.problem.roadway, playback_trace[min(t, length(playback_trace))]])
    end

    win = Blink.Window()
    body!(win, man)
end
