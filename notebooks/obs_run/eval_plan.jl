# out_is = planner.is_dist[:xpos_sut]

episodic_figures(planner.mdp, gui=true); POMDPStressTesting.gcf();

distribution_figures(planner.mdp, gui=true); POMDPStressTesting.gcf();

playback_trace = playback(planner, action_trace, BlackBox.distance, return_trace=true);

failure_rate = print_metrics(planner)

begin
    # TODO: get this index from the `trace` itself
    # findmax(planner.mdp.metrics.reward[planner.mdp.metrics.event])
    # findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])

    failure_likelihood =
        round(exp(maximum(planner.mdp.metrics.logprob[planner.mdp.metrics.event])), digits=8)

    Markdown.parse(string("\$\$p = ", failure_likelihood, "\$\$"))
end

playback_trace = playback(planner, action_trace, sim->sim.state, return_trace=true)

win = Blink.Window()

man = @manipulate for t=slider(1:length(playback_trace), value=1., label="t")
    AutomotiveVisualization.render([planner.mdp.sim.problem.roadway, crosswalk, playback_trace[min(t, length(playback_trace))]])
end;

body!(win, man)

