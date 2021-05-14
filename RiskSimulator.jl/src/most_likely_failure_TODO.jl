##############################################################################
"""
Evaluate Plan (Metrics)
"""

# episodic_figures(planner.mdp, gui=true); POMDPStressTesting.gcf();
# distribution_figures(planner.mdp, gui=true); POMDPStressTesting.gcf();

playback_trace = playback(planner, action_trace, BlackBox.distance, return_trace=true);
failure_rate = print_metrics(planner).failure_rate


function most_likely_failure(planner)
    # TODO: Move into AST.jl package itself.
    # TODO: get this index from the `trace` itself
    # findmax(planner.mdp.metrics.reward[planner.mdp.metrics.event])
    # findmax(ast_mdp.metrics.reward[ast_mdp.metrics.event])

    failure_likelihood = NaN
    if any(planner.mdp.metrics.event)
        # Failures were found.
        failure_likelihood = round(exp(maximum(planner.mdp.metrics.logprob[planner.mdp.metrics.event])), digits=8)
    end

    # Markdown.parse(string("\$\$p = ", failure_likelihood, "\$\$"))
    println("p = $failure_likelihood")
end
most_likely_failure(planner)
