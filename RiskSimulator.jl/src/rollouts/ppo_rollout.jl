function ppo_rollout(mdp::ASTMDP, s::ASTState, d::Int64, learned_planner::PPOPlanner)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        nn_state = GrayBox.state(mdp.sim) # state-proxy
        a::ASTAction = sample_action(learned_planner, nn_state)

        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(mdp)*ppo_rollout(mdp, sp, d-1, learned_planner)

        return q_value
    end
end
