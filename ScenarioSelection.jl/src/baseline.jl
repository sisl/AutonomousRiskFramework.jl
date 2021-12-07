function random_baseline(bn)
    tmp_sample = rand(bn)
    # @show tmp_sample
    tmp_s = DecisionState(tmp_sample[:type],tmp_sample[:sut],tmp_sample[:adv], true, 0.0)
    return (tmp_s, eval_AST(tmp_s))
    # return (tmp_s, nothing)
end

function simple_random_baseline()
    tmp_s = SimpleState(rand(get_actions(SimpleState()), 5), true, 0.0)
    return (tmp_s, eval_simple_reward(tmp_s))
end