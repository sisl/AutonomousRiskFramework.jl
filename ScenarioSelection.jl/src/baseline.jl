function random_baseline(bn)
    tmp_sample = rand(bn)
    # @show tmp_sample
    tmp_s = DecisionState(tmp_sample[:type],tmp_sample[:sut],tmp_sample[:adv], true, 0.0)
    return (tmp_s, eval_AST(tmp_s))
    # return (tmp_s, nothing)
end

function simple_random_baseline()
    tmp_s = SimpleState(rand(Distributions.Categorical(5), 4), true, 0.0)
    return (tmp_s, sum(tmp_s.levels)/(length(tmp_s.levels)*5))
end