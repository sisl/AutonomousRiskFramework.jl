function random_baseline(bn)
    tmp_sample = rand(bn)
    # @show tmp_sample
    tmp_s = DecisionState(tmp_sample[:type],tmp_sample[:sut],tmp_sample[:adv], true)
    return (tmp_s, eval_AST(tmp_s))
    # return (tmp_s, nothing)
end