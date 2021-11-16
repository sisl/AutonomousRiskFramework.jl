scenario_types = [T_HEAD_ON, T_LEFT, STOPPING, CROSSING, MERGING, CROSSWALK];
# scenario_types = [STOPPING];


function get_actions(parent, value)
    if parent === nothing
		return Distributions.Categorical(length(scenario_types))
    elseif parent == :type
        # @show parent, value
		options = get_scenario_options(scenario_types[value])
        range_s_sut = options["s_sut"]
        range_v_sut = options["v_sut"]
        actions = [
            Distributions.Uniform(range_s_sut[1], range_s_sut[2]),
            Distributions.Uniform(range_v_sut[1], range_v_sut[2]) 
                ]
        return product_distribution(actions)
    elseif parent == :sut
        # @show parent, value
		options = get_scenario_options(scenario_types[value])
        range_s_adv = options["s_adv"]
        range_v_adv = options["v_adv"]
        actions = [
            Distributions.Uniform(range_s_adv[1], range_s_adv[2]),
            Distributions.Uniform(range_v_adv[1], range_v_adv[2]) 
                ]
        return product_distribution(actions)
	end
end

function create_bayesnet()
    bn = BayesNet(); 
    push!(bn, StaticCPD(:type, get_actions(nothing, nothing)));
    push!(bn, CategoricalCPD(:sut, [:type], [length(scenario_types)], [get_actions(:type, x) for x in 1:length(scenario_types)]));
    push!(bn, CategoricalCPD(:adv, [:type], [length(scenario_types)], [get_actions(:sut, x) for x in 1:length(scenario_types)]));
    # @show rand(bn)
    return bn
end