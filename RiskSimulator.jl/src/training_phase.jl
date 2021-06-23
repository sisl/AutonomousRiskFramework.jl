function training_phase(simx::AutoRiskSim; seed=0, epochs=20)
    ##############################################################################
    # Generate initial samples
    ##############################################################################

    # Scenes based on current IDM
    Random.seed!(seed)
    BlackBox.initialize!(simx);
    d = copy(simx.disturbances)
    hr = HistoryRecorder(max_steps=simx.params.endtime)
    hist_noise = POMDPSimulators.simulate(hr, simx.problem, FunctionPolicy((s) -> d));
    idm_scenes = state_hist(hist_noise);

    # Sensor noises in the IDM scenes
    scenes = Vector{typeof(simx.state)}() 
    for epoch in 1:epochs
        for idm_scene in idm_scenes
            scene = copy(idm_scene)
            try
                noisy_scene!(scene, simx.roadway, sutid(simx.problem), true)
                push!(scenes, scene)
            catch err
                @warn err
            end
        end
    end


    ##############################################################################
    # Train MDN
    ##############################################################################
    feat, y = preprocess_data(1, scenes);
    net_params = MDNParams(batch_size=2, lr=1e-3);
    net = construct_mdn(net_params);
    train_nnet!(feat, y, net..., net_params);

    return net, net_params
end
