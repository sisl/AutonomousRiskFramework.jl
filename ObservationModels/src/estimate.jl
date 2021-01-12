@with_kw struct DistParams
    num_samples::Real = 50 # Estimation samples
end;

function gps_distribution_estimate(ent, noise_dist, params)
    
    base_meas = ObservationModels.measure_gps(ent, [0., 0., 0., 0., 0.])
    true_pos = posg(ent)
    velocity_noise = last(noise_dist)

    function one_sample(idx)
        function one_measurement(j)
            meas = base_meas[j]
            if typeof(meas)==Missing
                missing
            else
                GPSRangeMeasurement(sat=meas.sat, 
                                    range=meas.range, 
                                    noise=rand(noise_dist[j]))
            end
        end
        
        temp_meas = map(one_measurement, 1:length(base_meas))
        # temp_meas = Array{Union{Missing, GPSRangeMeasurement}}(undef, length(base_meas))

        # @distributed for j = 1:length(base_meas)
        #     temp_meas[j] = one_measurement(j)
        # end

        gps_fix = GPS_fix(temp_meas)
        noise = [gps_fix[1]-true_pos.x, gps_fix[2]-true_pos.y]
        noise
    end

    samples = map(one_sample, 1:params.num_samples)
    
    # samples = Array{Float64}(undef, 2, params.num_samples)
    # @distributed for i=1:params.num_samples
    #     samples[:, i] = one_sample(i)
    # end

    samples = hcat(samples...)

    xpos_samp = samples[1, :]
    ypos_samp = samples[2, :]

    # @show xpos_samp, ypos_samp
    # throw("Hi")

    
    xposition_noise = Distributions.fit(Normal{typeof(xpos_samp[1])}, xpos_samp, ones(length(xpos_samp)))
    yposition_noise = Distributions.fit(Normal{typeof(ypos_samp[1])}, ypos_samp, ones(length(ypos_samp)))

    return xposition_noise, yposition_noise, velocity_noise
end

function Base.rand(rng::AbstractRNG, d::Dict{Symbol, Vector{Sampleable}})
    Dict(k => rand.(Ref(rng), d[k]) for k in keys(d))
end

function Distributions.fit(d::Dict{Symbol, Vector{Sampleable}}, samples, weights; add_entropy = (x) -> x)
    N = length(samples)
    new_d = Dict{Symbol, Vector{Sampleable}}()
    for s in keys(d)
        dtype = typeof(d[s][1])
        m = length(d[s])
        new_d[s] = [add_entropy(fit(dtype, [samples[j][s][i] for j=1:N], weights)) for i=1:m]
    end
    new_d
end

function Distributions.logpdf(d::Dict{Symbol, Vector{Sampleable}}, x, i)
    sum([logpdf(d[k][i], x[k][i]) for k in keys(d)])
end

function Distributions.logpdf(d::Dict{Symbol, Vector{Sampleable}}, x)
    sum([logpdf(d, x, i) for i=1:length(first(x)[2])])
end

# function gen_gps_meas(noise::Array{Float64}, )

function obs_ce_loss(ent, noisy_pose, samp)
    base_meas = measure_gps(ent, [0., 0., 0., 0., 0.])
    
    temp_meas = Union{Missing,GPSRangeMeasurement}[]
    for j=1:length(samp[:ranges])
        if typeof(base_meas[j])==Missing
            continue
        else
            push!(temp_meas, GPSRangeMeasurement(sat=base_meas[j].sat, 
                                range=base_meas[j].range, 
                                noise=samp[:ranges][j]))
        end
    end

    true_pos = posg(ent)
    gps_fix = GPS_fix(temp_meas)
    
    return (gps_fix[1]-true_pos.x - noisy_pose[1])*(gps_fix[1]-true_pos.x - noisy_pose[1]), (gps_fix[2]-true_pos.y - noisy_pose[2])*(gps_fix[2]-true_pos.y - noisy_pose[2])
end

function traj_logprob(noisy_states, true_states, sim)
    max_t = length(true_states)
    
    # Make the logprob sequential 
    is_dist = []
    logprobs = []

    for t=1:max_t
        new_is_dist::Dict{Symbol, Vector{Sampleable}} = Dict(
            :ranges => [INormal_GMM(d.μ, d.σ) for d in sim.obs_noise[:ranges]]
        )

        # @show new_is_dist, sim.obs_noise
        # throw("Hi")
        push!(is_dist, new_is_dist)
        ent = true_states[t][AdversarialDriving.id(sim.adversary)]

        loss = (sample)->obs_ce_loss(ent, [noisy_states[:xpos][t], noisy_states[:ypos][t]], sample)
        
        meas_samples = rand(MersenneTwister(), is_dist[t], sim.dist_params.num_samples)

        x_prob = 0.0
        y_prob = 0.0

        for i=1:sim.dist_params.num_samples
            (x_loss, y_loss) = loss(meas_samples[i])

            is_wt = exp(logpdf(is_dist[t], meas_samples[i]))
            if x_loss < 2.0 
                x_prob += is_wt
            end
            if y_loss < 2.0 
                y_prob += is_wt
            end 
        end
        # @show x_prob, y_prob
        # throw("Hi")
        temp = Dict(
            :xpos => log(x_prob) - log(sim.dist_params.num_samples),
            :ypos => log(y_prob) - log(sim.dist_params.num_samples),
            :vel => logpdf(Normal(0., 1.), noisy_states[:vel][t])
        )
        push!(logprobs, temp)
    end

    
    return logprobs
end

function preprocess_data!(data_x, data_y)
    data_y[:] = data_y /40 .+ 0.5
    data_x[:] = data_x /40
end

function preprocess_data!(data_x)
    data_x[:] = data_x /40
end

function postprocess_data!(outs)
    sig_offset = Int(size(outs)[1]/2)
    outs[1:sig_offset, :] = (outs[1:sig_offset, :] .- 0.5)*40
    outs[sig_offset+1:end, :] = (exp.(outs[sig_offset+1:end, :]/2))*40
    outs
end

function train_nnet_mse!(data_x, data_y, net; batch_size=1, lr=1f-2, n_epoch=10)
    randIdx = collect(1:1:size(data_y)[2])
    numBatches = round(Int, floor(size(data_y)[2] / batch_size))

    opt = ADAM()

    sqnorm(x) = sum(abs2, x)
    penalty() = sum(sqnorm, Flux.params(net))

    function nll_loss(x, y) 
        outs = net(x)
        @assert size(y)[1]*2==size(outs)[1]

        return Flux.mse(outs[1:2, :], y)
    end

    evalcb() = @show(nll_loss(data_x, data_y))

    for epoch in 1:n_epoch
        Random.shuffle!(randIdx);
        data = [(data_x[:, randIdx[(i-1)*batch_size+1:i*batch_size]], data_y[:, randIdx[(i-1)*batch_size+1:i*batch_size]]) for i in 1:numBatches];
        Flux.train!(nll_loss, Flux.params(net), data, opt, cb=Flux.throttle(evalcb, 5))
    end
end

function train_nnet!(data_x, data_y, net; batch_size=1, lr=1f-2, n_epoch=10)
    randIdx = collect(1:1:size(data_y)[2])
    numBatches = round(Int, floor(size(data_y)[2] / batch_size))

    opt = ADAM()

    sqnorm(x) = sum(abs2, x)
    penalty() = sum(sqnorm, Flux.params(net))

    function nll_loss(x, y) 
        outs = net(x)
        @assert size(y)[1]*2==size(outs)[1]

        sig_offset = Int(size(outs)[1]/2)
        μ = outs[1:sig_offset, :]
        logσ_2 = outs[sig_offset+1:end, :]
        prec = exp.(-logσ_2)
        Δ = μ .- y
        coef = (Δ.^2).*prec*Float32(-0.5)
        ret = (logσ_2 .- coef) .+ Float32((log(2*π)/2))
        return mean(ret)

        # function single_nll(outs, idx)
        #     sig_offset = Int(size(outs)[1]/2)
        #     μ = outs[idx, :]
        #     logσ_2 = outs[sig_offset + idx, :]
        #     prec = exp.(-logσ_2)
        #     Δ = μ .- y[idx]
        #     coef = ((Δ.^2).*prec)*Float32(-0.5) 
        #     ret = (logσ_2 - coef).+ Float32((log(2*π)/2))
        #     ret
        # end
        
        # return sum(single_nll(outs, 1) + single_nll(outs, 2))
    end

    # function nll_loss(x, y) 
    #     outs = net(x)
    #     @assert size(y)[1]*2==size(outs)[1]

    #     return Flux.mse(outs[1:2, :], y)
    # end

    evalcb() = @show(nll_loss(data_x, data_y))

    for epoch in 1:n_epoch
        Random.shuffle!(randIdx);
        data = [(data_x[:, randIdx[(i-1)*batch_size+1:i*batch_size]], data_y[:, randIdx[(i-1)*batch_size+1:i*batch_size]]) for i in 1:numBatches];
        Flux.train!(nll_loss, Flux.params(net), data, opt, cb=Flux.throttle(evalcb, 5))
    end
end

function gen_data(sim, noise_dist; n_data = 100, n_samp = 100)
    rng = MersenneTwister()

    data_x = rand(Float32, (2, n_data))

    # # Pedestrian positions
    # data_x[1, :] = 30*data_x[1, :] .+ 10.0       # scaling from 10 to 40
    # data_x[2, :] = 50*data_x[2, :] .- 10.0      # scaling from -10 to 40

    # Vehicle positions
    data_x[1, :] = 30*data_x[1, :] .+ 0.0       # scaling from 0 to 30
    data_x[2, :] .= 0.0                         # zero

    base_meas = Vector{Vector{Union{Missing, GPSRangeMeasurement}}}(undef, n_data)
    for i=1:n_data
        base_meas[i] = ObservationModels.measure_gps(data_x[:, i], Float32[0., 0., 0., 0., 0.])
    end

    function one_sample(pos, bm, noise_dist, rng)
        function one_measurement(meas, nd, rng)
            if typeof(meas)==Missing
                missing
            else
                GPSRangeMeasurement(sat=meas.sat, 
                                    range=meas.range, 
                                    noise=rand(rng, nd))
            end
        end
        
        temp_meas = Vector{Union{Missing, GPSRangeMeasurement}}(undef, length(noise_dist))
        for j in 1:length(noise_dist)
            temp_meas[j] = one_measurement(bm[j], noise_dist[j], rng)
        end
        
        gps_fix = GPS_fix(temp_meas)
        noise = [gps_fix[1]-pos[1], gps_fix[2]-pos[2]]
        noise
    end

    data_y = Array{Float32}(undef, 2, n_samp, n_data)
    for i in 1:n_data
        for j in 1:n_samp
            data_y[:, j, i] = one_sample(data_x[:, i], base_meas[i], noise_dist, rng)
        end
    end

    # Create input copies for each sample
    # Reshape dataset
    data_x = repeat(data_x, inner=(1,n_samp))
    data_y = reshape(data_y, (2, :))

    data_x, data_y
end

function simple_distribution_fit(sim, noise_dist)
    data_x, data_y = gen_data(sim, noise_dist)
    
    net = Chain(Dense(2, 100, relu), Dense(100, 500, relu), Dense(500, 4)) |> gpu    
    
    preprocess_data!(data_x, data_y)
    data_x = data_x |> gpu
    data_y = data_y |> gpu
    # Pretrain
    train_nnet_mse!(data_x, data_y, net; batch_size=100, lr=1f-2, n_epoch=100)
    train_nnet!(data_x, data_y, net; batch_size=100, lr=1f-2, n_epoch=100)

    return net
end

function set_logprob(sim, net, true_states, noisy_states)
    max_t = length(true_states)
    in_traj = zeros(Float32, 2, max_t)
    for t in 1:max_t
        # ent = true_states[t][AdversarialDriving.id(sim.adversary)]  # Pedestrian
        ent = true_states[t][AdversarialDriving.id(sim.sut)]  # Vehicle
        ent_pos = posg(ent)
        in_traj[1, t] = ent_pos.x
        in_traj[2, t] = ent_pos.y
    end
    preprocess_data!(in_traj)
    in_traj = in_traj |> gpu
    traj_dist = net(in_traj)
    traj_dist = traj_dist |> cpu
    postprocess_data!(traj_dist)

    logprobs = []

    ## Pedestrian
    # for t in 1:max_t
    #     temp = Dict(
    #             :xpos_sut => logpdf(Normal(0., 5.), noisy_states[:xpos_sut][t]),
    #             :ypos_sut => logpdf(Normal(0., 5.), noisy_states[:ypos_sut][t]),
    #             :xpos_ped => logpdf(Normal(traj_dist[1, t], traj_dist[3, t]), noisy_states[:xpos_ped][t]),
    #             :ypos_ped => logpdf(Normal(traj_dist[2, t], traj_dist[4, t]), noisy_states[:ypos_ped][t])
    #         )
    #     push!(logprobs, temp)
    # end

    # Vehicle
    for t in 1:max_t
        temp = Dict(
                :xpos_sut => logpdf(Normal(traj_dist[1, t], traj_dist[3, t]), noisy_states[:xpos_sut][t]),
                :ypos_sut => logpdf(Normal(traj_dist[2, t], traj_dist[4, t]), noisy_states[:ypos_sut][t]),
                :xpos_ped => logpdf(Normal(noisy_states[:xpos_sut][t], 1.), noisy_states[:xpos_ped][t]),
                :ypos_ped => logpdf(Normal(noisy_states[:ypos_sut][t], 1.), noisy_states[:ypos_ped][t])
            )
        push!(logprobs, temp)
    end
    return logprobs
end