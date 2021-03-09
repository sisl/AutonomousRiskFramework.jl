@with_kw struct DistParams
    num_samples::Real = 50 # Estimation samples
end;

stop_gradient(f) = f()
Zygote.@nograd stop_gradient

function gps_distribution_estimate(ent, noise_dist, params)
    
    base_meas = measure_gps(ent, [0., 0., 0., 0., 0.])
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

# Loss for AST controlling range error disturbances
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

# Function to create neural network features from environment state
function create_features(data_x)
    n_dim, n_samp = size(data_x)
    @assert n_dim==2
    features = zeros(4, n_samp)
    for i in 1:n_samp
        features[1, i] = data_x[1, i]   # x-coord
        features[2, i] = data_x[2, i]   # y-coord
        features[3, i] = sqrt(data_x[1, i]^2 + data_x[2, i]^2) # Polar mag
        features[4, i] = atan(data_x[2, i], data_x[1, i]) # Polar ang
        # features[5, i] = 0.707 * data_x[1, i] + 0.707 * data_x[2, i] # 45° rotation
        # features[6, i] = 0.866 * data_x[1, i] + 0.5 * data_x[2, i] # 30° x rotation
        # features[7, i] = 0.5 * data_x[1, i] + 0.866 * data_x[2, i] # 30° y rotation
        # features[8, i] = data_x[1, i]^2   # x-coord degree 2
        # features[9, i] = data_x[2, i]^2   # y-coord degree 2
        # features[10, i] = data_x[1, i]^3   # x-coord degree 3
        # features[11, i] = data_x[2, i]^3   # y-coord degree 3
        # features[12, i] = data_x[1, i]^4   # x-coord degree 3
        # features[13, i] = data_x[2, i]^4   # y-coord degree 3
    end
    return features
end

# Preprocess inputs and labels for training
function preprocess_data!(feat_x, data_y)
    data_y[:] = data_y
    preprocess_data!(feat_x)
end

# Preprocess inputs for testing
function preprocess_data!(feat_x)
    feat_x[:] = feat_x
    # if size(feat_x)[1]>4
    #     feat_x[4, :] = feat_x[4, :]*40/pi
    # end
    # if size(feat_x)[1]>=8
    #     feat_x[8:9, :] = feat_x[8:9, :]/40
    # end
    # if size(feat_x)[1]>=10
    #     feat_x[10:11, :] = feat_x[10:11, :]/(40*40)
    # end
    # if size(feat_x)[1]>=12
    #     feat_x[12:13, :] = feat_x[12:13, :]/(40*40*40)
    # end
end

# postprocess neural network outputs 
function postprocess_data!(pi, mu, sigma; n_comp=2)
    for i in 1:2
        pi[1 + (i-1)*n_comp:i*n_comp, :] = softmax(pi[1 + (i-1)*n_comp:i*n_comp, :])
    end 
    # outs[1:sig_offset, :] = (outs[1:sig_offset, :] .- 0.5)*40
    # outs[sig_offset+1:end, :] = (exp.(outs[sig_offset+1:end, :]/2))
end

# Train neural network using mean squared error
function train_nnet_mse!(data_x, data_y, net; batch_size=1, lr=1f-2, n_epoch=10)
    randIdx = collect(1:1:size(data_y)[2])
    numBatches = round(Int, floor(size(data_y)[2] / batch_size))

    opt = RMSProp(lr, 0.99)

    sqnorm(x) = sum(abs2, x)
    penalty() = sum(sqnorm, Flux.params(net))

    function mse_loss(x, y) 
        outs = net(x)
        # wts = σ.(outs[5:6, :])
        @assert size(y)[1]*2==size(outs)[1]

        return Flux.mse(outs[1:2, :], y) # + Float32(1e-3)*penalty()
    end

    evalcb() = @show(mse_loss(data_x, data_y))

    for epoch in 1:n_epoch
        Random.shuffle!(randIdx);
        data = [(data_x[:, randIdx[(i-1)*batch_size+1:i*batch_size]], data_y[:, randIdx[(i-1)*batch_size+1:i*batch_size]]) for i in 1:numBatches];
        Flux.train!(mse_loss, Flux.params(net), data, opt, cb=Flux.throttle(evalcb, 5))
    end
end

# Probability of gaussian distribution
function gaussian_distribution(y, μ, σ)
    result = 1 ./ ((sqrt(2π).*σ)).*exp.(-0.5((y .- μ)./σ).^2)
end;

# Train neural network using mixture density log likelihood
function train_nnet!(data_x, data_y, pi_net, mu, sigma; batch_size=1, lr=1f-2, n_epoch=10)
    randIdx = collect(1:1:size(data_y)[2])
    numBatches = round(Int, floor(size(data_y)[2] / batch_size))

    opt = RMSProp(lr, 0.99)

    # sqnorm(x) = sum(abs2, x)
    # penalty() = sum(sqnorm, Flux.params(net))

    function mdn_loss(x, y; n_comp=2)
        π_full = pi_net(x)
        σ_full = sigma(x)
        μ_full = mu(x)
        result = []
        for i in 1:2
            i_π = softmax(π_full[1 + (i-1)*n_comp:i*n_comp, :])
            i_σ = σ_full[1 + (i-1)*n_comp:i*n_comp, :]
            i_μ = μ_full[1 + (i-1)*n_comp:i*n_comp, :]
            i_result = i_π.*gaussian_distribution(y, i_μ, i_σ)
            i_result = sum(i_result, dims=1)
            result = push!(result, -log.(i_result))
        end
        return mean(result)
    end;

    evalcb() = @show(mdn_loss(data_x, data_y))

    local training_loss
    ps = Flux.params(pi_net, mu, sigma);
    for epoch in 1:n_epoch
        Random.shuffle!(randIdx);
        data = [(data_x[:, randIdx[(i-1)*batch_size+1:i*batch_size]], data_y[:, randIdx[(i-1)*batch_size+1:i*batch_size]]) for i in 1:numBatches];
        # epoch_loss = 0.0;
        # for batch in data
        #     gs = gradient(ps) do
        #         # forward
        #         pi_out = pi_net(batch[1])
        #         sigma_out = sigma(batch[1])
        #         mu_out = mu(batch[1])
        #         training_loss = mdn_loss(pi_out, sigma_out, mu_out, batch[2])
        #         return training_loss
        #       end
        #     epoch_loss += training_loss
        #     # backward
        #     Flux.update!(opt, ps, gs)
        # end
        # @show(epoch_loss/size(data)[1])
        Flux.train!(mdn_loss, ps, data, opt, cb=Flux.throttle(evalcb, 5))
    end
end

# helper function to generate training dataset by simulating measurements
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

# Function to generate data, construct and train a neural network
function simple_distribution_fit(sim, noise_dist)
    data_x, data_y = gen_data(sim, noise_dist)
    feat_x = create_features(data_x)
    n_feat = size(feat_x)[1]
    n_hidden = 20;
    z_h = Dense(n_feat, n_hidden, tanh)
    z_π = Dense(n_hidden, 2*2)
    z_σ = Dense(n_hidden, 2*2, exp)
    z_μ = Dense(n_hidden, 2*2)    

    pi_net = Chain(z_h, z_π) |> gpu
    sigma = Chain(z_h, z_σ) |> gpu
    mu = Chain(z_h, z_μ) |> gpu
    
    preprocess_data!(feat_x, data_y)
    feat_x = feat_x |> gpu
    data_y = data_y |> gpu
    # Pretrain
    # train_nnet_mse!(feat_x, data_y, net; batch_size=500, lr=1f-2, n_epoch=100)
    train_nnet!(feat_x, data_y, pi_net, mu, sigma; batch_size=100, lr=1f-2, n_epoch=10)
    return Dict("pi" => pi_net, "sigma" => sigma, "mu" => mu)
end

# Function to evaluate log probability during testing
function calc_logprobs(sim, r_dist_net, true_states, noisy_states)
    N, N_ent, max_t = size(true_states)    
    
    in_traj = zeros(Float32, 2, max_t, N)
    for i_samp in 1:N
        for t in 1:max_t
            # TODO: Some weird bug creating wrong ordering
            ent_pos = true_states[i_samp, 2, t]  # Vehicle
            # ent_pos = true_states[i_samp, 1, t]  # Pedestrian
            # ent_pos = posg(ent)
            in_traj[1, t, i_samp] = ent_pos.x
            in_traj[2, t, i_samp] = ent_pos.y
        end
    end
    # @show in_traj[:, 1:5, 1:5]
    in_traj = reshape(in_traj, (2, max_t*N))
    in_traj = create_features(in_traj)
    preprocess_data!(in_traj)
    in_traj = in_traj |> gpu
    
    outs_π = r_dist_net["pi"](in_traj)
    outs_μ = r_dist_net["mu"](in_traj)
    outs_σ = r_dist_net["sigma"](in_traj)
    
    outs_π = outs_π |> cpu
    outs_μ = outs_μ |> cpu
    outs_σ = outs_σ |> cpu
    postprocess_data!(outs_π, outs_μ, outs_σ)
    outs_π = reshape(outs_π, (4, max_t, N))
    outs_μ = reshape(outs_μ, (4, max_t, N))
    outs_σ = reshape(outs_σ, (4, max_t, N))
    # @assert all(dist_traj[3:4, :, :] .>= 0)
    
    logprobs_all = []

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
    for i_samp in 1:N
        logprobs = Dict(
                    :xpos_sut => Array{Float64}(undef, max_t),
                    :ypos_sut => Array{Float64}(undef, max_t),
                    :xpos_ped => Array{Float64}(undef, max_t),
                    :ypos_ped => Array{Float64}(undef, max_t)
                )
        for t in 1:max_t
            # if any(isnan, dist_traj[:, t, i_samp])
            #     dist_traj[:, t, i_samp] .= 0    
            # end
            
            logprobs[:xpos_sut][t] = sum(outs_π[1:2, t, i_samp].*gaussian_distribution(noisy_states[i_samp][:xpos_sut][t], outs_μ[1:2, t, i_samp], outs_σ[1:2, t, i_samp]))
            # logprobs[:xpos_sut][t] = logpdf(Normal(0.0, 1.0), noisy_states[i_samp][:xpos_sut][t])
            logprobs[:ypos_sut][t] = sum(outs_π[3:4, t, i_samp].*gaussian_distribution(noisy_states[i_samp][:ypos_sut][t], outs_μ[3:4, t, i_samp], outs_σ[3:4, t, i_samp]))
            # logprobs[:xpos_ped][t] = logpdf(Normal(noisy_states[i_samp][:xpos_sut][t], 1.), noisy_states[i_samp][:xpos_ped][t])
            # logprobs[:ypos_ped][t] = logpdf(Normal(noisy_states[i_samp][:ypos_sut][t], 1.), noisy_states[i_samp][:ypos_ped][t])
            logprobs[:xpos_ped][t] = logpdf(Normal(noisy_states[i_samp][:xpos_sut][t], 1.0), noisy_states[i_samp][:xpos_ped][t])
            logprobs[:ypos_ped][t] = logpdf(Normal(noisy_states[i_samp][:ypos_sut][t], 0.5*abs(true_states[i_samp, 2, t].y - true_states[i_samp, 1, t].y)), noisy_states[i_samp][:ypos_ped][t])

            # CONSTRAINTS
            # Car not exceed lane
            if abs(noisy_states[i_samp][:ypos_sut][t]) > 1.0
                logprobs[:ypos_sut][t] = -1000
            end
            # No large positive noise (replace with not crossing crosswalk constraint)
            if noisy_states[i_samp][:xpos_sut][t] > 3.0
                logprobs[:xpos_sut][t] = -1000
            end
            # Pedestrian on crosswalk if car too far
            if abs(noisy_states[i_samp][:xpos_sut][t]) > 2.0
                logprobs[:xpos_ped][t] = logpdf(Normal(0, 1.), noisy_states[i_samp][:xpos_ped][t])
            end
        end
        push!(logprobs_all, logprobs)
    end
    return logprobs_all
end