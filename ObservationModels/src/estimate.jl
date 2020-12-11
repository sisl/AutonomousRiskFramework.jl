@with_kw struct DistParams
    num_samples::Real = 100 # Estimation samples
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
        push!(is_dist, sim.obs_noise)
        ent = true_states[t][AdversarialDriving.id(sim.adversary)]

        loss = (sample)->obs_ce_loss(ent, [noisy_states[:xpos][t], noisy_states[:ypos][t]], sample)
        
        meas_samples = rand(MersenneTwister(), is_dist[t], sim.dist_params.num_samples)

        x_prob = 0.0
        y_prob = 0.0

        for i=1:sim.dist_params.num_samples
            (x_loss, y_loss) = loss(meas_samples[i])

            if x_loss < 1.0 
                x_prob += 1.0
            end
            if y_loss < 1.0 
                y_prob += 1.0
            end 
        end
        temp = Dict(
            :xpos => log(x_prob) - log(sim.dist_params.num_samples),
            :ypos => log(y_prob) - log(sim.dist_params.num_samples),
            :vel => logpdf(Normal(0., 1.), noisy_states[:vel][t])
        )
        push!(logprobs, temp)
    end

    
    return logprobs
end