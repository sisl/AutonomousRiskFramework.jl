"""
Construct a Mixture Density Network and return individual networks for pi, mu and sigma
"""
function construct_mdn(params::MDNParams)
    z_h = Dense(params.in_dims, params.hidden_dims, tanh)
    z_π = Dense(params.hidden_dims, params.N_modes)
    z_σ = Dense(params.hidden_dims, params.out_dims*params.N_modes, exp)
    z_μ = Dense(params.hidden_dims, params.out_dims*params.N_modes)    

    pi = Chain(z_h, z_π, softmax)
    sigma = Chain(z_h, z_σ)
    mu = Chain(z_h, z_μ)

    (pi, mu, sigma)
end

"""
Train neural network using mixture density log likelihood
"""
function train_nnet!(feat::Array{T, 2}, data_y::Array{T, 2}, pi, mu, sigma, params::MDNParams) where T
    randIdx = collect(1:1:size(data_y)[2])
    numBatches = round(Int, floor(size(data_y)[2] / params.batch_size))

    opt = RMSProp(params.lr, params.momentum)
    
    function mdn_loss(x, y)
        π_full = pi(x)
        σ_full = sigma(x)
        μ_full = mu(x)
        μ_comps = [μ_full[((i-1)*params.out_dims + 1):(i*params.out_dims)] for i in 1:params.N_modes]
        σ_comps = [σ_full[((i-1)*params.out_dims + 1):(i*params.out_dims)] for i in 1:params.N_modes]

        logprobs = [log.(π_full)[i, :] .+ sum(log.(gaussian_distribution.(y, μ_comps[i], σ_comps[i])), dims=1)[1, :] for i in 1:params.N_modes]
        tot_logprob = logsumexp(hcat(logprobs...), dims=2) 

        return -mean(tot_logprob)
    end;

    evalcb() = @show(mdn_loss(feat, data_y))

    local training_loss
    ps = Flux.params(pi, mu, sigma);
    for epoch in 1:params.N_epoch
        Random.shuffle!(randIdx);
        data = [(feat[:, randIdx[(i-1)*params.batch_size+1:i*params.batch_size]], data_y[:, randIdx[(i-1)*params.batch_size+1:i*params.batch_size]]) for i in 1:numBatches];
        Flux.train!(mdn_loss, ps, data, opt, cb=Flux.throttle(evalcb, 5))
    end
end
