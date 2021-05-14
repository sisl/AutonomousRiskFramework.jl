## Rollouts
# - **$$Q$$-rollout**: explore based on existing $$Q$$-values
# - **$$\epsilon$$-greedy rollout**: take random action with probability $$\epsilon$$, best action otherwise
# - **CEM-rollout**: use low-level CEM optimization approach to select rollout action
# - Gaussian process-based $$Q$$-function approximation
# - Neural network-based $$Q$$-function approximation
#     -  $$Q(d,a)$$ encoding instead of $$Q(s,a)$$

global final_is_distrs = Any[nothing]

# convert(Vector{GrayBox.Environment}, final_is_distrs, 29)
# is_dist_0 = convert(Dict{Symbol, Vector{Sampleable}}, GrayBox.environment(ast_mdp.sim), 10)
# samples = rand(Random.GLOBAL_RNG, is_dist_0, 10)
# losses_fn = (d, samples) -> [POMDPStressTesting.cem_losses(d, samples; mdp=ast_mdp, initstate=initialstate(ast_mdp))]
# losses = losses_fn(is_dist_0, samples)
function cem_rollout(mdp::ASTMDP, s::ASTState, d::Int64)
    USE_PRIOR = false
    cem_mdp = mdp # deepcopy(mdp)
    prev_top_k = cem_mdp.params.top_k
    q_value = 0

    if USE_PRIOR # already computed importance sampling distribution
        is_distrs = final_is_distrs[1] # TODO: put this in `mdp`
    else
        cem_solver = CEMSolver(n_iterations=10,
                               num_samples=20,
                               episode_length=d,
                               show_progress=false)
        cem_mdp.params.top_k = 0
        cem_planner = solve(cem_solver, cem_mdp)
        is_distrs = convert(Vector{GrayBox.Environment}, search!(cem_planner, s), d)
        global final_is_distrs[1] = is_distrs
    end

    USE_MEAN = true # use the mean of the importance sampling distr, instead of rand.

    AST.go_to_state(mdp, s) # Records trace through this call # TODO: `record=true`????

    for i in 1:length(is_distrs) # TODO: handle min(d, length) to select is_dist associated with `d`
        is_distr = is_distrs[1]
        if USE_MEAN
            sample = mean(is_distr)
        else
            sample = rand(is_distr)
        end
        # @info sample
        # @info is_distr
        a::ASTAction = ASTSampleAction(sample)
        # a::ASTAction = ASTSampleAction(rand(GrayBox.environment(mdp.sim)))
        # AST.random_action(mdp)
        (s, r) = @gen(:sp, :r)(cem_mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + discount(cem_mdp)*q_value
        # AST.go_to_state(mdp, s) # Records trace through this call
    end
    # AST.go_to_state(mdp, s) # Records trace through this call
    cem_mdp.params.top_k = prev_top_k

    return q_value
end


D = Dict{Symbol,Distributions.Sampleable}(:vel => Distributions.Normal{Float64}(0.16191185557003204, 0.00010103246108517094),:xpos => Distributions.Normal{Float64}(-7.717689089890023, 5.7750315962668e-5),:ypos => Distributions.Normal{Float64}(0.8894044320100376, 3.3435841468310024e-6))

function Statistics.mean(d::Dict)
    meand = Dict()
    for k in keys(d)
        m = mean(d[k])
        meand[k] = GrayBox.Sample(m, logpdf(d[k], m))
    end
    return meand
end

@info logpdf(D[:vel], 0.16191185557003204)
@info rand(D)
@info mean(D)


# ╔═╡ 92e3e160-249f-11eb-0d10-c3c67a74428e
function ϵ_rollout(mdp::ASTMDP, s::ASTState, d::Int64; ϵ=0.5)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        if rand() < ϵ
            a::ASTAction = AST.random_action(mdp)
        else
            a = ASTSampleAction(BEST_ACTION.a)
        end

        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + AST.discount(mdp)*ϵ_rollout(mdp, sp, d-1; ϵ=ϵ)

        return q_value
    end
end


global 𝒟 = Tuple{Tuple{Real,ASTAction}, Real}[]

begin
    x = [d for ((d,a), q) in 𝒟]
    y = [q for ((d,a), q) in 𝒟]
end

begin
    PyPlot.svg(false)
    clf()
    hist2D(x, y)
    xlabel(L"d")
    ylabel(L"Q")
    gcf()
end

function prior_rollout(mdp::ASTMDP, s::ASTState, d::Int64)
    if d == 0 || isterminal(mdp, s)
        return 0.0
    else
        a::ASTAction = AST.random_action(mdp)
        distance = BlackBox.distance(mdp.sim)

        (sp, r) = @gen(:sp, :r)(mdp, s, a, Random.GLOBAL_RNG)
        q_value = r + AST.discount(mdp)*prior_rollout(mdp, sp, d-1)

        push!(𝒟, ((distance, a), q_value))

        return q_value
    end
end


function AST.search!(planner::CEMPlanner, s::ASTState)
    mdp::ASTMDP = planner.mdp
    return action(planner, s)
end

function Base.convert(::Type{Vector{GrayBox.Environment}}, distr::Dict{Symbol, Vector{Sampleable}}, max_steps::Integer=1)
    env_vector = GrayBox.Environment[]
    for t in 1:max_steps
        env = GrayBox.Environment()
        for k in keys(distr)
            env[k] = distr[k][t]
        end
        push!(env_vector, env)
    end
    return env_vector::Vector{GrayBox.Environment}
end


## Cross-Entropy Surrogate Method
# TODO?


## Neural Network Q-Approximator
# - Use distance $d$ as a _state proxy_
# - Approximate $Q(d,a)$ using a neural network (DQN?)
# - Collect data: $\mathcal{D} = (d, a) \to Q$
# - Train network: input $d$ output action $a$ (DQN) or input $(d,a)$ output $Q$
@with_kw mutable struct Args
    α::Float64 = 3e-4      # learning rate
    epochs::Int = 20       # number of epochs
    device::Function = cpu # gpu or cpu device
    throttle::Int = 1      # throttle print every X seconds
end


model = Chain(Dense(1+6, 32), Dense(32, 1)) # d + |A| -> Q
# @info 𝒟[1][1][2].sample
