using POMDPs
using Crux
using Flux
using Distributions
using POMDPGym
using PyCall
using GaussianDiscriminantAnalysis
import POMDPPolicies:FunctionPolicy
pyimport("adv_carla")

sensors = [
    Dict(
        "id" => "GPS",
        "lat" => Dict("mean" => 0, "std" => 0.0000001, "upper" => 0.0000001, "lower" => -0.0000001),
        "lon" => Dict("mean" => 0, "std" => 0.0000001, "upper" => 0.0000001, "lower" => -0.0000001),
        "alt" => Dict("mean" => 0, "std" => 0.00000001, "upper" => 0.0000001, "lower" => 0),
    ),
]
# "td3_policy.bson" used seed=1, scenario_type="Scenario2"
seed = 1
scenario_type = "Scenario2"
gym_args = (sensors=sensors, seed=seed, scenario_type=scenario_type, no_rendering=false)
mdp = GymPOMDP(Symbol("adv-carla"); gym_args...)


function td3_solver(mdp)
    global sensors

    S = state_space(mdp)
    amin = [Float32(sensors[1][k]["lower"]) for k in ["lat", "lon", "alt"]]
    amax = [Float32(sensors[1][k]["upper"]) for k in ["lat", "lon", "alt"]]
    rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

    state_dim = first(S.dims)
    action_dim = length(amin)
    QSA() = ContinuousNetwork(Chain(Dense(state_dim+action_dim, 64, relu), Dense(64, 64, relu), Dense(64, 1)))

    TD3_A() = ContinuousNetwork(Chain(Dense(state_dim, 64, relu), Dense(64, 64, relu), Dense(64, action_dim), x -> x .* 0.0000001), action_dim)

    on_policy = ActorCritic(TD3_A(), DoubleNetwork(QSA(), QSA()))

    off_policy = (S=S,
                  ΔN=300,
                  N=3000, # NOTE: was 30_000 (then 100)
                  buffer_size=Int(5e5),
                  buffer_init=1,
                  c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
                  a_opt=(batch_size=100, optimizer=ADAM(1e-3)),
                  log=(fns=[],),
                  π_explore=GaussianNoiseExplorationPolicy(0.00000001f0))

    return TD3(; π=on_policy, off_policy...)
end


# pass (mdp, s) check isterminal(mdp, s)
function extract_info(info)
    if info["done"] == true
        d = info["distance"]
        r = info["rate"]
        y = info["collision"]
        data_point = ([r,d], y)
    else
        data_point = missing
    end
    return data_point
end


function run_solver(mdp)
    # 1. run AST with data collection ON (records distance and rate at terminal state)
    @info "Running AST to collect dataset..."
    mdp_info = InfoCollector(mdp, extract_info)
    @time π_train = solve(td3_solver(mdp_info), mdp_info)

    # 2. compile data set of ([d,r], y)
    dataset = mdp_info.dataset
    failure_rate = sum(map(last, dataset)) / length(dataset)
    @info "Failure rate: $failure_rate"

    return π_train, dataset
end


function run(mdp, π, show_render=false)
    s = rand(initialstate(mdp))
    total_reward = 0
    iteration = 0
    A = []
    while !isterminal(mdp, s)
        iteration += 1
        a = action(π, Vector(s))
        s, o, r = gen(mdp, s, a)
        total_reward += r
        if show_render
            if hasfield(typeof(mdp), :pomdp)
                render(mdp.pomdp.env)
            else
                render(mdp.env)
            end
        end
        @show iteration, a
        push!(A, a) # collect actions/noise
    end
    return total_reward, A
end

# policy, dataset = run_solver(mdp)