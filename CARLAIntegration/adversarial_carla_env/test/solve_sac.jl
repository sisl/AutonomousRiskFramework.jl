using POMDPs
using Crux
using Flux
using Distributions
using POMDPGym
using PyCall
import POMDPPolicies:FunctionPolicy
pyimport("adv_carla")

sensors = [
    Dict(
        "id" => "GPS",
        "lat" => Dict("mean" => 0, "std" => 0.0001, "upper" => 10, "lower" => -10),
        "lon" => Dict("mean" => 0, "std" => 0.0001, "upper" => 10, "lower" => -10),
        "alt" => Dict("mean" => 0, "std" => 0.00000001, "upper" => 0.0000001, "lower" => 0),
    ),
]
seed = 2000
scenario_type = "Scenario4"
gym_args = (sensors=sensors, seed=seed, scenario_type=scenario_type)
mdp = GymPOMDP(Symbol("adv-carla"); gym_args...)
S = state_space(mdp)
amin = [Float32(sensors[1][k]["lower"]) for k in ["lat", "lon", "alt"]]
amax = [Float32(sensors[1][k]["upper"]) for k in ["lat", "lon", "alt"]]
# amin, amax = [-3f0, -3f0], [3f0, 3f0]
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))

state_dim = length(rand(initialstate(mdp)))
action_dim = length(amin)
QSA() = ContinuousNetwork(Chain(Dense(state_dim+action_dim, 64, relu), Dense(64, 64, relu), Dense(64, action_dim)))

function SAC_A()
    base = Chain(Dense(state_dim, 64, relu), Dense(64, 64, relu))
    mu = ContinuousNetwork(Chain(base..., Dense(64, action_dim)))
    logΣ = ContinuousNetwork(Chain(base..., Dense(64, action_dim)))
    SquashedGaussianPolicy(mu, logΣ)
end

off_policy = (S=S,
              ΔN=50,
              N=100, # NOTE: was 30_000
              buffer_size=Int(5e5),
              buffer_init=1000,
              c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
              a_opt=(batch_size=100, optimizer=ADAM(1e-3)),
              π_explore=FirstExplorePolicy(100, rand_policy, GaussianNoiseExplorationPolicy(0.5f0, a_min=amin, a_max=amax))) # NOTE: was 1000

𝒮_sac = SAC(; π=ActorCritic(SAC_A(), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_sac = solve(𝒮_sac, mdp)

# Run with SAC policy
mdp = GymPOMDP(Symbol("adv-carla"); gym_args...)
s = rand(initialstate(mdp))
total_reward = 0
while !isterminal(mdp, s)
    global s, total_reward
    a = action(π_sac, Vector(s)) # TODO: PyArray to Vector in POMDP.action(policy, s)
    s, o, r = gen(mdp, s, a)
    total_reward += r
end
close(mdp.env)

println("Total reward of $total_reward")
if mdp.env.info.get("collision")
    println("Found collision failure!")
end
