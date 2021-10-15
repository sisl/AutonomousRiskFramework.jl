using POMDPs
using Crux
using Flux
using Distributions
using POMDPGym
using PyCall
import POMDPPolicies:FunctionPolicy
pyimport("adv_carla")

mdp = GymPOMDP(Symbol("adv-carla"))
S = state_space(mdp)
rand_policy = FunctionPolicy((s) -> Float32.(rand.(Uniform.(amin, amax))))
amin, amax = [-3f0, -3f0], [3f0, 3f0]

QSA() = ContinuousNetwork(Chain(Dense(10, 64, relu), Dense(64, 64, relu), Dense(64, 2)))

function SAC_A()
    base = Chain(Dense(10, 64, relu), Dense(64, 64, relu))
    mu = ContinuousNetwork(Chain(base..., Dense(64, 2)))
    logΣ = ContinuousNetwork(Chain(base..., Dense(64, 2)))
    SquashedGaussianPolicy(mu, logΣ)
end

off_policy = (S=S,
              ΔN=50,
              N=100, # NOTE: was 30_000
              buffer_size=Int(5e5),
              buffer_init=1000,
              c_opt=(batch_size=100, optimizer=ADAM(1e-3)),
              a_opt=(batch_size=100, optimizer=ADAM(1e-3)),
              π_explore=FirstExplorePolicy(1000, rand_policy, GaussianNoiseExplorationPolicy(0.5f0, a_min=amin, a_max=amax)))

𝒮_sac = SAC(; π=ActorCritic(SAC_A(), DoubleNetwork(QSA(), QSA())), off_policy...)
@time π_sac = solve(𝒮_sac, mdp)
