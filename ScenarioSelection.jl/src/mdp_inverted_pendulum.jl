# Basic MDP
mdp = InvertedPendulumMDP(λcost=0, include_time_in_state=true)

# Learn a policy that solves it
policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))), [0f0]), 
                     ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))))
policy = solve(PPO(π=policy, S=state_space(mdp), N=20000, ΔN=400), mdp)

# Define the disturbance distribution based on a normal distribution
xnom = Normal(0f0, 0.5f0)
xs = Float32[-2., -0.5, 0, 0.5, 2.]
ps = exp.([logpdf(xnom, x) for x in xs])
ps ./= sum(ps)
px = DiscreteNonParametric(xs, ps)

# Redefine disturbance to find action space
POMDPGym.disturbances(mdp::AdditiveAdversarialMDP) = support(mdp.x_distribution)
POMDPGym.disturbanceindex(mdp::AdditiveAdversarialMDP, x) = findfirst(support(mdp.x_distribution) .== x)

# Define the disturbance distribution based on a normal distribution
xnom = Normal(0f0, 0.5f0)
xs = Float32[-2., -0.5, 0, 0.5, 2.]
ps = exp.([logpdf(xnom, x) for x in xs])
ps ./= sum(ps)
px = DiscreteNonParametric(xs, ps)

# Construct the adversarial MDP to get access to a transition function like gen(mdp, s, a, x)
amdp = AdditiveAdversarialMDP(mdp, px)

# Construct the risk estimation mdp where actions are disturbances
rmdp = RMDP(amdp, policy, (m, s) -> 1 / (abs(s[1] - mdp.failure_thresh) + 1e-3))

