using POMDPs
using POMDPGym
using PyCall
pyimport("adv_carla")

mdp = GymPOMDP(Symbol("adv-carla"))
env = mdp.env
# initial_obs = reset!(env) # rest already called during the GymPOMDP constructor
σ = 12 # noise variance
while !env.done
    action = σ*rand(2)
    reward, obs, info = step!(env, action)
    render(env)
end
close(env)
println("Total reward of $(env.total_reward)")
