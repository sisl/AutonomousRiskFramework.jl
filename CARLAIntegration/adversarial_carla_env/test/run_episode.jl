using POMDPs
using POMDPGym
using PyCall
pyimport("adv_carla")

mdp = GymPOMDP(Symbol("adv-carla"))
env = mdp.env
obs = reset!(env)
while !env.done
    action = rand(2) # [0, 0]
    reward, obs, info = step!(env, action)
end
close(env)
println("Total reward of $(env.total_reward)")
