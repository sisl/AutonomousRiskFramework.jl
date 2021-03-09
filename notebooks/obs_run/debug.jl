# NNET
using Flux

# r_dist_net = ObservationModels.simple_distribution_fit(simx, simx.obs_noise[:ranges])

tot_x = 30
max_x = 40
min_x = 1
test_x = zeros(2, tot_x);
test_x[1, :] = min_x:(max_x-min_x+1)/tot_x:max_x
test_x[2, :] .= 0.0; 
# test_x = rand(2, 10000)

# test_x[1, :] .= 25.0
# test_x[2, :] = -10+2:2:30

test_x = ObservationModels.create_features(test_x)
ObservationModels.preprocess_data!(test_x)
test_x = test_x |> gpu
outs_π = r_dist_net["pi"](test_x)
outs_μ = r_dist_net["mu"](test_x)
outs_σ = r_dist_net["sigma"](test_x)

outs_π = outs_π |> cpu
outs_μ = outs_μ |> cpu
outs_σ = outs_σ |> cpu
# outs[1:2, :] = outs[1:2, :] .- test_x

ObservationModels.postprocess_data!(outs_π, outs_μ, outs_σ)

new_x = zeros(2, tot_x);
new_x[1, :] = min_x:(max_x-min_x+1)/tot_x:max_x
new_x[2, :] .= 0.0; 

# Mu
plot(new_x[1, :], outs_μ[1, :])
plot!(new_x[1, :], outs_μ[2, :])
plot!(new_x[1, :], outs_μ[3, :])
plot!(new_x[1, :], outs_μ[4, :])
plot!(ylims = (-5.0, 5.0))

# Pi
plot(new_x[1, :], outs_π[1, :])
plot!(new_x[1, :], outs_π[2, :])
plot!(new_x[1, :], outs_π[3, :])
plot!(new_x[1, :], outs_π[4, :])
plot!(ylims = (0, 1.0))

# Sigma
plot(new_x[1, :], outs_σ[1, :])
plot!(new_x[1, :], outs_σ[2, :])
plot!(new_x[1, :], outs_σ[3, :])
plot!(new_x[1, :], outs_σ[4, :])
plot!(ylims = (0.0, 5.0))






# Data Analysis
data_x, data_y = ObservationModels.gen_data(simx, simx.obs_noise[:ranges]);


# ObservationModels.preprocess_data!(data_x, data_y)

# scatter(data_x[1, :], data_y[1, :])
scatter(data_x[1, :], data_y[2, :])






# Positions
# BlackBox.initialize!(simx)
cont_noise = Noise(pos = (-15, 0), vel = 0)
noisy_action = Disturbance[BlinkerVehicleControl(noise=cont_noise), PedestrianControl()]
hist_noise = POMDPSimulators.simulate(HistoryRecorder(), ad_mdp, FunctionPolicy((s) -> noisy_action));
ad_scenes_noise = state_hist(hist_noise);
@show posg(ad_scenes_noise[end-10][1])









# Visualize
win = Blink.Window()

man = @manipulate for t=slider(1:length(ad_scenes_noise), value=1., label="t")
    AutomotiveVisualization.render([ad_mdp.roadway, crosswalk, ad_scenes_noise[t]])
end;
body!(win, man)