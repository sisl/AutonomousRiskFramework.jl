# NNET
using Flux

r_dist_net = ObservationModels.simple_distribution_fit(simx, simx.obs_noise[:ranges])

tot_x = 30
max_x = 40
min_x = 1
test_x = zeros(2, tot_x);
test_x[1, :] = min_x:(max_x-min_x+1)/tot_x:max_x
test_x[2, :] .= 0.0; 
# test_x[1, :] .= 25.0
# test_x[2, :] = -10+2:2:30

ObservationModels.preprocess_data!(test_x)

outs = r_dist_net(test_x |> gpu)

outs = outs |> cpu

ObservationModels.postprocess_data!(outs)

plot(test_x[1, :], outs[1, :])
plot!(test_x[1, :], outs[2, :])

plot(outs[3, :])
plot!(outs[4, :])






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