using CSV
using DataFrames

function export_carla_script(planner, buildingmap; filename="carla_playback.csv", origin_offset=true)
    failure_action_set = filter(d->d[2], planner.mdp.dataset)

    displayed_action_trace = RiskSimulator.most_likely_failure(planner)
    playback_trace = RiskSimulator.playback(planner, displayed_action_trace, sim->sim.state, return_trace=true)

    # man = @manipulate for t=slider(1:length(playback_trace), value=1., label="t")
        # AutomotiveVisualization.render([planner.mdp.sim.problem.roadway, buildingmap, playback_trace[min(t, length(playback_trace))]])
    # end;

    # Header:
    # step, v_x_car, v_y_car, x_car, y_car, v_x_ped_0, v_y_ped_0, x_ped_0, y_ped_0, a_x_0, a_y_0, noise_v_x_0, noise_v_y_0, noise_x_0, noise_y_0, reward
    N = length(playback_trace)
    steps = 0:N-1
    v_x_car = zeros(N)
    v_y_car = zeros(N)
    x_car = zeros(N)
    y_car = zeros(N)
    v_x_ped_0 = zeros(N)
    v_y_ped_0 = zeros(N)
    x_ped_0 = zeros(N)
    y_ped_0 = zeros(N)
    a_x_0 = zeros(N)
    a_y_0 = zeros(N)
    noise_v_x_0 = zeros(N)
    noise_v_y_0 = zeros(N)
    noise_x_0 = zeros(N)
    noise_y_0 = zeros(N)
    reward = zeros(N)

    for t in 1:N
        ego_state = playback_trace[t].entities[2].state.veh_state # NOTE INDEX
        other_state = playback_trace[t].entities[1].state.veh_state # NOTE INDEX

        # TODO: x-y velocity separate
        v_x_car[t] = ego_state.v
        # v_y_car[t] = 0
        x_car[t] = ego_state.posG.x
        y_car[t] = ego_state.posG.y
        # TODO: θ

        v_x_ped_0[t] = other_state.v
        # v_y_ped_0[t] = 0
        x_ped_0[t] = other_state.posG.x
        y_ped_0[t] = other_state.posG.y

        if origin_offset
            # Other car is stopped
            x_car[t] -= x_ped_0[t]
            y_car[t] -= y_ped_0[t]
            x_ped_0[t] = 0
            y_ped_0[t] = 0
        end

        # TODO/ a_x_0? a_y_0?
        # TODO. Noise.
        ego_noise = playback_trace[t].entities[2].state.noise
        other_noise = playback_trace[t].entities[1].state.noise

        noise_v_x_0[t] = ego_noise.vel
        # noise_v_y_0[t] = 0 # TODO.
        noise_x_0[t] = ego_noise.pos.x
        noise_y_0[t] = ego_noise.pos.y

        # if RiskSimulator.AdversarialDriving.any_collides(playback_trace[t])
        if abs(x_car[t] - x_ped_0[t]) < 4.5
            # Stop vehicles when they collide
            v_x_car[t] = 0
            v_y_car[t] = 0
            v_x_ped_0[t] = 0
            v_y_ped_0[t] = 0
            x_car[t] = x_car[t-1]
            y_car[t] = y_car[t-1]
            # NOTE, assuming other vehicle is stationary.
        end
    end

    df = DataFrame()
    df["step"] = steps
    df["v_x_car"] = v_x_car
    df["v_y_car"] = v_y_car
    df["x_car"] = x_car
    df["y_car"] = y_car
    df["v_x_ped_0"] = v_x_ped_0
    df["v_y_ped_0"] = v_y_ped_0
    df["x_ped_0"] = x_ped_0
    df["y_ped_0"] = y_ped_0
    df["a_x_0"] = a_x_0
    df["a_y_0"] = a_y_0
    df["noise_v_x_0"] = noise_v_x_0
    df["noise_v_y_0"] = noise_v_y_0
    df["noise_x_0"] = noise_x_0
    df["noise_y_0"] = noise_y_0
    df["reward"] = reward

    CSV.write(filename, df)

    return playback_trace
end

playback_trace = export_carla_script(planner, buildingmap)
