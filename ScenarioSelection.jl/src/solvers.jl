include("ast_td3_solver.jl")
include("mc_solver.jl")

# TODO: pass (mdp, s) check isterminal(mdp, s)
function extract_info(info)
    data_point = missing
    if info["done"] == true
        data_point = Dict()
        data_point["cost"] = info["cost"]
        data_point["reward"] = info["reward"]
        data_point["speed_before_collision"] = info["speed_before_collision"]
        data_point["delta_v"] = info["delta_v"]
    end
    return data_point
end