using AdversarialDriving

@with_kw mutable struct Scenario
    roadway::Roadway
    sut
    adversary
    disturbances = Disturbance[BlinkerVehicleControl(), BlinkerVehicleControl()]
end

Scenario(roadway, sut, adversary) = Scenario(roadway=roadway, sut=sut, adversary=adversary)


include("t_intersection.jl")
include("x_intersection.jl")
include("two_lane_roadway.jl")
include("multi_lane_roadway.jl")
include("crosswalk_roadway.jl")
include("merging_roadway.jl")
include("traffic_circle_roadway.jl")


@enum SCENARIO begin
    CROSSING
    T_HEAD_ON
    T_LEFT
    STOPPING
    MERGING
    CROSSWALK
end


function get_scenario(scenario_enum::SCENARIO; kwargs...)
    if scenario_enum == T_HEAD_ON
        return scenario_t_head_on_turn(; kwargs...)
    elseif scenario_enum == T_LEFT
        return scenario_t_left_turn(; kwargs...)
    elseif scenario_enum == STOPPING
        return scenario_hw_stopping(; kwargs...)
    elseif scenario_enum == CROSSING
        return scenario_crossing(; kwargs...)
    elseif scenario_enum == MERGING
        return scenario_hw_merging(; kwargs...)
    elseif scenario_enum == CROSSWALK
        return scenario_pedestrian_crosswalk(; kwargs...)
    end
end


# Ranges computed based on manual rendering and visual inspection.
function get_scenario_options(scenario_enum::SCENARIO)
    if scenario_enum == T_HEAD_ON
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == T_LEFT
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == STOPPING
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == CROSSING
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == MERGING
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    elseif scenario_enum == CROSSWALK
        return Dict("s_sut" => [2.0, 20.0], "s_adv" => [2.0, 20.0], "v_sut" => [1.0, 5.0], "v_adv" =>[1.0, 5.0])
    end
end


function get_scenario_string(scenario_enum::SCENARIO)
    if scenario_enum == T_HEAD_ON
        return "T-intersection (head-on turn)"
    elseif scenario_enum == T_LEFT
        return "T-intersection (left turn)"
    elseif scenario_enum == STOPPING
        return "Stopping on highway"
    elseif scenario_enum == CROSSING
        return "Crossing intersection"
    elseif scenario_enum == MERGING
        return "Highway merging"
    elseif scenario_enum == CROSSWALK
        return "Pedestrian in crosswalk"
    end
end



function append_to_curve!(target::Curve, newstuff::Curve)
    s_end = target[end].s
    for c in newstuff
        push!(target, CurvePt(c.pos, c.s+s_end, c.k, c.kd))
    end
    return target
end
