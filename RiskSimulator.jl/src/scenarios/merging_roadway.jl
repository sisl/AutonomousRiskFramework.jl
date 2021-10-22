"""
Create highway merging roadway.
- w: lane width
- l: lane length
- n: number of lanes
"""
function merging_roadway(; r=5.0, w=DEFAULT_LANE_WIDTH, l=50.0, n=2)
    mid_factor = 6
    # roadway = multi_lane_roadway(w=w, l=l*(mid_factor-1)/mid_factor, n=n)
    roadway = multi_lane_roadway(w=w, l=l, n=n)

    A = VecSE2(0,-w,0)
    B = VecSE2(l,0,0)
    mid = VecE2(l/mid_factor,0)
    C = B-mid
    E = VecSE2(0,0,0)
    D = A+mid

    # append_to_curve!(roadway.segments[1].lanes[1].curve, gen_straight_curve(convert(VecE2, mid), convert(VecE2, B), 2)[2:end])


    # Onramp
    curve = gen_straight_curve(convert(VecE2, A), convert(VecE2,D), 2)
    merge_index = curveindex_end(curve)
    append_to_curve!(curve, gen_bezier_curve(D, C, 3r, 3r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, C), convert(VecE2, B), 2)[2:end])

    SPLIT_ROADWAY = true
    if SPLIT_ROADWAY
        roadway2 = multi_lane_roadway(w=w, l=l/mid_factor, n=n, origin=C)
        roadway2.segments[1].id = 2
        roadway2.segments[1].lanes[1].tag = LaneTag(2,1)
        roadway2.segments[1].lanes[2].tag = LaneTag(2,2)
        push!(roadway.segments, roadway2.segments...)

        highway_lane_id = length(roadway.segments)-1
    else
        highway_lane_id = length(roadway.segments)
    end

    highway_tag = LaneTag(highway_lane_id, 1)
    merge_lane_id = length(roadway.segments)+1
    merge_tag = LaneTag(merge_lane_id,1)

    lane = Lane(merge_tag, curve, boundary_left=LaneBoundary(:broken, :white), next=RoadIndex(merge_index, highway_tag))
    # lane = Lane(merge_tag, curve)

    # lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    return roadway
end


function hw_straight(; roadway::Roadway, id::Int64=1, noise::Noise=Noise(), s::Float64=3.0, v::Float64=8.0, goals::Vector=[1], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=1, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


function hw_merging(; roadway::Roadway, id::Int64=1, noise::Noise=Noise(), s::Float64=9.0, v::Float64=8.0, goals::Vector=[1], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=3, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


function AutomotiveSimulator.propagate(veh::Entity{BlinkerState, D, I}, action::BlinkerVehicleControl, roadway::Roadway, Δt::Float64) where {D,I}
    # set the new goal
    vs = veh.state.veh_state
    if action.toggle_goal
        gs = goals(veh)
        curr_index = findfirst(gs .== laneid(veh))
        @assert !isnothing(curr_index)
        new_goal = gs[curr_index % length(gs) + 1]
        if can_have_goal(veh, new_goal, roadway)
            vs = set_lane(vs, new_goal, roadway)
        end
    end

    starting_lane = laneid(vs)

    # Update the kinematics of the vehicle (don't allow v < 0)
    vs_entity = Entity(vs, veh.def, veh.id)
    vs = propagate(vs_entity, LaneFollowingAccel(action.a + action.da), roadway, Δt)

    # Set the blinker state and return
    new_blink = action.toggle_blinker ? !blinker(veh) : blinker(veh)
    bs = BlinkerState(vs, new_blink, goals(veh), action.noise)
    # @assert starting_lane == laneid(bs) # TODO: Merge into master.
    bs
end


# TODO: (kwargs...) for `t_intersection`
# TODO: UrbanIDM???
function scenario_hw_merging(; init_noise_1::Noise=Noise(), init_noise_2::Noise=Noise(),
                             s_sut::Float64=3.0, s_adv::Float64=3.0, v_sut::Float64=8.0, v_adv::Float64=8.0)
    roadway = merging_roadway()

    # sut_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=!params.ignore_sensors)
    sut_model = TIDM(get_XIDM_template())
    sut_model.idm.v_des = 15
    sut_model.noisy_observations = true # TODO. Flip who is SUT/adversary?
    sut_model.goals[3] = [3]
    map(k->delete!(sut_model.goals, k), 4:7)
    sut = BlinkerVehicleAgent(hw_straight(id=1, noise=init_noise_1, roadway=roadway, s=s_sut, v=v_sut), sut_model)

    # adversary_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=false)
    adversary_model = TIDM(get_XIDM_template())
    adversary_model.idm.v_des = 15
    adversary_model.noisy_observations = true # TODO: important?
    adversary_model.goals[3] = [3]
    map(k->delete!(adversary_model.goals, k), 4:7)
    adversary = BlinkerVehicleAgent(hw_merging(id=2, noise=init_noise_2, roadway=roadway, s=s_adv, v=v_adv), adversary_model)

    return Scenario(roadway, sut, adversary)
end
