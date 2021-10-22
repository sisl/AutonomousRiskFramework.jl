function get_intersection_points(;r=5.0, w=DEFAULT_LANE_WIDTH)
    A = VecSE2(0.0,w,-π)
    B = VecSE2(0.0,0.0,0.0)
    C = VecSE2(r,-r,-π/2)
    D = VecSE2(r+w,-r,π/2)
    E = VecSE2(2r+w,0,0)
    F = VecSE2(2r+w,w,-π)
    G = VecSE2(r+w,r+w,π/2)
    H = VecSE2(r,r+w,-π/2)
    I = VecSE2(r,-r,-π/2)
    J = VecSE2(0,w,0.0)
    K = VecSE2(2r+w,0,0.0)

    return (A=A, B=B, C=C, D=D, E=E, F=F, G=G, H=H, I=I, J=J, K=K)
end

"""
Create T-intersection roadway.
- r: turn radius
- w: lane width
- l: lane length
"""
function t_intersection(; r=5.0, w=DEFAULT_LANE_WIDTH, l=25)
    roadway = Roadway()

    points = get_intersection_points(r=r, w=w)
    A, B, C, D, E, F = points.A, points.B, points.C, points.D, points.E, points.F

    # First curve / lane
    # Append right turn coming from the left
    curve = gen_straight_curve(convert(VecE2, B+VecE2(-l,0)), convert(VecE2, B), 2)
    append_to_curve!(curve, gen_bezier_curve(B, C, 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve, gen_straight_curve(convert(VecE2, C), convert(VecE2, C+VecE2(0,-l)), 2))
    lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    # Second curve / lane
    # Append straight from left
    curve2 = gen_straight_curve(convert(VecE2, B+VecE2(-l,0)), convert(VecE2, B), 2)
    append_to_curve!(curve2, gen_straight_curve(convert(VecE2, B), convert(VecE2, E), 2)[2:end])
    append_to_curve!(curve2, gen_straight_curve(convert(VecE2, E), convert(VecE2, E+VecE2(l,0)), 2))
    lane2 = Lane(LaneTag(length(roadway.segments)+1,1), curve2)
    push!(roadway.segments, RoadSegment(lane2.tag.segment, [lane2]))

    # Third curve / lane
    # Append straight from right
    curve3 = gen_straight_curve(convert(VecE2, F+VecE2(l,0)), convert(VecE2, F), 2)
    append_to_curve!(curve3, gen_straight_curve(convert(VecE2, F), convert(VecE2, A), 2)[2:end])
    append_to_curve!(curve3, gen_straight_curve(convert(VecE2, A), convert(VecE2, A+VecE2(-l,0)), 2))
    lane3 = Lane(LaneTag(length(roadway.segments)+1,1), curve3)
    push!(roadway.segments, RoadSegment(lane3.tag.segment, [lane3]))

    # Fourth curve / lane
    # Append left turn coming from the right
    curve4 = gen_straight_curve(convert(VecE2, F+VecE2(l,0)), convert(VecE2, F), 2)
    append_to_curve!(curve4, gen_bezier_curve(F, C, 0.9r, 0.9r, 51)[2:end])
    append_to_curve!(curve4, gen_straight_curve(convert(VecE2, C), convert(VecE2, C+VecE2(0,-l)), 2))
    lane4 = Lane(LaneTag(length(roadway.segments)+1,1), curve4)
    push!(roadway.segments, RoadSegment(lane4.tag.segment, [lane4]))

    # Fifth curve / lane
    # Append right turn coming from below
    curve5 = gen_straight_curve(convert(VecE2, D+VecE2(0,-l)), convert(VecE2, D), 2)
    append_to_curve!(curve5, gen_bezier_curve(D, E, 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve5, gen_straight_curve(convert(VecE2, E), convert(VecE2, E+VecE2(l,0)), 2))
    lane5 = Lane(LaneTag(length(roadway.segments)+1,1), curve5)
    push!(roadway.segments, RoadSegment(lane5.tag.segment, [lane5]))

    # Sixth curve / lane
    # Append left turn coming from below
    curve6 = gen_straight_curve(convert(VecE2, D+VecE2(0,-l)), convert(VecE2, D), 2)
    append_to_curve!(curve6, gen_bezier_curve(D, A, 0.9r, 0.9r, 51)[2:end])
    append_to_curve!(curve6, gen_straight_curve(convert(VecE2, A), convert(VecE2, A+VecE2(-l,0)), 2))
    lane6 = Lane(LaneTag(length(roadway.segments)+1,1), curve6)
    push!(roadway.segments, RoadSegment(lane6.tag.segment, [lane6]))

    return roadway
end


function t_left_to_right(; roadway::Roadway, id::Int64=1, noise::Noise=Noise(), s::Float64=4.0, v::Float64=8.0, goals::Vector=[1], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=2, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


function t_right_to_turn(; roadway::Roadway, id::Int64=1, noise::Noise=Noise(), s::Float64=8.0, v::Float64=8.0, goals::Vector=[1], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=4, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


function t_bottom_to_turn_left(; roadway::Roadway, id::Int64=1, noise::Noise=Noise(), s::Float64=8.0, v::Float64=8.0, goals::Vector=[1], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=6, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


# TODO: (kwargs...) for `t_intersection`
# TODO: UrbanIDM???
function scenario_t_head_on_turn(; init_noise_1::Noise=Noise(), init_noise_2::Noise=Noise(),
                                 s_sut::Float64=4.0, s_adv::Float64=8.0, v_sut::Float64=8.0, v_adv::Float64=8.0)
    roadway = t_intersection()

    # sut_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=!params.ignore_sensors)
    sut_model = TIDM(get_XIDM_template())
    sut_model.idm.v_des = 15
    sut_model.noisy_observations = true # TODO. Flip who is SUT/adversary?
    sut = BlinkerVehicleAgent(t_left_to_right(id=1, noise=init_noise_1, roadway=roadway, s=s_sut, v=v_sut), sut_model)

    # adversary_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=false)
    adversary_model = TIDM(get_XIDM_template())
    adversary_model.idm.v_des = 15
    adversary_model.noisy_observations = true # TODO: important?
    adversary = BlinkerVehicleAgent(t_right_to_turn(id=2, noise=init_noise_2, roadway=roadway, s=s_adv, v=v_adv), adversary_model)

    return Scenario(roadway, sut, adversary)
end


# TODO: (kwargs...) for `t_intersection`
# TODO: UrbanIDM???
function scenario_t_left_turn(; init_noise_1::Noise=Noise(), init_noise_2::Noise=Noise(),
                              s_sut::Float64=4.0, s_adv::Float64=8.0, v_sut::Float64=8.0, v_adv::Float64=8.0)
    roadway = t_intersection()

    # sut_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=!params.ignore_sensors)
    sut_model = TIDM(get_XIDM_template())
    sut_model.idm.v_des = 15
    sut_model.noisy_observations = true # TODO. Flip who is SUT/adversary?
    sut = BlinkerVehicleAgent(t_left_to_right(id=1, noise=init_noise_1, roadway=roadway, s=s_sut, v=v_sut), sut_model)

    # adversary_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=false)
    adversary_model = TIDM(get_XIDM_template())
    adversary_model.idm.v_des = 15
    adversary_model.noisy_observations = true # TODO: important?
    adversary = BlinkerVehicleAgent(t_bottom_to_turn_left(id=2, noise=init_noise_2, roadway=roadway, s=s_adv, v=v_adv), adversary_model)

    return Scenario(roadway, sut, adversary)
end
