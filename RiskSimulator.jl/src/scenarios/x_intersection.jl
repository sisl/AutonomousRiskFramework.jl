"""
Create cross intersection roadway.
- r: turn radius
- w: lane width
- l: lane length
"""
function x_intersection(; r=5.0, w=DEFAULT_LANE_WIDTH, l=25)
    # Modified from:
    # https://sisl.github.io/AutomotiveSimulator.jl/dev/tutorials/intersection/#Intersection-1

    # Reuse T-intersection
    roadway = t_intersection(r=r, w=w, l=l)
    # roadway = Roadway()
    points = get_intersection_points(r=r, w=w)
    B, D, F, G, H, I, J, K = points.B, points.D, points.F, points.G, points.H, points.I, points.J, points.K
    
    # Append straight from below
    curve1 = gen_straight_curve(convert(VecE2, D+VecE2(0,-l)), convert(VecE2,D), 2)
    append_to_curve!(curve1, gen_straight_curve(convert(VecE2, D), convert(VecE2, G), 2)[2:end])
    append_to_curve!(curve1, gen_straight_curve(convert(VecE2, G), convert(VecE2, G+VecE2(0,l)), 2))
    lane1 = Lane(LaneTag(length(roadway.segments)+1,1), curve1)
    push!(roadway.segments, RoadSegment(lane1.tag.segment, [lane1]))

    # Append straight from above
    curve2 = gen_straight_curve(convert(VecE2, H+VecE2(0,l)), convert(VecE2,H), 2)
    append_to_curve!(curve2, gen_straight_curve(convert(VecE2, H), convert(VecE2, I), 2)[2:end])
    append_to_curve!(curve2, gen_straight_curve(convert(VecE2, I), convert(VecE2, I+VecE2(0,-l)), 2))
    lane2 = Lane(LaneTag(length(roadway.segments)+1,1), curve2)
    push!(roadway.segments, RoadSegment(lane2.tag.segment, [lane2]))

    # Append right turn coming from above
    curve3 = gen_straight_curve(convert(VecE2, H+VecE2(0,l)), convert(VecE2,H), 2)
    append_to_curve!(curve3, gen_bezier_curve(H, J, 0.6r, -0.6r, 51)[2:end])
    append_to_curve!(curve3, gen_straight_curve(convert(VecE2, J), convert(VecE2, J+VecE2(-l,0)), 2))
    lane3 = Lane(LaneTag(length(roadway.segments)+1,1), curve3)
    push!(roadway.segments, RoadSegment(lane3.tag.segment, [lane3]))

    # Append left turn coming from above
    curve4 = gen_straight_curve(convert(VecE2, H+VecE2(0,l)), convert(VecE2,H), 2)
    append_to_curve!(curve4, gen_bezier_curve(H, K, 0.9r, 0.9r, 51)[2:end])
    append_to_curve!(curve4, gen_straight_curve(convert(VecE2, K), convert(VecE2, K+VecE2(l,0)), 2))
    lane4 = Lane(LaneTag(length(roadway.segments)+1,1), curve4)
    push!(roadway.segments, RoadSegment(lane4.tag.segment, [lane4]))

    # Append right turn coming from right
    curve5 = gen_straight_curve(convert(VecE2, F+VecE2(l,0)), convert(VecE2,F), 2)
    append_to_curve!(curve5, gen_bezier_curve(F, G, 0.6r, 0.6r, 51)[2:end])
    append_to_curve!(curve5, gen_straight_curve(convert(VecE2, G), convert(VecE2, G+VecE2(0,l)), 2))
    lane5 = Lane(LaneTag(length(roadway.segments)+1,1), curve5)
    push!(roadway.segments, RoadSegment(lane5.tag.segment, [lane5]))

    # Append left turn coming from left
    curve6 = gen_straight_curve(convert(VecE2, B+VecE2(-l,0)), convert(VecE2,B), 2)
    append_to_curve!(curve6, gen_bezier_curve(B, G, 0.9r, 0.9r, 51)[2:end])
    append_to_curve!(curve6, gen_straight_curve(convert(VecE2, G), convert(VecE2, G+VecE2(0,l)), 2))
    lane6 = Lane(LaneTag(length(roadway.segments)+1,1), curve6)
    push!(roadway.segments, RoadSegment(lane6.tag.segment, [lane6]))

    return roadway
end


function get_XIDM_template()
    XIDM_template = deepcopy(Tint_TIDM_template)
    XIDM_template.goals[7] = [5,6]
    XIDM_template.intersection_enter_loc[7] = copy(XIDM_template.intersection_enter_loc[6])
    XIDM_template.intersection_exit_loc[7] = copy(XIDM_template.intersection_exit_loc[6])
    XIDM_template.should_blink[7] = copy(XIDM_template.should_blink[6])
    XIDM_template.yields_way[7] = copy(XIDM_template.yields_way[6])
    Tint_signal_right[7] = false
    return XIDM_template
end


function cross_left_to_right(; id::Int64=1, noise::Noise=Noise(), roadway::Roadway, s::Float64=4.0, v::Float64=8.0, goals::Vector=[1], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=2, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


function cross_bottom_to_top(; id::Int64=1, noise::Noise=Noise(), roadway::Roadway, s::Float64=4.0, v::Float64=8.0, goals::Vector=[1], blinker::Bool=false)
    vehicle = BlinkerVehicle(id=id, roadway=roadway, lane=7, s=s, v=v, goals=goals, blinker=blinker, noise=noise)
    return (rng::AbstractRNG=Random.GLOBAL_RNG) -> vehicle
end


# TODO: (kwargs...) for `x_intersection`
# TODO: UrbanIDM???
function scenario_crossing(; init_noise_1::Noise=Noise(), init_noise_2::Noise=Noise(),
                           s_sut::Float64=4.0, s_adv::Float64=4.0, v_sut::Float64=8.0, v_adv::Float64=8.0)
    roadway = x_intersection()

    # sut_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=true, ignore_idm=!params.ignore_sensors)
    sut_model = TIDM(get_XIDM_template())
    sut_model.idm.v_des = 15
    sut_model.noisy_observations = true # TODO. Flip who is SUT/adversary?
    sut = BlinkerVehicleAgent(cross_left_to_right(id=1, noise=init_noise_1, roadway=roadway, s=s_sut, v=v_sut), sut_model)

    # adversary_model = UrbanIDM(idm=IntelligentDriverModel(v_des=15.0), noisy_observations=false)
    adversary_model = TIDM(get_XIDM_template())
    adversary_model.idm.v_des = 15
    adversary_model.noisy_observations = true # TODO: important?
    adversary = BlinkerVehicleAgent(cross_bottom_to_top(id=2, noise=init_noise_2, roadway=roadway, s=s_adv, v=v_adv), adversary_model)

    return Scenario(roadway, sut, adversary)
end
