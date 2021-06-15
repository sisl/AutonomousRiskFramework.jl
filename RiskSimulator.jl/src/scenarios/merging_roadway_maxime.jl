function generate_merging_roadway(lane_width::Float64 = 3.0, 
                                   main_lane_vmax::Float64 = 20.0,
                                   merge_lane_vmax::Float64 = 15.0,
                                   main_lane_length::Float64 = 20.0, 
                                   merge_lane_length::Float64 = 20.0,
                                   after_merge_length::Float64 = 20.0,
                                   main_lane_angle::Float64 = float(pi)/4, 
                                   merge_lane_angle::Float64 = float(pi)/4) 
    # init empty roadway 
    roadway = Roadway()
    n_pts = 2 # sample points for the roadway, only two needed each time, since all segments are straight
    main_lane_id = 1
    merge_lane_id = 2
    main_tag = LaneTag(main_lane_id, 1)
    merge_tag = LaneTag(merge_lane_id, 1)
    # after_merge_tag = LaneTag(AFTER_merge_lane_id, 1)

    # define curves
    merge_point = VecE2(0.0, 0.0) 
    main_lane_startpt = merge_point + polar(main_lane_length, -float(pi) - main_lane_angle)
    main_curve = gen_straight_curve(main_lane_startpt, merge_point, n_pts)
    merge_index = curveindex_end(main_curve)
    append_to_curve!(main_curve, gen_straight_curve(merge_point, merge_point + polar(after_merge_length, 0.0), n_pts)[2:end])
    merge_lane_startpt = merge_point + polar(merge_lane_length, float(pi) + merge_lane_angle)
    merge_curve = gen_straight_curve(merge_lane_startpt, merge_point, n_pts)


    # define lanes with connections 
    main_lane = Lane(main_tag, main_curve, width = lane_width, speed_limit=SpeedLimit(0.,main_lane_vmax))
    merge_lane = Lane(merge_tag, merge_curve, width = lane_width,speed_limit=SpeedLimit(0.,merge_lane_vmax),
                        next=RoadIndex(merge_index, main_tag))

    # add segments to roadway 
    push!(roadway.segments, RoadSegment(main_lane_id, [main_lane]))
    push!(roadway.segments, RoadSegment(merge_lane_id, [merge_lane]))
  
    return roadway
end