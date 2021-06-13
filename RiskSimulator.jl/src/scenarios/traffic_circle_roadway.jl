"""
Create traffic circle merging roadway.
- w: lane width
- l: lane length
- length: statium length
- len: statium width
- radius: statium radius
- n: number of lanes
"""
function traffic_circle_roadway(; w=DEFAULT_LANE_WIDTH, l=50.0, n=2, len=1.0, width=1.0, radius=15.0)
    roadway = gen_stadium_roadway(n; length=len, width=width, radius=radius)
    # return roadway

    mid_factor = 6

    MID_CIRCLE_MERGE = false

    if MID_CIRCLE_MERGE
        A = VecSE2(-6w,2w,π/2)
        B = VecSE2(-8w,4w,π/2)
        # B = VecSE2(l,0,0)
        mid = VecE2(l,0)
        C = B-mid
        E = VecSE2(0,0,0)
        D = B-mid/2

        # Onramp
        curve = gen_straight_curve(convert(VecE2, D), convert(VecE2,B), 2)
        merge_index = curveindex_end(curve)
        append_to_curve!(curve, gen_bezier_curve(B, A, 0.6r, -0.6r, 51)[2:end])
        # append_to_curve!(curve, gen_straight_curve(convert(VecE2, C), convert(VecE2, B), 2)[2:end])
        F = VecSE2(-6w, 3w, -π/2)
    else
        A = VecSE2(0,-w,π/2)
        B = VecSE2(l,0,0)
        mid = VecE2(l,0)
        C = B-mid
        E = VecSE2(0,0,0)
        D = A-mid

        # Onramp
        curve = gen_straight_curve(convert(VecE2, D), convert(VecE2,A), 2)
        merge_index = curveindex_end(curve)
        # append_to_curve!(curve, gen_bezier_curve(B, A, 0.6r, -0.6r, 51)[2:end])
        # append_to_curve!(curve, gen_straight_curve(convert(VecE2, C), convert(VecE2, B), 2)[2:end])
        F = VecSE2(-6w, 3w, -π/2)
    end
    # append_to_curve!(roadway.segments[1].lanes[1].curve, gen_straight_curve(convert(VecE2, mid), convert(VecE2, B), 2)[2:end])



    # curve = gen_straight_curve(convert(VecE2, B+VecE2(-l,0)), convert(VecE2, B), 2)
    # append_to_curve!(curve, gen_bezier_curve(B, C, 0.6r, 0.6r, 51)[2:end])
    # append_to_curve!(curve, gen_straight_curve(convert(VecE2, C), convert(VecE2, C+VecE2(0,-l)), 2))
    # lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    # push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))


    SPLIT_ROADWAY = false
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

    if MID_CIRCLE_MERGE
        lane = Lane(merge_tag, curve, boundary_left=LaneBoundary(:broken, :white))
    else
        lane = Lane(merge_tag, curve, boundary_left=LaneBoundary(:broken, :white), next=RoadIndex(merge_index, highway_tag))
    end

    # lane = Lane(LaneTag(length(roadway.segments)+1,1), curve)
    push!(roadway.segments, RoadSegment(lane.tag.segment, [lane]))

    return roadway
end
