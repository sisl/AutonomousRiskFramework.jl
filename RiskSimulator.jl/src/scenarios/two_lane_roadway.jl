"""
Create two lane roadway.
- w: lane width
- l: lane length
"""
function two_lane_roadway(; w=DEFAULT_LANE_WIDTH, l=25)
    roadway = Roadway()

    points = get_intersection_points(r=r, w=w)
    A, B, C, D, E, F = points.A, points.B, points.C, points.D, points.E, points.F

    # Append straight from left
    curve2 = gen_straight_curve(convert(VecE2, B+VecE2(-l,0)), convert(VecE2, B), 2)
    append_to_curve!(curve2, gen_straight_curve(convert(VecE2, B), convert(VecE2, E), 2)[2:end])
    append_to_curve!(curve2, gen_straight_curve(convert(VecE2, E), convert(VecE2, E+VecE2(l,0)), 2))
    lane2 = Lane(LaneTag(length(roadway.segments)+1,1), curve2)
    push!(roadway.segments, RoadSegment(lane2.tag.segment, [lane2]))

    # Append straight from right
    curve3 = gen_straight_curve(convert(VecE2, F+VecE2(l,0)), convert(VecE2, F), 2)
    append_to_curve!(curve3, gen_straight_curve(convert(VecE2, F), convert(VecE2, A), 2)[2:end])
    append_to_curve!(curve3, gen_straight_curve(convert(VecE2, A), convert(VecE2, A+VecE2(-l,0)), 2))
    lane3 = Lane(LaneTag(length(roadway.segments)+1,1), curve3)
    push!(roadway.segments, RoadSegment(lane3.tag.segment, [lane3]))

    return roadway
end
