using Test
using ObservationModels
using AutomotiveSimulator
using AdversarialDriving
using Distributions

struct DummyLandmark <: Landmark end
struct DummySensorObservation <: SensorObservation end

@testset "structs" begin
    d = DummyLandmark()
    @test_throws ErrorException ObservationModels.position(d)

    d = DummySensorObservation()
    @test_throws ErrorException ObservationModels.noise(d)

    d = Building(1, 1., 1., VecSE2(0., 0., 0.))
    @test ObservationModels.position(d) == VecE3(0., 0., 0.)

    d = Satellite()
    @test ObservationModels.position(d) == VecE3(0., 0., 0.)

    d = GPSRangeMeasurement()
    @test ObservationModels.noise(d)[1] == 0.0

    d = BuildingMap()
    @test length(d.buildings) == 0

    d = DistParams()
    @test d.num_samples == 50
end

@testset "sensors" begin
    roadway = gen_straight_roadway(1, 100.)

    init_noise = Noise(pos = (0, 0), vel = 0)

    scene = Scene([
        Entity(BlinkerState(VehicleState(VecSE2(0.0, 1.0, 0), roadway, 0.0), false, [], init_noise), VehicleDef(), 1),
        Entity(BlinkerState(VehicleState(VecSE2(10.0, 1.0, 0), roadway, 0.0), false, [], init_noise), VehicleDef(), 2)
    ])

    old_state1 = scene[1].state.veh_state
    old_id1 = scene[1].id
    old_state2 = scene[2].state.veh_state
    old_id2 = scene[2].id
    update_noiseless!(scene[1], scene)
    @test scene[1].state.veh_state == old_state1
    @test scene[2].state.veh_state == old_state2
    @test scene[1].id == old_id1
    @test scene[2].id == old_id2

    old_state1 = scene[1].state.veh_state
    old_id1 = scene[1].id
    old_state2 = scene[2].state.veh_state
    old_id2 = scene[2].id
    update_gaussian_noise!(scene[1], scene)
    @test scene[1].state.veh_state == old_state1
    @test scene[2].state.veh_state == old_state2
    @test scene[1].id == old_id1
    @test scene[2].id == old_id2

    old_state1 = scene[1].state.veh_state
    old_id1 = scene[1].id
    old_state2 = scene[2].state.veh_state
    old_id2 = scene[2].id
    update_rb_noise!(scene[1], scene)
    @test scene[1].state.veh_state == old_state1
    @test scene[2].state.veh_state == old_state2
    @test scene[1].id == old_id1
    @test scene[2].id == old_id2

    bmap = BuildingMap()
    fixed_sats = [
    ObservationModels.Satellite(pos=VecE3(-1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(-1e7, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(100.0, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 0.0, 1e7), clk_bias=0.0)
    ]

    old_state1 = scene[1].state.veh_state
    old_id1 = scene[1].id
    old_state2 = scene[2].state.veh_state
    old_id2 = scene[2].id
    update_gps_noise!(scene[1], scene, bmap, fixed_sats)
    @test scene[1].state.veh_state == old_state1
    @test scene[2].state.veh_state == old_state2
    @test scene[1].id == old_id1
    @test scene[2].id == old_id2
end

@testset "distributions" begin
    d = Fsig_Normal(0.0)
    @test Distributions.mean(d) == 0.0
    @test Distributions.var(d) == 25.0

    d = INormal_GMM(0.0, 5.0)
    @test Distributions.mean(d) == 0.0
    @test Distributions.var(d) == 25.0

    d = INormal_Uniform(0.0, 5.0)
    @test Distributions.mean(d) == 0.0
    @test Distributions.var(d) == 25.0

    noise_t = [Noise(pos=VecE2(rand(), rand()), vel=rand()) for i=1:10]
    @test length(Distributions.pdf(MvNormal([0.,0.,0.], [1.,1.,1.]), noise_t)) == 10
    @test length(mean(Distributions.fit(MvNormal, noise_t))) == 3
end

@testset "learned probability" begin
    roadway = gen_straight_roadway(1, 100.)

    init_noise = Noise(pos = (0, 0), vel = 0)

    scene = Scene([
        Entity(BlinkerState(VehicleState(VecSE2(0.0, 1.0, 0), roadway, 0.0), false, [], init_noise), VehicleDef(), 1),
        Entity(BlinkerState(VehicleState(VecSE2(10.0, 1.0, 0), roadway, 0.0), false, [], init_noise), VehicleDef(), 2)
    ])

    fixed_sats = [
    ObservationModels.Satellite(pos=VecE3(-1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 1e7, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(-1e7, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(100.0, 0.0, 1e7), clk_bias=0.0),
    ObservationModels.Satellite(pos=VecE3(1e7, 0.0, 1e7), clk_bias=0.0)
    ]

    params = DistParams(satpos=fixed_sats)
    d = sample_and_fit(scene[1], scene, params)
    @test length(d) == 2

    scenes = [scene for i=1:5]
    feat, y = preprocess_data(1, scenes)
    @test size(feat)[2] == 5
    @test size(feat)[2] == 5
    @test size(feat)[1]%2 == 0
    @test size(feat)[1]%2 == 0
end