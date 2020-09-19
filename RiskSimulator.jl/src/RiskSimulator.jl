module RiskSimulator # Naming?

using Parameters

export Simulator,
       Vehicle,
       Agent,
       Sensor,
       Scenario,
       simulate,
       evaluate


@with_kw mutable struct Vehicle
    agent = nothing
    dynamics = nothing
end

@with_kw mutable struct Scenario
    file = nothing
end

@with_kw mutable struct Simulator
    vehicles::Vector{Vehicle} = Vehicle[]
    scenario::Scenario = Scenario()

    Simulator(vehicles::Vector{Vehicle}) = new(vehicles, Scenario())
    Simulator(vehicles::Vector{Vehicle}, scenario::Scenario) = new(vehicles, scenario)
end


"""
Full simulation with output data for plotting and analysis.
"""
function simulate(sim::Simulator)
    @info "Running simulation:\n$sim"
end


"""
Streamlined evaluation for aggregate metrics.
"""
function evaluate(sim::Simulator)
    @info "Running evaluation:\n$sim"
end

end # module
