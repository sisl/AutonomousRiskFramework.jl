push!(LOAD_PATH,"../src/")
using ObservationModels

using Documenter

makedocs(
    sitename = "ObservationModels.jl",
    modules  = [ObservationModels],
    pages=[
        "Home" => "index.md"
    ])

deploydocs(;
    repo="github.com/shubhg1996/ObservationModels.jl.git",
)