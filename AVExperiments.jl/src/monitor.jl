using Distributed

function is_carla_running()
    if Sys.iswindows()
        tasks = read(`tasklist`, String)
        return occursin("CarlaUE4.exe", tasks)
    else
        error("Checking CARLA executable not setup for Linux.")
    end
end

function start_carla_monitor(time=5)
    # Check if CARLA executable is running
    if Sys.iswindows()
        while is_carla_running()
            # CARLA already open
            sleep(time)
        end
        # CARLA not open, so open it.
        carla_start = joinpath(@__DIR__, "..", "..", "CARLAIntegration", "adversarial_carla_env", "carla-start.bat")
        @info "Re-opening CARLA executable."
        run(`cmd /c $carla_start`)
        start_carla_monitor()
    else
        @warn "CARLA monitoring only setup for Windows."
    end
end