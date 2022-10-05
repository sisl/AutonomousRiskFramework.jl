using Distributed

function is_carla_running()
    if Sys.iswindows()
        tasks = read(`tasklist`, String)
        return occursin("CarlaUE4.exe", tasks)
    else
        tasks = read(`ps -aux`, String)
        return occursin("CarlaUE4.sh", tasks)
    end
end

function start_carla_monitor(time=5)
    # Check if CARLA executable is running
    while is_carla_running()
        # CARLA already open
        sleep(time)
    end

    # CARLA not open, so open it.
    if Sys.iswindows()    
        carla_start = joinpath(@__DIR__, "..", "..", "CARLAIntegration", "adversarial_carla_env", "carla-start.bat")
        @info "Re-opening CARLA executable."
        run(`cmd /c $carla_start`)
        
    else
        carla_start = joinpath(@__DIR__, "..", "..", "CARLAIntegration", "adversarial_carla_env", "carla-start.sh")
        @info "Re-opening CARLA executable."
        run(`$carla_start`, wait=false)
    end
    sleep(5)  # To give CARLA a chance to fully come up.
    start_carla_monitor()
end
