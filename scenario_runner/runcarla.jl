using Distributions
using LightXML
using Random

global COLLISION_COUNT = 0
global PORT = "2222" # CARLA TCP port
if !@isdefined(CARLA_PROCESS)
    global CARLA_PROCESS = nothing
end

py_ver = read(`python --version`, String)
if py_ver[1:10] != "Python 3.7"
    # run(`../set_python37.bat`)
    error("Please set Python to 3.7")
end

## 1) Run this first. Wait.
function open_carla()
    global PORT, CARLA_PROCESS
    @info "Opening CARLA on port $PORT."

    use_config_py = false

    kill_carla()
    @async run(`..\\..\\CarlaUE4.exe -carla-rpc-port=$PORT`)
    # CARLA_PROCESS = run(`..\\..\\CarlaUE4.exe -carla-rpc-port=$PORT`, wait=false)

    if use_config_py
        # preload the correct map (Town01 for red light scenarios)
        @async run(`python ..\\util\\config.py --port=$PORT --map Town01`)
    end
end


"""
Kill the CARLA process (on Windows x64).
"""
function kill_carla()
    global CARLA_PROCESS
    # if !isnothing(CARLA_PROCESS)
    #     @info "Killing $CARLA_PROCESS"
    #     kill(CARLA_PROCESS)
    # end
    try
        run(`taskkill /f /t /im CarlaUE4-Win64-Shipping.exe`)
    catch err end
end


function add_disturbance(xdoc)
    x = parse(Float64, attribute(root(xdoc)["ego_vehicle"][1], "x"))
    noise_model = Uniform(0, 20)
    ϵ = rand(noise_model)
    x′ = string(x + ϵ)
    @info x′
    set_attribute(root(xdoc)["ego_vehicle"][1], "x", x′)
end

function disturb_file()
    @info "Adding disturbance to initial positions."
    xdoc = parse_file("red_light_config.xml")
    add_disturbance(xdoc)
    disturb_filename = "red_light_config_2.xml"
    save_file(xdoc, disturb_filename)
    return disturb_filename
end


## 2) Then run this. Wait.
function start_scenario_runner(; reload_world=true, tm_port=8001, xml="red_light_config.xml")
    global PORT
    @info "Starting scenario runner."

    # NOTE, fails to run iteratively in "--sync" mode.
    # arguments = ["--scenario", "OppositeVehicleRunningRedLight001", "--configFile", xml, "--trafficManagerPort", "$tm_port", "--timeout", "120", "--port", PORT, "--sync"]

    # Running full set of scenarios
    # arguments = ["--scenario", "group:OppositeVehicleRunningRedLight", "--configFile", "RunningRedLightExample.xml", "--trafficManagerPort", "$tm_port", "--timeout", "120", "--port", PORT]

    arguments = ["--scenario", "OppositeVehicleRunningRedLight001", "--configFile", xml, "--trafficManagerPort", "$tm_port", "--timeout", "120", "--port", PORT]
    if reload_world
        push!(arguments, "--reloadWorld")
    end

    scenerio_runner_cmd = ["python", "scenario_runner.py", arguments...]
    @info scenerio_runner_cmd
    @async run(Cmd(scenerio_runner_cmd))
end


## 3) Then run this.
# @async run(`python ast_no_rendering_mode.py`)
ENV["PYTHON"] = "C:\\Users\\shubh\\AppData\\Local\\pypoetry\\Cache\\virtualenvs\\carla-controls-b_oJSFnG-py3.7\\Scripts\\python.exe"
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
ENV["CARLA_ROOT"] = "C:\\Users\\shubh\\OneDrive\\Documents\\Carla\\WindowsNoEditor"
ENV["SCENARIO_RUNNER_ROOT"] = "C:\\Users\\shubh\\OneDrive\\Documents\\Carla\\WindowsNoEditor\\PythonAPI\\scenario_runner"
ENV["PYTHONPATH"] = ENV["CARLA_ROOT"]*"\\PythonAPI\\carla\\dist\\carla-0.9.10-py3.7-win-amd64.egg"
ENV["PYTHONPATH"] = ENV["PYTHONPATH"]*";"*ENV["CARLA_ROOT"]*"\\PythonAPI\\carla\\agents"
ENV["PYTHONPATH"] = ENV["PYTHONPATH"]*";"*ENV["CARLA_ROOT"]*"\\PythonAPI\\carla"
ENV["PYTHONPATH"] = ENV["PYTHONPATH"]*";"*ENV["CARLA_ROOT"]*"\\PythonAPI"
using PyCall

# Run AST agent without rendering
function run_no_render(rerun=false)
    global PORT
    @info "Running AST agent in 'no render' mode."
    # @async run(`python ast_no_rendering_mode.py --port $PORT`)

    ## NOTE: Comment out "if __name__" in ast_no_rendering_mode.py when using his py"" call style.
    # Async Python call.
    if rerun
        py"""
        rerun_game_loop()
        """
        # py"main($PORT)"
    else
        # Load Python code into enviroment
        py"""
        exec(open('ast_no_rendering_mode.py').read())
        """

        py"""
        main($PORT)
        """
    end
end

function runcarla(newcarla=false; N=10, reload_world=true, tm_port=8001, rerun=false)
    global CARLA_PROCESS

    # Traffic Manager Port: increase every run? (https://github.com/carla-simulator/carla/issues/2789#issuecomment-760951212)
    if newcarla
        # CARLA_PROCESS = nothing
        @warn "Please run `open_carla()` separately, then `runcarla(false)`"
        open_carla()
        sleep(10)
    end

    for n in 1:N
        @info "Running simulation $n"
        Random.seed!(n)

        # 2. scenario runner
        disturb_filename = disturb_file()
        # start_scenario_runner(tm_port=tm_port, xml=disturb_filename, reload_world=reload_world)
        start_scenario_runner(tm_port=tm_port, xml=disturb_filename, reload_world=reload_world)
        # start_scenario_runner(tm_port=tm_port, xml="red_light_config_no_collision.xml", reload_world=reload_world)
        sleep(10) # wait to run "no render" until this finishes (TODO: exact.)

        # 3. AST agent ("no render" mode)
        run_no_render(rerun || n > 1) # TODO. speed up?

        # println("Going to kill...")
        # sleep(20)
        # println("Killing...")
        # kill(process)
        num_collisions = py"NUM_COLLISIONS"
        @info num_collisions, num_collisions / n
    end
end

# function destroy()




##################################################
# Main: Open CARLA before running anything.
##################################################
open_carla()