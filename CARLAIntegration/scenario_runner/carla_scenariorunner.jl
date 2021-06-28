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

## 3) Then run this.
# @async run(`python ast_no_rendering_mode.py`)
ENV["PYTHON"] = raw'C:\Users\shubh\AppData\Local\pypoetry\Cache\virtualenvs\carla-controls-b_oJSFnG-py3.7\Scripts\python.exe'
ENV["PYTHONHOME"] = raw'C:\Users\shubh\AppData\Local\pypoetry\Cache\virtualenvs\carla-controls-b_oJSFnG-py3.7\Scripts\python.exe'
ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")
ENV["CARLA_ROOT"] = "C:\\Users\\shubh\\OneDrive\\Documents\\Carla\\WindowsNoEditor"
ENV["SCENARIO_RUNNER_ROOT"] = "C:\\Users\\shubh\\OneDrive\\Documents\\Carla\\WindowsNoEditor\\PythonAPI\\scenario_runner"
ENV["PYTHONPATH"] = ENV["CARLA_ROOT"]*"\\PythonAPI\\carla\\dist\\carla-0.9.10-py3.7-win-amd64.egg"
ENV["PYTHONPATH"] = ENV["PYTHONPATH"]*";"*ENV["CARLA_ROOT"]*"\\PythonAPI\\carla\\agents"
ENV["PYTHONPATH"] = ENV["PYTHONPATH"]*";"*ENV["CARLA_ROOT"]*"\\PythonAPI\\carla"
ENV["PYTHONPATH"] = ENV["PYTHONPATH"]*";"*ENV["CARLA_ROOT"]*"\\PythonAPI"
using PyCall

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
        @async run(`python ..\\util\\config.py --port=$PORT --map Town01 --spectator-loc 80.37 25.30 0.0`)
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

# Run AST scenario runner
function run_no_render(rerun=false)
    global PORT
    @info "Running AST agent in scenario runner."
    py"""
    exec(open('scenario_runner_tools.py').read())
    """
    args = py"create_args"(
        "127.0.0.1",
        PORT,            
        10.0,
        8000,
        [".\\srunner\\data\\routes_ast.xml", ".\\srunner\\data\\ast_scenarios.json"],
        ".\\srunner\\autoagents\\ast_agent.py",
        "recordings",
        1
    )
    srunner = py"create_srunner"(args)
    
    @show srunner
end
run_no_render()