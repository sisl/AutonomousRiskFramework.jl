ScenarioRunner for CARLA (Modified for Adaptive Stress Testing)
========================
Modified from [ScenarioRunner](https://github.com/carla-simulator/scenario_runner)


Install Instructions
========================
1. Download CARLA 0.9.11: https://carla.org/2020/12/22/release-0.9.11/

1. Set up environment variables
    - See `env_set.bat` (change these to Unix-style and put them somewhere permanent)

1. Get Python packages:
    ```
    pip install -r requirements.txt
    ```


Running Instructions
========================
1. Launch CARLA server (from the folder containing `CarlaUE4.{exe|sh}`)
    ```
    CarlaUE4 -carla-rpc-port=2222 -windowed -ResX=320 -ResY=240 -benchmark -fps=10 -quality-level=Low
    ```
1. Load configs (remove no-rendering if needed)
    ```
    python ../util/config.py --map Town01 --port 2222 --spectator-loc 80.37 25.30 0.0 --no-rendering
    ```
1. Execute stress testing and record the final trajectory (requires setting environment variables, follow [env_set.txt](https://github.com/sisl/AutonomousRiskFramework/blob/shubh/carla_integration/CARLAIntegration/scenario_runner/env_set.txt))
    ```
    python scenario_runner_ast_gym.py --route /srunner/data/routes_ast.xml /srunner/data/ast_scenarios.json --port 2222 --agent /srunner/autoagents/ast_agent.py --record recordings
    ```
1. (SKIP FOR NOW) Generate plots of the AST search metrics (update the input filepath)
    ```
    python gen_plots.py
    ```


TODO:
=====================
1. Connect as fork of the main repository
2. Clean and structure the codebase
3. Remove hard-coded settings