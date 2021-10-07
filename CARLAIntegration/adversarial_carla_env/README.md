Adversarial CARLA Gym Environment
========================
Based on [ScenarioRunner](https://github.com/carla-simulator/scenario_runner), modified for _adaptive stress testing_ (AST).

> **⚠ WORK IN PROGRESS ⚠**

Install Instructions
========================
1. Download CARLA 0.9.11: https://carla.org/2020/12/22/release-0.9.11/

1. Download `scenario_runner` 0.9.11
    > **NOTE: Make sure `scenario_runner` and `adversarial_carla_env` are in the same directory.**
    ```
    git clone https://github.com/carla-simulator/scenario_runner
    cd scenario_runner
    git checkout v0.9.11
    ```

1. Set up environment variables
    - See `env_set.bat` (change these to Unix-style and put them somewhere permanent)

1. Get Python packages (TODO: `adversarial_carla_env` requirements.txt):
    ```
    cd scenario_runner
    pip install -r requirements.txt
    cd ../adversarial_carla_env
    pip install -r requirements.txt
    ```


Running Instructions
========================
1. Launch CARLA server (from the folder containing `CarlaUE4.{exe|sh}`)
    ```
    CarlaUE4 -carla-rpc-port=2222 -windowed -ResX=320 -ResY=240 -benchmark -fps=10 -quality-level=Low
    ```
1. Load configs (**remove `--no-rendering` if needed/for debugging**)
    ```
    python ../util/config.py --map Town01 --port 2222 --spectator-loc 80.37 25.30 0.0 --no-rendering
    ```
1. Execute stress testing and record the final trajectory (requires setting environment variables, follow `env_set.bat`)
    ```
    python adversarial_carla_env.py
    ```
