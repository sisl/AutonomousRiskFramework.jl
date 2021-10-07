Adversarial CARLA Gym Environment
========================
Based on [ScenarioRunner](https://github.com/carla-simulator/scenario_runner), modified for _adaptive stress testing_ (AST).

> **⚠ WORK IN PROGRESS ⚠**

Install Instructions
========================
1. Download CARLA 0.9.11: https://github.com/carla-simulator/carla/releases/tag/0.9.11

1. Download `scenario_runner` 0.9.11
    > **NOTE: Make sure `scenario_runner` and `adversarial_carla_env` are in the same directory.**
    ```
    git clone https://github.com/carla-simulator/scenario_runner
    cd scenario_runner
    git checkout v0.9.11
    ```

1. Get Python packages:
    ```
    pip install -r requirements.txt -r ..\adversarial_carla_env\requirements.txt
    ```

1. Set up environment variables
    - See `env_set.txt`



Running Instructions
========================

1. Execute stress testing and record the final trajectory (requires setting environment variables, follow `env_set.txt`). Note this launches the CARLA executable if not already open.
    ```
    python run.py
    ```
