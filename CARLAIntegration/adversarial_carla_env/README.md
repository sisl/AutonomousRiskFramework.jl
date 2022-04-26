Adversarial CARLA Gym Environment
========================
Based on [ScenarioRunner](https://github.com/carla-simulator/scenario_runner), modified for _adaptive stress testing_ (AST).

> **⚠ WORK IN PROGRESS ⚠**

Install Instructions
========================
1. Download CARLA 0.9.13: https://github.com/carla-simulator/carla/releases/tag/0.9.13
    - On Linux:
    ```
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
    mkdir CARLA_0.9.13
    tar -xvzf CARLA_0.9.13.tar.gz -C CARLA_0.9.13
    ```     

1. Download `scenario_runner` 0.9.13
    > **NOTE: Make sure `scenario_runner` is placed in the `<CARLA_ROOT>/PythonAPI/` directory.**
    ```
    git clone https://github.com/carla-simulator/scenario_runner
    cd scenario_runner
    git checkout v0.9.13
    ```

    - Install `scenario_runner` Python packages:
        ```
        pip install -r requirements.txt
        ```
        > If you get errors, try running this beforehand: `pip install --upgrade pip`
    - Apply patch to `scenario_runner`:
        ```
        git apply ../../../adversarial_carla_env/adv_scenario_runner.patch
        ```

1. Install `adversarial_carla_env` Python packages:
    ```
    cd <PATH_TO>/adversarial_carla_env
    pip install -r requirements.txt
     ```

     - Install `adv-carla-v0` gym environment (from the same directory):
    ```
    pip install -e .
    ```

1. Set up environment variables
    - For Windows, either do this every time:
    ```batch
    :: %CARLA_ROOT% is the CARLA installation directory
    :: %SCENARIO_RUNNER% is the ScenarioRunner installation directory
    :: In a build from source, the .egg files may be in: ${CARLA_ROOT}/PythonAPI/dist/ instead of ${CARLA_ROOT}/PythonAPI

    :: <<<<CHANGE THIS TO YOUR CARLA_ROOT, EVERYTHING ELSE IS UNCHANGED!>>>>
    set CARLA_ROOT=<YOUR_PATH_TO>\CARLA_0.9.13

    set SCENARIO_RUNNER_ROOT=%CARLA_ROOT%\PythonAPI\scenario_runner
    set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg
    set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\agents
    set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla
    set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI

    ```

    - (Windows) Or, add these to your permanent environment variables:
        - `CARLA_ROOT`

            - `<YOUR_PATH_TO>\CARLA_0.9.13`
        - `SCENARIO_RUNNER_ROOT`

            - `%CARLA_ROOT%\PythonAPI\scenario_runner`
        - `PYTHONPATH` (append this):
        
            ```batch
            ;%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.13-py3.7-win-amd64.egg;%CARLA_ROOT%\PythonAPI\carla\agents;%CARLA_ROOT%\PythonAPI\carla;%CARLA_ROOT%\PythonAPI
            ```

    - For Linux:
    ```bash

    # <<<<CHANGE THIS TO YOUR CARLA_ROOT, EVERYTHING ELSE IS UNCHANGED!>>>>
    export CARLA_ROOT=/path/to/CARLA_0.9.13

    export SCENARIO_RUNNER_ROOT=$CARLA_ROOT/PythonAPI/scenario_runner
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/agents
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
    export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI
    ```



Running Instructions
========================
1. Start CARLA using the `carla_start.sh` (Unix) or `carla_start.bat` (Windows) scripts.
1. Execute stress testing and record the final trajectory (requires setting environment variables, see above).
    ```
    python test/run_episode.py
    ```
> **Note**: you may need to specify the correct python verion:
> ```
> sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
> ```
