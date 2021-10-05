@ECHO OFF
rem # %CARLA_ROOT% is the CARLA installation directory
rem # %SCENARIO_RUNNER% is the ScenarioRunner installation directory
rem # <VERSION> is the correct string for the Python version being used
rem # In a build from source, the .egg files may be in: ${CARLA_ROOT}/PythonAPI/dist/ instead of ${CARLA_ROOT}/PythonAPI

rem <<<<CHANGE THIS TO YOUR CARLA_ROOT, EVERYTHING ELSE IS UNCHANGED!>>>>
set CARLA_ROOT=C:\Users\mossr\Code\sisl\ast\Allstate\AutonomousRiskFramework\CARLAIntegration\CARLA_0.9.11

set SCENARIO_RUNNER_ROOT=%CARLA_ROOT%\PythonAPI\scenario_runner
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\dist\carla-0.9.11-py3.7-win-amd64.egg
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla\agents
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI\carla
set PYTHONPATH=%PYTHONPATH%;%CARLA_ROOT%\PythonAPI