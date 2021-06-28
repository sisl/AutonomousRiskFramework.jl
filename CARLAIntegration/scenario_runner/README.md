ScenarioRunner for CARLA (Modified for Adaptive Stress Testing)
========================
Modified from [ScenarioRunner](https://github.com/carla-simulator/scenario_runner)

TODO: connect as fork of the main repository

Instructions
========================
1. Launch CARLA server
```
.\CarlaUE4.exe -carla-rpc-port=2222 -windowed -ResX=320 -ResY=240 -benchmark -fps=10 -quality-level=Low
```

2. Load configs (remove no-rendering if needed)
```
python config.py --map Town01 --port 2222 --spectator-loc 80.37 25.30 0.0 --no-rendering
```

3. Execute stress testing and record the final trajectory (requires setting environment variables, follow [env_set.txt](https://github.com/sisl/AutonomousRiskFramework/blob/shubh/carla_integration/CARLAIntegration/scenario_runner/env_set.txt))
```
python scenario_runner_ast.py --route .\srunner\data\routes_ast.xml .\srunner\data\ast_scenarios.json --port 2222 --agent .\srunner\autoagents\ast_agent.py --record recordings
```
