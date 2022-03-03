@ECHO OFF
start %CARLA_ROOT%\CarlaUE4.exe -carla-rpc-port=3000 -windowed -ResX=700 -ResY=500 -benchmark -fps=10 
rem -quality-level=Low
