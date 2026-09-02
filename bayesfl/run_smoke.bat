@echo off
python main.py --config configs\smoke.yaml --experiment fig2 --methods fedavg,proposed
if errorlevel 1 exit /b %errorlevel%
python utils.py --input results --figure fig2
