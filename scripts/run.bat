@echo off
setlocal

REM Change to the folder where this .bat lives
cd /d "%~dp0"\..

REM Run the Python script from the installed package location
python -m hr_energy_lab.hr_interactive

REM Keep window open so you can see any messages
pause
