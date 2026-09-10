@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_session.ps1" -Problem 3 -Mode practice
pause
