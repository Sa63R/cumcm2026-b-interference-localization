@echo off
chcp 65001 >nul
(
ver
whoami
where python
where py
dir C:\Users\baiwc\Downloads /b
echo %DATE% %TIME%
)>C:\Users\baiwc\Downloads\q4_environment_probe.txt 2>&1
