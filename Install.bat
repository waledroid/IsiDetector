@echo off
REM First-time install — double-click entry point. Auto-elevates and runs
REM install.ps1 with execution-policy bypass. Docker Desktop install
REM requires admin, hence the elevation prompt the operator will see.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','%~dp0deploy\windows\install.ps1'"
