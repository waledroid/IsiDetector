@echo off
REM Daily start — double-click entry point. Calls up.ps1 with execution-
REM policy bypass so PowerShell's default restrictions don't block a
REM non-signed script. No admin rights needed for daily runs.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\windows\up.ps1" %*
if errorlevel 1 pause
