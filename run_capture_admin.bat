@echo off
:: Run realtime_capture.py as Administrator
:: Right-click this file and select "Run as administrator"
:: OR double-click and accept the UAC prompt

PowerShell -Command "Start-Process powershell -ArgumentList '-NoExit -Command cd ''%~dp0''; python realtime_capture.py' -Verb RunAs"
