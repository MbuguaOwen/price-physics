@echo off
REM Local project shim to use Python 3.11 via py launcher
REM Ensures `python ...` commands run with venv-friendly interpreter
py -3.11 %*
