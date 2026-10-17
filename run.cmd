@echo off
setlocal
rem ============================================================================
rem  openVideoAI launcher
rem
rem  Always runs the app with the project virtual environment (.venv) so you
rem  don't have to remember the long python path or worry about which Python
rem  `py` / `python` resolves to. All arguments are forwarded to main.py.
rem
rem  Usage:
rem    run                      Desktop window (default)
rem    run --mode server        Server only  (for the Vite dev server)
rem    run --mode server --port 8000
rem    run --help
rem ============================================================================

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [openVideoAI] No virtual environment found at .venv
  echo [openVideoAI] Creating one and installing dependencies ^(this may take a while^)...
  py -3 -m venv "%ROOT%.venv" || goto :fail
  "%PY%" -m pip install --upgrade pip || goto :fail
  "%PY%" -m pip install -r "%ROOT%requirements.txt" || goto :fail
)

"%PY%" "%ROOT%main.py" %*
exit /b %ERRORLEVEL%

:fail
echo [openVideoAI] Setup failed. See the messages above.
exit /b 1
