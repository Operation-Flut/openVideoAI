@echo off
setlocal
rem ============================================================================
rem  openVideoAI - local Windows executable build (PyInstaller)
rem
rem  Produces a self-contained desktop build with the project virtual
rem  environment so the result matches the GitHub release workflow
rem  (.github/workflows/release.yml -> "Build Windows").
rem
rem  The app serves the compiled SPA from UI\dist, so the UI is built first and
rem  then bundled next to the executable by openVideoAI.spec.
rem
rem  Usage:
rem    build_exe             Build dist\openVideoAI\openVideoAI.exe   (onedir)
rem    build_exe portable    Also build dist\openVideoAI-portable.exe (onefile)
rem ============================================================================

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo [build] No virtual environment found at .venv
  echo [build] Run run.cmd once to create it and install dependencies, then retry.
  exit /b 1
)

rem --- 1. Build the UI (the app serves UI\dist) -------------------------------
where npm >nul 2>nul
if errorlevel 1 (
  echo [build] npm was not found on PATH. Install Node.js 22+ and retry.
  exit /b 1
)

pushd "%ROOT%UI"
if exist "package-lock.json" goto :npm_ci
call npm install || goto :fail
goto :npm_build
:npm_ci
call npm ci || goto :fail
:npm_build
call npm run build || goto :fail
popd

if not exist "%ROOT%UI\dist\index.html" (
  echo [build] UI build did not produce UI\dist\index.html.
  exit /b 1
)
echo [build] UI built.

rem --- 2. Ensure PyInstaller is available -------------------------------------
"%PY%" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
  echo [build] Installing PyInstaller...
  "%PY%" -m pip install pyinstaller || goto :fail
)

rem --- 3. Build the onedir bundle from the checked-in spec --------------------
echo [build] Running PyInstaller ^(openVideoAI.spec^)...
"%PY%" -m PyInstaller --noconfirm --clean "%ROOT%openVideoAI.spec" || goto :fail

if not exist "%ROOT%dist\openVideoAI\openVideoAI.exe" (
  echo [build] Expected dist\openVideoAI\openVideoAI.exe was not created.
  exit /b 1
)
echo [build] Built dist\openVideoAI\openVideoAI.exe

rem --- 4. Optional portable single-file build ---------------------------------
if /I not "%~1"=="portable" goto :done
echo [build] Running PyInstaller ^(onefile portable^)...
"%PY%" -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name openVideoAI-portable ^
  --icon "%ROOT%assets\icon\black_glass.ico" ^
  --add-data "%ROOT%UI\dist;UI/dist" ^
  --add-data "%ROOT%assets;assets" ^
  --add-data "%ROOT%models.json;." ^
  --collect-all src --collect-all webview --collect-all uvicorn --collect-all fastapi ^
  --hidden-import webview --hidden-import webview.platforms.winforms --hidden-import clr ^
  "%ROOT%main.py" || goto :fail
echo [build] Built dist\openVideoAI-portable.exe

:done
echo [build] Done.
exit /b 0

:fail
echo [build] Build failed. See the messages above.
exit /b 1
