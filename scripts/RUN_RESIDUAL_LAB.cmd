@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
pushd "%~dp0" || exit /b 1
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONSAFEPATH="
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "RESULT=1"

if defined TRACELIGHT_PYTHON goto explicit
py -3.12 -c "import sysconfig; raise SystemExit(sysconfig.get_platform() != 'win-amd64')" >nul 2>&1
if not errorlevel 1 goto launcher
python -c "import sys,sysconfig; raise SystemExit(sys.version_info[:2] != (3,12) or sysconfig.get_platform() != 'win-amd64')" >nul 2>&1
if not errorlevel 1 goto installed
echo Windows x64 Python 3.12 is required. If already installed, set TRACELIGHT_PYTHON to its python.exe.
goto finish

:explicit
"%TRACELIGHT_PYTHON%" -s -m scripts.residual_lab.bootstrap %*
set "RESULT=%ERRORLEVEL%"
goto finish

:launcher
py -3.12 -s -m scripts.residual_lab.bootstrap %*
set "RESULT=%ERRORLEVEL%"
goto finish

:installed
python -s -m scripts.residual_lab.bootstrap %*
set "RESULT=%ERRORLEVEL%"

:finish
popd
if "%TRACELIGHT_NO_PAUSE%"=="1" exit /b %RESULT%
echo Exit code %RESULT%
pause
exit /b %RESULT%
