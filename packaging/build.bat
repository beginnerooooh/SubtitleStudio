@echo off
rem ============================================================
rem  Subtitle Studio - one-click portable build (run on Windows)
rem
rem  Examples:
rem    build.bat                          CPU build, no models
rem    build.bat --torch-index cu124      GPU (CUDA 12.4) build
rem    build.bat --with-models --preset full --installer
rem                                       offline full build + Setup.exe
rem
rem  All arguments are forwarded to packaging\build_portable.py
rem ============================================================
setlocal
cd /d "%~dp0\.."
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python not found in PATH. Install Python 3.10+ first.
    exit /b 1
)
python packaging\build_portable.py %*
endlocal
