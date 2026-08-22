@echo off
rem ============================================================
rem  Subtitle Studio - debug launcher (console + verbose logs)
rem ============================================================
setlocal
cd /d "%~dp0"
if not exist "runtime\python.exe" (
    echo [ERROR] runtime\python.exe not found.
    pause
    exit /b 1
)
"runtime\python.exe" "launcher.py" --debug %*
echo.
echo Subtitle Studio exited. Press any key to close this window...
pause >nul
endlocal
