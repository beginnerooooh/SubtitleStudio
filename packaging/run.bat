@echo off
rem ============================================================
rem  Subtitle Studio - silent launcher (no console window)
rem  The console is hidden by launcher.py right after start.
rem  Pass --debug to keep it visible, or use run-debug.bat.
rem ============================================================
setlocal
cd /d "%~dp0"
if not exist "runtime\python.exe" (
    echo [ERROR] runtime\python.exe not found.
    echo This looks like a broken installation. Please reinstall.
    pause
    exit /b 1
)
"runtime\python.exe" "launcher.py" %*
endlocal
exit /b 0
