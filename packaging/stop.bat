@echo off
rem ============================================================
rem  Subtitle Studio - graceful stop
rem  Creates stop.flag; launcher polls it and shuts the WebUI
rem  down cleanly (HTTP server closed, GPU memory released).
rem ============================================================
setlocal
cd /d "%~dp0"
type nul > "stop.flag"
echo Stop requested. Subtitle Studio will exit within a few seconds.
timeout /t 3 /nobreak >nul
endlocal
