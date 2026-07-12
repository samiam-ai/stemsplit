@echo off
title Stopping StemSplit...
echo.
echo  ================================================
echo    Stopping StemSplit Services
echo  ================================================
echo.

REM ── Close the three terminal windows by exact title ──────────────────
echo  Closing terminal windows...

REM Use PowerShell for reliable title-based matching
powershell -NoProfile -Command ^
  "Get-Process | Where-Object { $_.MainWindowTitle -in @('StemSplit + AceStep AI','AceStep Worker','StemSplit') } | Stop-Process -Force" ^
  > nul 2>&1

REM Fallback: taskkill by exact window title in case PowerShell is restricted
taskkill /fi "WindowTitle eq StemSplit + AceStep AI" /f > nul 2>&1
taskkill /fi "WindowTitle eq AceStep Worker"         /f > nul 2>&1
taskkill /fi "WindowTitle eq StemSplit"              /f > nul 2>&1

REM ── Kill any remaining Python processes on ports 5000 / 5001 ─────────
echo  Releasing ports 5000 and 5001...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000 " ^| findstr "LISTENING"') do (
    taskkill /pid %%a /f > nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5001 " ^| findstr "LISTENING"') do (
    taskkill /pid %%a /f > nul 2>&1
)

timeout /t 1 /nobreak > nul

REM ── Confirm ports are free ────────────────────────────────────────────
netstat -aon | findstr ":5000 " | findstr "LISTENING" > nul 2>&1
if %errorlevel%==0 (
    echo  WARNING: Port 5000 still in use. Try running as Administrator.
) else (
    echo  Port 5000 free.
)

netstat -aon | findstr ":5001 " | findstr "LISTENING" > nul 2>&1
if %errorlevel%==0 (
    echo  WARNING: Port 5001 still in use. Try running as Administrator.
) else (
    echo  Port 5001 free.
)

echo.
echo  All done. Run start_all.bat to restart.
echo.
timeout /t 2 /nobreak > nul
exit
