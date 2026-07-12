@echo off
title StemSplit + AceStep AI
echo.
echo  ================================================
echo    StemSplit with AI Remix
echo  ================================================

set STEM_DIR=%~dp0
set AS_DIR=%~dp0..\ACE-Step-1.5

REM Check AceStep exists
if not exist "%AS_DIR%" (
  echo.
  echo  AceStep not found. Run setup_acestep.bat first.
  echo  Or use start.bat to run StemSplit without AI Remix.
  echo.
  pause & exit /b 1
)

echo.
echo  Starting AceStep Worker (loading GPU models)...
echo  Note: First run downloads ~5GB - this window will show progress.
echo.
start "AceStep Worker" cmd /k "cd /d "%AS_DIR%" && uv run --with flask python "%STEM_DIR%acestep_worker.py" --acestep-dir . && pause"

echo  Waiting 4 seconds...
timeout /t 4 /nobreak > nul

echo  Starting StemSplit...
start "StemSplit" cmd /k "cd /d "%STEM_DIR%" && python app.py && pause"

echo.
echo  Both servers starting in separate windows.
echo  StemSplit:       http://localhost:5000  (opens in browser)
echo  AceStep Worker:  http://localhost:5001/health
echo.
echo  The AI Remix panel shows "Loading models..." until AceStep is ready.
echo  First run takes longer due to model download.
echo.
pause
