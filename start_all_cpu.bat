@echo off
title StemSplit + AceStep AI (CPU inference)
echo.
echo  ================================================
echo    StemSplit with AI Remix  [CPU inference mode]
echo  ================================================
echo.
echo  AI generation runs on CPU instead of GPU.
echo  No VRAM limit, but generation takes several minutes per clip.
echo.

set STEM_DIR=%~dp0
set AS_DIR=%~dp0..\ACE-Step-1.5

if not exist "%AS_DIR%" (
  echo  AceStep not found. Run setup_acestep.bat first.
  echo.
  pause & exit /b 1
)

echo  Starting AceStep Worker (CPU mode -- loading models into RAM)...
echo  Note: First run downloads ~5GB. Model loading takes longer on CPU.
echo.
start "AceStep Worker [CPU]" cmd /k "cd /d "%AS_DIR%" && uv run --with flask python "%STEM_DIR%acestep_worker.py" --acestep-dir . --cpu-inference && pause"

echo  Waiting 4 seconds...
timeout /t 4 /nobreak > nul

echo  Starting StemSplit...
start "StemSplit" cmd /k "cd /d "%STEM_DIR%" && python app.py && pause"

echo.
echo  Both servers starting in separate windows.
echo  StemSplit:       http://localhost:5000
echo  AceStep Worker:  http://localhost:5001/health  (CPU mode)
echo.
echo  AI generation will take several minutes per clip on CPU.
echo  The worker window shows progress.
echo.
pause
