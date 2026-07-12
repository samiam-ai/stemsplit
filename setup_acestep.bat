@echo off
title StemSplit - AceStep AI Setup
color 0B
echo.
echo  ====================================================
echo    AceStep AI Setup
echo    This installs the AI remix engine (~5GB download)
echo  ====================================================
echo.

REM Check for git
git --version >nul 2>&1
if errorlevel 1 (
  echo  ERROR: Git not found. Install from https://git-scm.com/
  pause & exit /b 1
)

REM Check for uv
uv --version >nul 2>&1
if errorlevel 1 (
  echo  Installing uv package manager...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  echo  Please CLOSE and REOPEN this window, then run setup_acestep.bat again.
  pause & exit /b 0
)

REM Go up one level from stemsplit folder to clone alongside it
cd /d "%~dp0.."

if exist "ACE-Step-1.5" (
  echo  ACE-Step-1.5 folder already exists. Pulling latest...
  cd ACE-Step-1.5
  git pull
) else (
  echo  Cloning ACE-Step-1.5...
  git clone https://github.com/ACE-Step/ACE-Step-1.5.git
  cd ACE-Step-1.5
)

echo.
echo  Installing Python dependencies (this may take a few minutes)...
uv sync

echo.
echo  ====================================================
echo    AceStep setup complete!
echo    Model weights (~5GB) download automatically on
echo    first use when you run start_all.bat
echo  ====================================================
echo.
pause
