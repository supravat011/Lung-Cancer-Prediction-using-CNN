@echo off
echo ============================================================
echo   Lung Cancer Prediction - Quick Start
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import tensorflow" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    echo This may take a few minutes...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo Dependencies already installed!
)

echo.
echo ============================================================
echo   Starting Prediction Script
echo ============================================================
echo.

REM Run the prediction script
python predict.py

pause
