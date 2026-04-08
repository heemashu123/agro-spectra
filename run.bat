@echo off
:: =============================================================================
:: Agro-Spectra | One-Click Windows Launcher
:: =============================================================================
:: Detects Python, installs dependencies, generates data, and trains the agent.
:: =============================================================================

title Agro-Spectra RL Launcher

echo.
echo  ============================================================
echo   Agro-Spectra ^| Precision Agriculture RL Ecosystem
echo   Meta PyTorch ^& OpenEnv Hackathon Submission
echo  ============================================================
echo.

:: ---- Find Python ------------------------------------------------------------
set PYTHON_CMD=
where python >nul 2>&1 && set PYTHON_CMD=python
if "%PYTHON_CMD%"=="" where python3 >nul 2>&1 && set PYTHON_CMD=python3
if "%PYTHON_CMD%"=="" if exist "D:\Python311\python.exe" set PYTHON_CMD=D:\Python311\python.exe
if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python not found. Please install Python 3.9+ and re-run.
    echo         Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [INFO] Using Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

:: ---- Install dependencies ---------------------------------------------------
echo [STEP 1/3] Installing dependencies...
%PYTHON_CMD% -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo [OK] Dependencies ready.
echo.

:: ---- Generate synthetic dataset --------------------------------------------
echo [STEP 2/3] Generating synthetic farm dataset...
%PYTHON_CMD% data_generator.py
if errorlevel 1 (
    echo [ERROR] Data generation failed.
    pause
    exit /b 1
)
echo.

:: ---- Train PPO agent --------------------------------------------------------
echo [STEP 3/3] Training PPO agent (50,000 timesteps)...
echo            This will take approx. 5-10 minutes on CPU.
echo.
%PYTHON_CMD% train_agent.py
if errorlevel 1 (
    echo [ERROR] Training failed. See output above for details.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Training complete! Files generated:
echo     agro_ppo_model.zip     - Final trained model
echo     best_agro_model/       - Best checkpoint during training
echo     mock_farm_data.csv     - Synthetic farm dataset
echo  ============================================================
echo.
pause
