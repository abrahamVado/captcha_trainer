@echo off
setlocal enabledelayedexpansion

:: CONFIGURATION
set PYTHON_VERSION=3.10.11
set PYTHON_INSTALLER=python-%PYTHON_VERSION%-amd64.exe
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/%PYTHON_INSTALLER%

:: STEP 1: Download Python
echo 🔽 Downloading Python %PYTHON_VERSION% installer...
curl -L -o %PYTHON_INSTALLER% %PYTHON_URL%

:: STEP 2: Install Python system-wide
echo 🧩 Installing Python %PYTHON_VERSION% (system-wide)...
%PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

:: STEP 3: Wait for Python to be available
echo 🕒 Waiting for Python installation...
timeout /t 10 > nul

:: STEP 4: Verify Python is available
python --version || (
    echo ❌ Python not detected. Please restart terminal or check PATH.
    pause
    exit /b 1
)

:: STEP 5: Upgrade pip and install packages globally
echo 📦 Installing packages system-wide...
python -m pip install --upgrade pip
pip install tensorflow==2.11 opencv-python pillow numpy scikit-learn
pip install tqdm


echo ✅ Setup complete!
pause
