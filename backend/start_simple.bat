@echo off
echo Starting Simple Deepfake Detector (Fast Install)...
echo.

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing lightweight dependencies (no PyTorch - much faster)...
pip install -r requirements_simple.txt

echo.
echo Starting API server on http://localhost:5000
echo.
python deepfake_simple.py

pause

