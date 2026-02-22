@echo off
echo Starting Free Deepfake Detection Backend...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install requirements
echo Installing dependencies...
pip install -r requirements_free.txt

REM Start the server
echo.
echo Starting API server on http://localhost:5000
echo.
python deepfake_free.py

pause

