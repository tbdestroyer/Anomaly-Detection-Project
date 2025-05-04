@echo off
echo Starting Anomaly Detection Project...

:: Create necessary directories
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs

:: Step 1: Train individual models
echo Training individual models...
echo Training Isolation Forest...
python models/train_isolation_forest.py
if errorlevel 1 (
    echo Error training Isolation Forest
    pause
    exit /b 1
)

echo Training One-Class SVM...
python models/train_oneclass_svm.py
if errorlevel 1 (
    echo Error training One-Class SVM
    pause
    exit /b 1
)

echo Training Elliptic Envelope...
python models/train_elliptic_envelope.py
if errorlevel 1 (
    echo Error training Elliptic Envelope
    pause
    exit /b 1
)

echo Training Local Outlier Factor...
python models/train_lof.py
if errorlevel 1 (
    echo Error training Local Outlier Factor
    pause
    exit /b 1
)

echo Training Autoencoder...
python models/train_autoencoder.py
if errorlevel 1 (
    echo Error training Autoencoder
    pause
    exit /b 1
)

:: Step 2: Train ensemble models
echo Training ensemble models...
echo Training Light Ensemble...
python models/ensemble_light.py
if errorlevel 1 (
    echo Error training Light Ensemble
    pause
    exit /b 1
)

echo Training Medium Ensemble...
python models/ensemble_medium.py
if errorlevel 1 (
    echo Error training Medium Ensemble
    pause
    exit /b 1
)

echo Training Full Ensemble...
python models/ensemble_full.py
if errorlevel 1 (
    echo Error training Full Ensemble
    pause
    exit /b 1
)

:: Step 3: Start FastAPI server in a new window
echo Starting FastAPI server...
start "FastAPI Server" cmd /k "uvicorn main:app --reload"

:: Wait for FastAPI server to start
timeout /t 10 /nobreak >nul

:: Step 4: Start Locust in a new window
echo Starting Locust load testing...
start "Locust" cmd /k "locust -f locustfile.py --host http://localhost:8000"

:: Wait for Locust to start
timeout /t 5 /nobreak >nul

:: Step 5: Start Streamlit dashboard in a new window
echo Starting Streamlit dashboard...
start "Streamlit Dashboard" cmd /k "streamlit run evaluation_dashboard.py"

echo All components started!
echo - FastAPI server running at http://localhost:8000
echo - Locust web interface at http://localhost:8089
echo - Streamlit dashboard at http://localhost:8501
echo.
echo Press any key to close all windows...
pause >nul

:: Close all windows
taskkill /FI "WINDOWTITLE eq FastAPI Server*" /F
taskkill /FI "WINDOWTITLE eq Locust*" /F
taskkill /FI "WINDOWTITLE eq Streamlit Dashboard*" /F 