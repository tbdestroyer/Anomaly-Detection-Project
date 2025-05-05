@echo off
echo Starting Anomaly Detection Project...

:: Create necessary directories
if not exist "logs" mkdir logs
if not exist "outputs" mkdir outputs
if not exist "outputs\drift_plots" mkdir outputs\drift_plots

:: Step 1: Data Preparation
echo Preparing data...
echo Preprocessing credit card data...
python creditcard_preprocess.py
if errorlevel 1 (
    echo Error preprocessing data
    pause
    exit /b 1
)

echo Scaling features...
python feature_scaling.py
if errorlevel 1 (
    echo Error scaling features
    pause
    exit /b 1
)

echo Splitting data into training and API sets...
python data_split.py
if errorlevel 1 (
    echo Error splitting data
    pause
    exit /b 1
)

echo Analyzing data and generating visualizations...
python data_analysis.py
if errorlevel 1 (
    echo Error in data analysis
    pause
    exit /b 1
)

:: Step 2: Train individual models
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

:: Step 3: Evaluate models and generate reports
echo Evaluating models and generating reports...
python evaluate_model.py
if errorlevel 1 (
    echo Error evaluating models
    pause
    exit /b 1
)

echo Generating benchmark report...
python generate_benchmark_report.py
if errorlevel 1 (
    echo Error generating benchmark report
    pause
    exit /b 1
)

echo Generating full report...
python generate_full_report.py
if errorlevel 1 (
    echo Error generating full report
    pause
    exit /b 1
)

:: Step 4: Train ensemble models
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

:: Step 5: Run drift detection
echo Running drift detection...
python drift_detection.py
if errorlevel 1 (
    echo Warning: Drift detection completed with issues
)

:: Step 6: Start FastAPI server in a new window
echo Starting FastAPI server...
start "FastAPI Server" cmd /k "uvicorn main:app --reload --log-level debug"

:: Wait for FastAPI server to start and models to load
echo Waiting for models to load (this may take a few minutes)...
timeout /t 30 /nobreak >nul

:: Check if FastAPI is responding and models are loaded
echo Checking if FastAPI server is ready...
curl -s http://localhost:8000/ >nul
if errorlevel 1 (
    echo FastAPI server is not responding. Please check the server window for errors.
    pause
    exit /b 1
)

:: Test API prediction endpoint
echo Testing API prediction endpoint...
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"data\":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}" >nul
if errorlevel 1 (
    echo Error: API prediction endpoint is not working correctly
    pause
    exit /b 1
)
echo FastAPI server is ready and working correctly!

:: Step 7: Start Locust in a new window
echo Starting Locust load testing...
echo Verifying API is ready for Locust...
curl -s http://localhost:8000/ >nul
if errorlevel 1 (
    echo Error: API is not ready for Locust testing
    pause
    exit /b 1
)

echo Starting Locust with default parameters...
start "Locust" cmd /k "locust -f locustfile.py --host http://localhost:8000 --headless --users 10 --spawn-rate 1 --run-time 1m"

:: Wait for Locust to start
timeout /t 5 /nobreak >nul

:: Verify Locust is running
echo Checking if Locust is running...
curl -s http://localhost:8089/ >nul
if errorlevel 1 (
    echo Warning: Locust web interface is not accessible. Continuing anyway...
)

:: Step 8: Start Streamlit dashboard in a new window
echo Starting Streamlit dashboard...
start "Streamlit Dashboard" cmd /k "streamlit run evaluation_dashboard.py"

:: Wait for Streamlit to start
timeout /t 5 /nobreak >nul

:: Open dashboards in browser
echo Opening monitoring dashboards...
start http://localhost:8089
start http://localhost:8501

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