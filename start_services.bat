@echo off
echo Starting Services...

:: Kill any existing processes
taskkill /F /IM python.exe /IM locust.exe /IM streamlit.exe 2>nul

:: Start Locust
echo Starting Locust...
start "Locust" cmd /k "locust -f locustfile.py --host http://localhost:8000"

:: Wait for Locust to start
timeout /t 10 /nobreak >nul

:: Open Locust web interface in browser
echo Opening Locust web interface...
start http://localhost:8089

:: Start Streamlit
echo Starting Streamlit...
start "Streamlit" cmd /k "streamlit run evaluation_dashboard.py"

:: Wait for Streamlit to start
timeout /t 10 /nobreak >nul

:: Open Streamlit dashboard in browser
echo Opening Streamlit dashboard...
start http://localhost:8501

echo Services started!
echo - Locust web interface at http://localhost:8089
echo - Streamlit dashboard at http://localhost:8501
echo.
echo Press any key to close all windows...
pause >nul

:: Close all windows
taskkill /F /IM python.exe /IM locust.exe /IM streamlit.exe 2>nul 