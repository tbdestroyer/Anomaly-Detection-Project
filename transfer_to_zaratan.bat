@echo off
echo Transferring files to Zaratan...

:: Replace with your UMD Directory ID
set UMD_USER=your_directory_id

:: Replace with your project directory on Zaratan
set ZARATAN_DIR=/lustre/scratch/users/%UMD_USER%/anomaly_detection

:: Create directory on Zaratan
echo Creating directory on Zaratan...
ssh %UMD_USER%@login.zaratan.umd.edu "mkdir -p %ZARATAN_DIR%"

:: Transfer files using sftp
echo Transferring project files...
echo cd %ZARATAN_DIR% > sftp_commands.txt
echo put *.py >> sftp_commands.txt
echo put *.sh >> sftp_commands.txt
echo put *.csv >> sftp_commands.txt
echo put *.bat >> sftp_commands.txt
echo put *.md >> sftp_commands.txt
echo put requirements.txt >> sftp_commands.txt
echo mkdir models >> sftp_commands.txt
echo cd models >> sftp_commands.txt
echo lcd models >> sftp_commands.txt
echo put models\*.py >> sftp_commands.txt
echo mkdir logs >> sftp_commands.txt
echo mkdir outputs >> sftp_commands.txt
echo mkdir outputs/drift_plots >> sftp_commands.txt
echo quit >> sftp_commands.txt

sftp -b sftp_commands.txt %UMD_USER%@login.zaratan.umd.edu

:: Submit training job
echo Submitting training job...
ssh %UMD_USER%@login.zaratan.umd.edu "cd %ZARATAN_DIR% && sbatch zaratan_train.sh"

:: Wait for training to complete
echo Waiting for training to complete...
echo Check job status with: ssh %UMD_USER%@login.zaratan.umd.edu "squeue -u %UMD_USER%"
echo Press any key when training is complete to submit inference job...
pause >nul

:: Submit inference job
echo Submitting inference job...
ssh %UMD_USER%@login.zaratan.umd.edu "cd %ZARATAN_DIR% && sbatch zaratan_infer.sh"

echo Jobs submitted!
echo Check job status with: ssh %UMD_USER%@login.zaratan.umd.edu "squeue -u %UMD_USER%"
echo View logs with: ssh %UMD_USER%@login.zaratan.umd.edu "cd %ZARATAN_DIR% && cat train_log.out infer_log.out"

:: Clean up
del sftp_commands.txt

"C:\Program Files\PuTTY\pscp.exe" -r tbulbul@login.zaratan.umd.edu:~/anomaly_detection/outputs/* outputs/

"C:\Program Files\PuTTY\pscp.exe" zaratan_api.sh tbulbul@login.zaratan.umd.edu:~/anomaly_detection/ 