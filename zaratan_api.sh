#!/bin/bash
#SBATCH -t 04:00:00  # 4 hours runtime
#SBATCH -n 1         # 1 core
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=api_log.out
#SBATCH --error=api_error.out

# Load environment
. ~/.bashrc

# Load required modules
module load python/3.10.10/gcc/11.3.0/nocuda/linux-rhel8-zen2

# Use your home directory
SCRATCH_DIR="/home/tbulbul/anomaly_detection"
mkdir -p $SCRATCH_DIR/logs $SCRATCH_DIR/outputs

# Change to scratch directory
cd $SCRATCH_DIR

# Only create venv if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install --no-cache-dir -r requirements.txt
else
    source venv/bin/activate
fi

# Start FastAPI server (do NOT use & at the end)
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000

echo "API server started successfully!" 