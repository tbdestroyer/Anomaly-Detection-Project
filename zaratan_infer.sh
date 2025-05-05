#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 2
#SBATCH --mem-per-cpu=2G
#SBATCH --oversubscribe
#SBATCH --output=infer_log.out
#SBATCH --error=infer_error.out

# Load environment
. ~/.bashrc

# Load required modules
module load python/3.10
module load cuda/11.7

# Create necessary directories in scratch space
SCRATCH_DIR="/lustre/scratch/users/$USER/anomaly_detection"
mkdir -p $SCRATCH_DIR/logs $SCRATCH_DIR/outputs $SCRATCH_DIR/outputs/drift_plots

# Change to scratch directory
cd $SCRATCH_DIR

# Install requirements in a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run drift detection
echo "Running drift detection..."
python drift_detection.py

# Start FastAPI server in background
echo "Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Wait for server to start
sleep 10

# Run load testing with proper resource allocation
echo "Running load testing..."
locust -f locustfile.py --host http://localhost:8000 --headless --users 100 --spawn-rate 10 --run-time 5m --worker

# Generate performance report
echo "Generating performance report..."
python generate_benchmark_report.py

# Copy results to home directory for backup
cp -r $SCRATCH_DIR/outputs/* ~/anomaly_detection/outputs/
cp -r $SCRATCH_DIR/logs/* ~/anomaly_detection/logs/

echo "Inference testing completed successfully!" 