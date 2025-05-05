#!/bin/bash
#SBATCH -t 01:00:00
#SBATCH -n 4
#SBATCH --mem-per-cpu=4G
#SBATCH --oversubscribe
#SBATCH --output=train_log.out
#SBATCH --error=train_error.out

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

# Train individual models in parallel using GNU parallel
echo "Training individual models..."
parallel --jobs 4 ::: \
    "python models/train_isolation_forest.py" \
    "python models/train_oneclass_svm.py" \
    "python models/train_elliptic_envelope.py" \
    "python models/train_lof.py"

# Train autoencoder (GPU-accelerated)
echo "Training Autoencoder..."
python models/train_autoencoder.py

# Train ensemble models in parallel
echo "Training ensemble models..."
parallel --jobs 3 ::: \
    "python models/ensemble_light.py" \
    "python models/ensemble_medium.py" \
    "python models/ensemble_full.py"

# Generate reports
echo "Generating evaluation reports..."
python evaluate_model.py
python generate_benchmark_report.py
python generate_full_report.py

# Copy results to home directory for backup
cp -r $SCRATCH_DIR/outputs/* ~/anomaly_detection/outputs/
cp -r $SCRATCH_DIR/logs/* ~/anomaly_detection/logs/

echo "Training completed successfully!" 