#!/usr/bin/env bash
# ==============================================================================
# setup.sh
# Cloud Infrastructure & VM Automation for Genomic Fine-Tuning.
# ==============================================================================
#
# PROVISIONING COMMANDS (Execute locally on your laptop to create the GCP VM):
#
# NOTE: The generic 'pytorch-latest-gpu' family has been deprecated by GCP.
# You can query available PyTorch GPU image families using this command:
#   gcloud compute images list --project="deeplearning-platform-release" --no-standard-images --format="value(FAMILY)" | sort -u | grep pytorch
#
# Common available families include:
#   - pytorch-2-2-cu121-ubuntu-2204
#   - pytorch-2-9-cu129-ubuntu-2204-nvidia-580
#
# 1. Option A: NVIDIA L4 GPU (Highly Cost-Effective, ~ $0.90 - $1.00 / hour)
#    gcloud compute instances create dna-fine-tune-l4 \
#        --project="YOUR_PROJECT_ID" \
#        --zone="us-central1-a" \
#        --machine-type="g2-standard-8" \
#        --image-family="pytorch-2-2-cu121-ubuntu-2204" \
#        --image-project="deeplearning-platform-release" \
#        --accelerator="count=1,type=nvidia-l4" \
#        --maintenance-policy="TERMINATE" \
#        --boot-disk-size="150GB" \
#        --boot-disk-type="pd-ssd" \
#        --scopes="https://www.googleapis.com/auth/cloud-platform" \
#        --metadata="repo_url=https://github.com/your-username/dna_2.git,gcs_bucket=your-gcs-bucket-name" \
#        --metadata-from-file="startup-script=setup.sh"
#
# 2. Option B: NVIDIA A100 GPU (Maximum Performance, ~ $3.60 / hour)
#    gcloud compute instances create dna-fine-tune-a100 \
#        --project="YOUR_PROJECT_ID" \
#        --zone="us-central1-a" \
#        --machine-type="a2-highgpu-1g" \
#        --image-family="pytorch-2-2-cu121-ubuntu-2204" \
#        --image-project="deeplearning-platform-release" \
#        --accelerator="count=1,type=nvidia-tesla-a100" \
#        --maintenance-policy="TERMINATE" \
#        --boot-disk-size="200GB" \
#        --boot-disk-type="pd-ssd" \
#        --scopes="https://www.googleapis.com/auth/cloud-platform" \
#        --metadata="repo_url=https://github.com/your-username/dna_2.git,gcs_bucket=your-gcs-bucket-name" \
#        --metadata-from-file="startup-script=setup.sh"
# ==============================================================================

# Redirect standard output and error to a log file for diagnostics
exec > >(tee -i /var/log/startup-script-execution.log) 2>&1

echo "=================================================="
echo "Starting GCP Deep Learning VM Automated Provisioning Setup"
echo "Timestamp: $(date)"
echo "=================================================="

# 1. Fetch parameters from GCP instance metadata
echo "Fetching configuration from metadata server..."
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
HEADER="Metadata-Flavor: Google"

REPO_URL=$(curl -s -H "${HEADER}" "${METADATA_URL}/repo_url")
GCS_BUCKET=$(curl -s -H "${HEADER}" "${METADATA_URL}/gcs_bucket")

if [ -z "${REPO_URL}" ]; then
    echo "Error: 'repo_url' attribute not set in instance metadata. Exiting script."
    exit 1
fi

echo "Repository Target: ${REPO_URL}"
echo "Storage Bucket: ${GCS_BUCKET:-'Not Provided (Skipping GCS backups)'}"

# 2. System updates and required base packages
echo "Updating packages..."
apt-get update -y
apt-get install -y git git-lfs python3-pip python3-venv

# Ensure Git LFS is active (some genomic files are large)
git lfs install

# 3. Clone Repository
echo "Cloning codebase..."
WORK_DIR="/opt/dna_classifier"
rm -rf "${WORK_DIR}"
git clone "${REPO_URL}" "${WORK_DIR}"
cd "${WORK_DIR}" || exit 1

# 4. Set up python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 5. Install dependencies
echo "Installing project dependencies..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Installing standard training stack manually."
    pip install torch transformers accelerate scikit-learn pandas numpy requests google-cloud-storage einops tqdm
fi

# 6. Stream and Preprocess the Real Genomic Dataset
# Scan depth is configured high to build a robust dataset for deep fine-tuning
echo "Running DNA preprocessing and streaming engine..."
python preprocess.py --limit 1000 --flank 50 --output_file "./processed_variants.csv"

# 7. Start Fine-Tuning Execution
# Optimizations: Uses bf16 if NVIDIA L4 or A100 is present (Tensor Cores supported)
# Enables gradient accumulation to handle deep context lengths on VRAM
echo "Starting model fine-tuning..."
python train.py \
    --model_name "zhihan1996/DNABERT-2-117M" \
    --data_path "./processed_variants.csv" \
    --use_mock "False" \
    --epochs 3 \
    --batch_size 8 \
    --accum_steps 4 \
    --bf16 "True" \
    --output_dir "./outputs" \
    --gcs_bucket "${GCS_BUCKET}" \
    --gcs_prefix "runs/$(date +%Y%m%d_%H%M%S)"

echo "Training sequence finished successfully."

# 8. Unconditional self-shutdown to prevent idle billing leakage
echo "=================================================="
echo "Shutting down instance immediately to prevent idle cost leakage."
echo "Timestamp: $(date)"
echo "=================================================="

# Initiates OS poweroff. GCP transitions VM status to TERMINATED, halting core/GPU billing.
sudo poweroff
