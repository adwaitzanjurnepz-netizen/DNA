# Genomic Disease Variant Classifier (DNABERT-2)

An end-to-end, high-performance machine learning pipeline designed to fine-tune the **DNABERT-2-117M** genomic foundation model for predicting clinical disease associations from genetic variants. 

By analyzing the flanking sequences surrounding genetic variants, this model learns to classify mutations into one of four categories:
1. **Benign / Normal** (Control variants)
2. **Cystic Fibrosis** (CFTR gene mutations)
3. **Lynch Syndrome** (MLH1, MSH2, MSH6, PMS2 gene mutations)
4. **BRCA1/2 Mutation** (Breast/ovarian cancer risk genes)

---

## 🚀 Key Features

* **High-Speed Preprocessing:** Extracts flanking sequence windows using local chromosome indexing (`pyfaidx`) against the GRCh38/hg38 reference genome, speeding up pipeline execution from hours to seconds.
* **Vertex AI Managed Jobs:** Submits training pipelines directly to Google Cloud Vertex AI Custom Training Jobs, utilizing NVIDIA A100 GPUs for fast, managed training.
* **Auto GCS Backups:** Backs up final fine-tuned model checkpoints automatically to a Google Cloud Storage (GCS) bucket.
* **In-House Validation Metrics:** Implements a custom NumPy-based evaluation suite for macro/micro classification metrics matching Scikit-learn outputs.

---

## 📁 Repository Structure

* `train.py`: PyTorch fine-tuning loop wrapper integrated with Hugging Face Trainer.
* `preprocess.py`: Extracts and clean mutations from NCBI ClinVar VCF files.
* `entrypoint.py`: Automation wrapper that orchestrates sequence lookup and training on the cloud VM.
* `dataset.py`: DNA sequence tokenization and Dataset utilities.
* `predict.py`: CLI inference utility to evaluate arbitrary DNA sequences.
* `evaluation_metrics.py`: Validation metric calculations (F1, precision, recall, confusion matrix).
* `setup.py`: Package assembly and environment declaration for GCP containers.

---

## 🛠️ Getting Started

### 1. Training on Vertex AI (100,000+ Variants)
Zip your current workspace directory, upload it to your staging bucket, and submit it to Vertex AI:
```bash
# Package the trainer code
tar -czf trainer.tar.gz entrypoint.py setup.py train.py dataset.py preprocess.py evaluation_metrics.py

# Stage the archive on GCS
gcloud storage cp trainer.tar.gz gs://<your-bucket>/trainer.tar.gz

# Submit the A100 training job
gcloud ai custom-jobs create \
    --project="<your-project>" \
    --region="us-central1" \
    --display-name="dna-fine-tune-100k-a100-job" \
    --worker-pool-spec="machine-type=a2-highgpu-1g,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest,python-module=entrypoint" \
    --python-package-uris="gs://<your-bucket>/trainer.tar.gz" \
    --args="--gcs_bucket=<your-bucket>,--limit=100000"
```

### 2. Testing the Fine-Tuned Model (Inference)
Download the model weights from GCS and run the inference helper:
```bash
# Create directory and download checkpoints
mkdir fine_tuned_model
gcloud storage cp -r gs://<your-bucket>/genomic-fine-tune/final_model ./fine_tuned_model

# Run prediction
python3 predict.py \
    --model_dir="./fine_tuned_model" \
    --sequence="ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
```

---

## 🔮 Coming Soon in the Near Future!

We are actively developing the next version of this platform. Expected features include:
* **Interactive Web UI:** A beautiful, responsive user interface to upload genomic files (FASTA/VCF), inspect validation graphs, and run real-time classification reports in the browser.
* **Expanded Gene and Disease Classes:** Support for broader pathogenicity classification using larger datasets like gnomAD.
* **Multi-GPU Distributed Training:** Out-of-the-box support for multi-GPU configurations on Vertex AI to train on millions of variants.
