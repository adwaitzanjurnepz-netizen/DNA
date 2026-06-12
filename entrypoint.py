#!/usr/bin/env python3
import subprocess
import sys
import argparse

def main():
    # Parse GCS and Limit args first
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_bucket", type=str, default="")
    parser.add_argument("--gcs_prefix", type=str, default="genomic-fine-tune")
    parser.add_argument("--limit", type=int, default=100000)
    args, unknown = parser.parse_known_args()

    # Download hg38 reference genome and index to speed up flanking sequence extraction
    import os
    print("=== Checking local reference genome hg38 ===")
    os.makedirs("./cache", exist_ok=True)
    fasta_path = "./cache/hg38.fasta"
    fai_path = "./cache/hg38.fasta.fai"
    
    if not os.path.exists(fasta_path):
        print("Downloading Homo_sapiens_assembly38.fasta from public GCS bucket...")
        subprocess.run([
            "gsutil", "cp", 
            "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta", 
            fasta_path
        ], check=True)
        
    if not os.path.exists(fai_path):
        print("Downloading Homo_sapiens_assembly38.fasta.fai from public GCS bucket...")
        subprocess.run([
            "gsutil", "cp", 
            "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta.fai", 
            fai_path
        ], check=True)

    print("=== Running Preprocessing ===")
    preprocess_cmd = [
        sys.executable, "-m", "preprocess",
        "--limit", str(args.limit),
        "--flank", "50",
        "--output_file", "./processed_variants.csv"
    ]
    subprocess.run(preprocess_cmd, check=True)

    print("=== Running Fine-tuning ===")
    train_cmd = [
        sys.executable, "-m", "train",
        "--model_name", "zhihan1996/DNABERT-2-117M",
        "--data_path", "./processed_variants.csv",
        "--use_mock", "False",
        "--epochs", "3",
        "--batch_size", "8",
        "--accum_steps", "4",
        "--bf16", "True",
        "--output_dir", "./outputs"
    ]

    if args.gcs_bucket:
        train_cmd += ["--gcs_bucket", args.gcs_bucket, "--gcs_prefix", args.gcs_prefix]

    # Forward any unknown extra args
    train_cmd += unknown

    subprocess.run(train_cmd, check=True)

if __name__ == "__main__":
    main()
