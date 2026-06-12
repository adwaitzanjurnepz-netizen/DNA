#!/usr/bin/env python3
import subprocess
import sys
import argparse

def main():
    print("=== Running Preprocessing ===")
    preprocess_cmd = [
        sys.executable, "-m", "preprocess",
        "--limit", "1000",
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

    # Parse GCS args if passed through
    parser = argparse.ArgumentParser()
    parser.add_argument("--gcs_bucket", type=str, default="")
    parser.add_argument("--gcs_prefix", type=str, default="genomic-fine-tune")
    args, unknown = parser.parse_known_args()

    if args.gcs_bucket:
        train_cmd += ["--gcs_bucket", args.gcs_bucket, "--gcs_prefix", args.gcs_prefix]

    # Forward any unknown extra args
    train_cmd += unknown

    subprocess.run(train_cmd, check=True)

if __name__ == "__main__":
    main()
