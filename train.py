#!/usr/bin/env python3
"""
train.py
Fine-Tuning & Training Engine for Genomic Disease Classification.
Sets up the Hugging Face Trainer pipeline, optimizes for high-compute GCP VMs,
calculates classification metrics, and backs up checkpoints to a GCS bucket.
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    AutoConfig
)

# Import custom dataset components
from dataset import DNADiseaseDataset, generate_mock_data

def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """Computes Multi-class Accuracy, Precision, Recall, and F1-score."""
    logits, labels = eval_pred
    # In case logits is a tuple (some models return hidden states)
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    
    # Macro-averaged metrics are suitable for multi-class classification
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def backup_checkpoints_to_gcs(local_dir: str, bucket_name: str, gcs_prefix: str):
    """Recursively uploads local model checkpoints folder to a GCS bucket."""
    print(f"Backing up checkpoint folder {local_dir} to GCS bucket 'gs://{bucket_name}/{gcs_prefix}'...")
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                # Determine relative path in bucket
                rel_path = os.path.relpath(local_file_path, local_dir)
                blob_name = os.path.join(gcs_prefix, rel_path).replace("\\", "/")
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file_path)
                
        print("GCS Checkpoint backup successfully completed!")
    except ImportError:
        print("Warning: google-cloud-storage is not installed. Skipping GCS backup.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to backup checkpoints to GCS due to error: {e}", file=sys.stderr)

class CustomDNABERTClassifier(nn.Module):
    """
    Fallback custom wrapper model that extracts features from base DNABERT-2 encoder
    and feeds them to a linear classification head. Used if AutoModelForSequenceClassification
    encounters remote code registry loading errors.
    """
    def __init__(self, model_name: str, num_labels: int = 4):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # Check hidden size configuration
        config = self.encoder.config
        hidden_size = getattr(config, "hidden_size", 768)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # Sequence outputs represent shape: (batch_size, seq_len, hidden_size)
        sequence_output = outputs[0]
        
        # Mean pooling of token embeddings weighted by attention mask
        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = torch.mean(sequence_output, dim=1)
            
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )

def main():
    parser = argparse.ArgumentParser(description="Genomic Disease Classification Fine-Tuning Pipeline")
    parser.add_argument("--model_name", type=str, default="zhihan1996/DNABERT-2-117M", help="Pretrained genomic foundation model path or HF name")
    parser.add_argument("--data_path", type=str, default="./processed_variants.csv", help="Path to cleaned dataset CSV")
    parser.add_argument("--use_mock", type=str, default="False", help="Whether to generate and use mock training data ('True' or 'False')")
    parser.add_argument("--mock_size", type=int, default=200, help="Number of samples to generate in mock mode")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps to simulate larger batches")
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak fine-tuning learning rate")
    parser.add_argument("--bf16", type=str, default="False", help="Use bfloat16 mixed-precision training ('True' or 'False')")
    parser.add_argument("--fp16", type=str, default="False", help="Use fp16 mixed-precision training ('True' or 'False')")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence truncation length")
    parser.add_argument("--gcs_bucket", type=str, default="", help="Google Cloud Storage bucket name for backups")
    parser.add_argument("--gcs_prefix", type=str, default="genomic-fine-tune", help="Prefix/folder inside the GCS bucket")
    
    args = parser.parse_args()
    
    # Parse string boolean variables
    use_mock = args.use_mock.lower() == "true"
    bf16_enabled = args.bf16.lower() == "true"
    fp16_enabled = args.fp16.lower() == "true"
    
    print("--------------------------------------------------")
    print(f"Genomic Disease Classification Training Engine")
    print(f"Model: {args.model_name}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print("--------------------------------------------------")
    
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to load custom tokenizer for {args.model_name}: {e}")
        print("Falling back to loading standard bert-base-uncased tokenizer for testing purposes.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
    # 2. Prepare Data
    if use_mock:
        print(f"Generating mock genomic training dataset (size={args.mock_size})...")
        df = generate_mock_data(num_samples=args.mock_size, seq_len=args.max_length)
    else:
        print(f"Loading cleaned dataset from {args.data_path}...")
        if not os.path.exists(args.data_path):
            print(f"Error: Dataset not found at {args.data_path}. Set --use_mock True to run local tests.", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(args.data_path)
        
    # Split into train/validation sets (80% train, 20% val)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Dataset split: Train samples = {len(train_df)}, Validation samples = {len(eval_df)}")
    
    # Determine if k-mer representation is needed (original DNABERT requires spaced k-mers)
    # DNABERT-2 uses raw string BPE tokenization, so use_kmer should be False
    use_kmer = "dnabert-" in args.model_name.lower() and "-2" not in args.model_name.lower()
    if use_kmer:
         print("Note: Original DNABERT pattern detected. Activating 6-mer chunking formatting.")

    train_dataset = DNADiseaseDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_kmer=use_kmer,
        seq_column="mut_seq"
    )
    
    eval_dataset = DNADiseaseDataset(
        dataframe=eval_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_kmer=use_kmer,
        seq_column="mut_seq"
    )
    
    # 3. Load Model
    print("Loading model...")
    num_labels = 4 # Classes: Benign, Cystic Fibrosis, Lynch Syndrome, BRCA1/2
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Warning: Failed to load {args.model_name} with AutoModelForSequenceClassification: {e}")
        print("Falling back to custom feature extractor classifier head structure...")
        try:
            model = CustomDNABERTClassifier(args.model_name, num_labels=num_labels)
        except Exception as e_fallback:
            print(f"Error: Fallback custom loader failed: {e_fallback}")
            print("Falling back to a standard bert-base-uncased sequence classification model for testing purposes.")
            model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
            
    # 4. Configure Training Arguments
    # Optimized for memory and compute efficiency on A100/L4 GPUs
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        report_to="none", # Disables 3rd party cloud telemetry for clean execution
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        save_total_limit=2
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    # 6. Execute Fine-Tuning
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save final model
    print(f"Saving fine-tuned model checkpoint to {args.output_dir}/final_model...")
    final_output_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_output_path)
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(final_output_path)
        
    print("Training process successfully completed!")
    
    # 7. Backup checkpoints to GCS (if bucket name is provided)
    if args.gcs_bucket:
        backup_checkpoints_to_gcs(args.output_dir, args.gcs_bucket, args.gcs_prefix)

if __name__ == "__main__":
    main()
