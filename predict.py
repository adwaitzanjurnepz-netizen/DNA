#!/usr/bin/env python3
import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Class mapping
CLASSES = [
    "Benign/Normal",
    "Cystic Fibrosis",
    "Lynch Syndrome",
    "BRCA1/2 Mutation"
]

def main():
    parser = argparse.ArgumentParser(description="Predict disease associations from DNA sequences using fine-tuned DNABERT-2")
    parser.add_argument("--model_dir", type=str, default="./fine_tuned_model", help="Path to local directory with saved model weights")
    parser.add_argument("--sequence", type=str, required=True, help="Raw DNA sequence string to evaluate (e.g. ATCG...)")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found.")
        print("Please download your model from GCS first using:")
        print("  gcloud storage cp -r gs://my-dna-checkpoints/genomic-fine-tune/final_model ./fine_tuned_model")
        return

    print(f"Loading fine-tuned model and tokenizer from '{args.model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, trust_remote_code=True)
    model.eval()

    # Preprocess sequence (standardize casing)
    seq = args.sequence.upper().strip()
    print(f"Evaluating sequence: {seq}")

    # Tokenize
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=128)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1).squeeze(0)
        
    predicted_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_idx].item()

    print("\n" + "="*50)
    print(f"Prediction: {CLASSES[predicted_idx]} (Confidence: {confidence:.2%})")
    print("="*50)
    print("\nClass Probabilities:")
    for idx, prob in enumerate(probabilities):
        print(f"  - {CLASSES[idx]:<20} : {prob.item():.2%}")

if __name__ == "__main__":
    main()
