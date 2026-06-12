#!/usr/bin/env python3
"""
dataset.py
DNA Tokenization & Dataset Pipeline.
Provides k-mer parsing, a custom PyTorch Dataset class,
and a local mock data generator to safely run local testing.
"""

import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Union, Optional
from transformers import PreTrainedTokenizer

def sequence_to_kmers(sequence: str, k: int = 6) -> str:
    """
    Converts a raw DNA sequence string into space-separated k-mers.
    Example: "ATCGATCG" -> "ATCGAT TCGATC CGATCG"
    """
    sequence = sequence.upper().strip()
    if len(sequence) < k:
        # Fallback if sequence is too short
        return sequence
    return " ".join([sequence[i : i + k] for i in range(len(sequence) - k + 1)])

class DNADiseaseDataset(Dataset):
    """Custom PyTorch Dataset for Genomic Disease Variant Classification."""
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 128,
        use_kmer: bool = False,
        k: int = 6,
        seq_column: str = "mut_seq",
        label_column: str = "label"
    ):
        """
        Args:
            csv_path: Path to the processed variants CSV.
            dataframe: Optional pre-loaded pandas DataFrame.
            tokenizer: Hugging Face PreTrainedTokenizer instance.
            max_length: Padding and truncation length.
            use_kmer: Whether to convert raw DNA sequences to space-separated k-mers.
            k: K-mer size if use_kmer is True.
            seq_column: Column name containing the DNA sequence string.
            label_column: Column name containing the integer class label.
        """
        if dataframe is not None:
            self.df = dataframe
        elif csv_path is not None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Dataset CSV file not found at: {csv_path}")
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Must provide either 'csv_path' or 'dataframe' parameter.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_kmer = use_kmer
        self.k = k
        self.seq_column = seq_column
        self.label_column = label_column
        
        # Ensure data is clean (remove rows with missing sequences)
        self.df = self.df.dropna(subset=[self.seq_column]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        sequence = str(row[self.seq_column])
        label = int(row[self.label_column])

        # If model expects k-mers (e.g. original DNABERT), convert sequence
        if self.use_kmer:
            sequence = sequence_to_kmers(sequence, k=self.k)

        # Tokenize
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                sequence,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            # Squeeze out batch dimension (Hugging Face tokenizer returns batch of 1 with return_tensors='pt')
            item = {key: val.squeeze(0) for key, val in encoding.items()}
        else:
            # Simple fallback if no tokenizer is provided (returns raw characters as index representations)
            # Useful for basic non-transformer testing
            vocab = {"PAD": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
            encoded = [vocab.get(char, 5) for char in sequence[:self.max_length]]
            # Pad manually
            encoded += [0] * (self.max_length - len(encoded))
            item = {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor([1 if x > 0 else 0 for x in encoded], dtype=torch.long)
            }

        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

def generate_mock_data(num_samples: int = 100, seq_len: int = 101, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Generates synthetic DNA sequences with realistic labels and mutation coordinates
    to support testing the training pipeline end-to-end.
    
    Classes:
      - 0: Benign/Normal (Random baseline variations)
      - 1: Cystic Fibrosis (Simulates CFTR variants)
      - 2: Lynch Syndrome (Simulates MMR gene variants)
      - 3: BRCA1/2 Mutation (Simulates breast cancer susceptibility variants)
    """
    bases = ["A", "C", "G", "T"]
    data = []
    
    # Specific mutations or motifs to inject for disease classes to allow model learning
    disease_motifs = {
        1: "ATGCFTRAAA",   # Cystic Fibrosis signature motif
        2: "CGMLH2MSH6G", # Lynch Syndrome signature motif
        3: "TBRCA1BRCA2"  # BRCA1/2 signature motif
    }
    
    genes = {
        0: "ACTB",
        1: "CFTR",
        2: "MLH1",
        3: "BRCA1"
    }
    
    diseases = {
        0: "Benign/Normal",
        1: "Cystic Fibrosis",
        2: "Lynch Syndrome",
        3: "BRCA1/2 Mutation"
    }

    for i in range(num_samples):
        # Choose class
        label = random.randint(0, 3)
        disease = diseases[label]
        gene = genes[label]
        
        # Generate base wild-type sequence
        wt_list = [random.choice(bases) for _ in range(seq_len)]
        wt_seq = "".join(wt_list)
        
        # Injects mutation allele
        ref = random.choice(bases)
        alt = random.choice([b for b in bases if b != ref])
        
        # Insert signature motifs inside the mutated sequence
        mut_list = list(wt_seq)
        if label > 0:
            motif = disease_motifs[label]
            # Replace a random slice in the middle with the motif
            start_idx = seq_len // 2 - len(motif) // 2
            mut_list[start_idx : start_idx + len(motif)] = list(motif)
        else:
            # Benign variant: just single point mutation without disease motif
            pos_idx = seq_len // 2
            mut_list[pos_idx] = alt
            
        mut_seq = "".join(mut_list)
        
        data.append({
            "chrom": f"chr{random.randint(1, 22)}",
            "pos": random.randint(100000, 99999999),
            "ref": ref,
            "alt": alt,
            "gene": gene,
            "clnsig": "Pathogenic" if label > 0 else "Benign",
            "disease": disease,
            "label": label,
            "wt_seq": wt_seq,
            "mut_seq": mut_seq
        })

    df = pd.DataFrame(data)
    
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Mock data successfully written to {output_path}")
        
    return df

if __name__ == "__main__":
    # Test generation and tokenization
    df = generate_mock_data(10, output_path="./mock_variants.csv")
    print(df.head(2))
    
    # Try custom dataset class
    dataset = DNADiseaseDataset(dataframe=df, max_length=20)
    print("Dataset length:", len(dataset))
    item = dataset[0]
    print("Example Item keys:", item.keys())
    print("Input IDs representation:", item["input_ids"])
    print("Label representation:", item["labels"])
