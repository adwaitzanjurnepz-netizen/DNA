#!/usr/bin/env python3
"""
preprocess.py
Data Sourcing, Streaming, and Cleaning Strategy for Genomic Disease Classification.
Streams NCBI ClinVar VCF files, filters variants associated with specific diseases,
and fetches flanking sequences using the UCSC Genome Browser REST API.
"""

import os
import sys
import gzip
import csv
import urllib.request
import json
import argparse
import time
from typing import Dict, Any, List, Optional
import requests
from tqdm import tqdm

# Constants for ClinVar URLs (GRCh38 assembly version is recommended)
CLINVAR_GRCH38_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
UCSC_API_URL = "https://api.genome.ucsc.edu/getData/sequence"

# Genes associated with target genetic diseases
DISEASE_GENE_MAPPING = {
    "Cystic Fibrosis": ["CFTR"],
    "Lynch Syndrome": ["MLH1", "MSH2", "MSH6", "PMS2"],
    "BRCA1/2 Mutation": ["BRCA1", "BRCA2"]
}

# Label encoder mapping
LABEL_MAP = {
    "Benign/Normal": 0,
    "Cystic Fibrosis": 1,
    "Lynch Syndrome": 2,
    "BRCA1/2 Mutation": 3
}

def parse_info_field(info_str: str) -> Dict[str, str]:
    """Parses the INFO field of a VCF line into a key-value dictionary."""
    info_dict = {}
    for item in info_str.split(";"):
        if "=" in item:
            parts = item.split("=", 1)
            info_dict[parts[0]] = parts[1]
        else:
            info_dict[item] = "true"
    return info_dict

def download_file(url: str, dest_path: str):
    """Downloads a file with a visual progress bar using requests streaming."""
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, "wb") as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

def fetch_flanking_sequence(chrom: str, pos: int, ref: str, flank: int = 50, genome: str = "hg38") -> Optional[str]:
    """
    Fetches the reference flanking sequence from UCSC Genome Browser API.
    Handles coordinate conversion: VCF is 1-based, UCSC REST API is 0-based, half-open.
    """
    # Normalize chromosome name for UCSC (must start with 'chr')
    chrom_clean = chrom.lower()
    if not chrom_clean.startswith("chr"):
        if chrom_clean in ["mt", "m"]:
            chrom_clean = "chrM"
        else:
            chrom_clean = f"chr{chrom}"
    else:
        # standard capitalization
        if "chr" in chrom_clean:
            chrom_clean = "chr" + chrom_clean.split("chr")[1].upper()
            if chrom_clean == "CHRM":
                chrom_clean = "chrM"

    # Start and End coordinates (0-based, half-open)
    ref_len = len(ref)
    ref_start_0based = pos - 1
    
    fetch_start = max(0, ref_start_0based - flank)
    fetch_end = ref_start_0based + ref_len + flank
    
    params = {
        "genome": genome,
        "chrom": chrom_clean,
        "start": fetch_start,
        "end": fetch_end
    }
    
    try:
        # Respectful API access - brief pause
        time.sleep(0.1)
        response = requests.get(UCSC_API_URL, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            retrieved_seq = data.get("dna")
            if not retrieved_seq:
                return None
            
            # Defensive check: ensure the reference allele in the middle matches the VCF's REF
            # Calculated relative position of REF within the retrieved window
            ref_offset = ref_start_0based - fetch_start
            extracted_ref = retrieved_seq[ref_offset : ref_offset + ref_len].upper()
            
            if extracted_ref != ref.upper():
                print(f"Warning: Reference mismatch at {chrom}:{pos}. VCF REF='{ref}', UCSC='{extracted_ref}'", file=sys.stderr)
                # We can still return the sequence but log a warning
                
            return retrieved_seq.upper()
        else:
            print(f"Error fetching from UCSC API: Status code {response.status_code}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Exception during UCSC sequence fetch: {e}", file=sys.stderr)
        return None

def process_clinvar(vcf_path: str, limit: int, flank: int, output_file: str):
    """Streams the ClinVar VCF file, filters pathogenic and benign variants, and extracts sequences."""
    print(f"Streaming and parsing variants from {vcf_path}...")
    
    processed_count = 0
    records = []
    
    # Track distributions
    class_counts = {k: 0 for k in LABEL_MAP.keys()}
    
    with gzip.open(vcf_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            
            # Stop if we have hit the target sample limit
            if processed_count >= limit:
                break
                
            columns = line.strip().split("\t")
            if len(columns) < 8:
                continue
                
            chrom, pos_str, _, ref, alt, _, _, info_str = columns[:8]
            pos = int(pos_str)
            
            # Parse INFO fields for disease and significance labels
            info = parse_info_field(info_str)
            clnsig = info.get("CLNSIG", "").lower()
            clndn = info.get("CLNDN", "").lower()
            geneinfo = info.get("GENEINFO", "")
            
            # Extract gene symbol (e.g., "CFTR:1080" -> "CFTR")
            gene_symbol = geneinfo.split(":")[0] if geneinfo else ""
            
            # Determine pathogenicity
            is_pathogenic = any(term in clnsig for term in ["pathogenic", "likely_pathogenic"])
            is_benign = any(term in clnsig for term in ["benign", "likely_benign"])
            
            if not (is_pathogenic or is_benign):
                continue
                
            disease_class = None
            
            if is_benign:
                # To maintain class balance, we tag benign variants inside our target genes
                # or general benign variants as Normal controls.
                # Let's verify if the variant falls into one of our target genes
                for disease, genes in DISEASE_GENE_MAPPING.items():
                    if gene_symbol in genes:
                        disease_class = "Benign/Normal"
                        break
            elif is_pathogenic:
                # Pathogenic variant: check which target disease it maps to
                for disease, genes in DISEASE_GENE_MAPPING.items():
                    if gene_symbol in genes or any(g.lower() in clndn for g in genes):
                        disease_class = disease
                        break
            
            if disease_class is None:
                continue
                
            # Fetch wild-type (reference) flanking sequence
            wt_sequence = fetch_flanking_sequence(chrom, pos, ref, flank=flank)
            if not wt_sequence:
                continue
                
            # Construct mutated sequence
            # The REF allele in wt_sequence starts at index `flank` (or less if close to chrom start)
            # Find the actual offset
            ref_start_0based = pos - 1
            fetch_start = max(0, ref_start_0based - flank)
            ref_offset = ref_start_0based - fetch_start
            
            # Assemble mutated sequence
            mut_sequence = wt_sequence[:ref_offset] + alt + wt_sequence[ref_offset + len(ref):]
            
            record = {
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "gene": gene_symbol,
                "clnsig": info.get("CLNSIG", ""),
                "disease": disease_class,
                "label": LABEL_MAP[disease_class],
                "wt_seq": wt_sequence,
                "mut_seq": mut_sequence
            }
            
            records.append(record)
            class_counts[disease_class] += 1
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{limit} records. Distribution: {dict(class_counts)}")
                
    # Save to CSV
    if records:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["chrom", "pos", "ref", "alt", "gene", "clnsig", "disease", "label", "wt_seq", "mut_seq"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        print(f"Successfully processed {len(records)} records and saved to {output_file}")
        print(f"Final Class Distribution: {dict(class_counts)}")
    else:
        print("No matching variants found in VCF stream. Try increasing the scan depth or running mock generation.")

def main():
    parser = argparse.ArgumentParser(description="Genomic Disease Dataset Pre-processing and Streaming Tool")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of variants to process")
    parser.add_argument("--flank", type=int, default=50, help="Flanking bp upstream and downstream (total sequence len = 2*flank + len(REF))")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Local directory to store download cache")
    parser.add_argument("--output_file", type=str, default="./processed_variants.csv", help="Output CSV dataset path")
    args = parser.parse_args()
    
    vcf_path = os.path.join(args.cache_dir, "clinvar.vcf.gz")
    
    # Download ClinVar VCF if not already cached
    if not os.path.exists(vcf_path):
        download_file(CLINVAR_GRCH38_URL, vcf_path)
    else:
        print(f"Using cached ClinVar dataset found at: {vcf_path}")
        
    process_clinvar(vcf_path, args.limit, args.flank, args.output_file)

if __name__ == "__main__":
    main()
