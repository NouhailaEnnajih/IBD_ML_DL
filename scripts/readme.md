# Microbiome Data Processing Pipeline

This repository contains a set of scripts for processing 16S rRNA microbiome data, from downloading raw sequences to preparing taxa tables for downstream analysis.  

## Scripts Overview

### Script 1: Data Download
- **Purpose:** Download raw sequencing data.  
- **Input:** A text file containing accession numbers (one per line). For this project, we used `1359accessions.txt`.  
- **Notes:** Metadata associated with the downloaded samples will be used later to define labels (e.g., patient vs. control) in the final analysis script.  
- **Sources:** Data can be downloaded from **EBI** or **NCBI**.

### Script 2: DADA2 Pipeline
- **Purpose:** Perform quality control, filtering, and generate taxa tables from the raw sequence data.  
- **Notes:** This script handles the core processing from raw FASTQ files to an abundance table with identified taxa.

### Script 3: Taxa Renaming
- **Purpose:** Standardize and rename taxa in the tables produced by Script 2 for consistent downstream analysis.

## Usage Notes
- Ensure that all input files (e.g., `1359accessions.txt`) are correctly formatted before running Script 1.  
- Scripts should be run sequentially: **Script 1 → Script 2 → Script 3**.  
