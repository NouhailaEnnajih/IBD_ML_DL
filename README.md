# Microbiome ML/DL Benchmarking Pipeline

This repository provides a pipeline for benchmarking machine learning (ML) and deep learning (DL) models on 16S rRNA microbiome data. The goal is to compare different computational approaches in terms of predictive performance and robustness for microbiome-based classification tasks.

### Workflow
![Workflow of the study](workflowIBD1356.png)

## Purpose

- Evaluate and compare ML and DL models on 16S rRNA data.  
- Handle common preprocessing steps such as data splitting, balancing with SMOTE, and normalization.  
- Provide cross-validation metrics and statistical tests to assess model performance.  
- Identify the most relevant taxa contributing to classification results.

## Contents

- **1_download_data.R**: Fetch raw sequence data from public databases using accession numbers.  
- **2_dada2.R**: Process raw sequences into taxa tables.  
- **3_renaming_taxa.R**: Standardize taxa names for downstream analysis.  
- **ML and DL models.ipynb** and **ML and DL models.py**: Train and evaluate models using the processed taxa tables.

## Notes

- The ML/DL analysis is provided in both Python script (`.py`) and Jupyter notebook (`.ipynb`) formats; both perform the same steps.  
- Designed for reproducibility and benchmarking, allowing comparison across multiple modeling approaches.



