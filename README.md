# dysregulation-in-antibiotic-tolerance-persistence
This repository contains the code and data used in our paper:

**"Genome-wide Dysregulation in Antibiotic Tolerance and Persistence"**  
# 
## 📌 Overview

We present a computational framework to quantify the global correlation strength and level of dysregulation in bacterial cells under acute stress. We assess gene correlations by calculating the correlation spectrum of the experimental data and comparing it to our theoritical model - the Generalized Marchenko-Pastur (GMP) distribution. 
This repository includes:
- Code for reproducing all main figures
- Scripts for RNA-seq preprocessing and gene correlation analysis
- Derivation of the GMP distribution
- processed sequencing data and additional data that appears in the paper 
---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/alongutf/dysregulation-in-antibiotic-tolerance-persistence.git
conda env create -f environment.yml
```
Installing source packages:
```python
import src.data_functions as df
import src.analysis_functions as af
import src.bulk_functions as bf
```
## 🛠️ Usage
All figures can be generated using the corresponded figureX.py file in scripts/figures
Notebooks for different pipelines are available in scripts:
- analysis_notebook.ipynb: initial processing, including transformation of probe count data to cell-gene count matrices, cell calling, gene filtering and calculation of correlation eigenvalues.
- random_matrices.ipynb: generate random wishart matrices with non-diagonal covariance - to compare with the GMP model.
- scanpy_analysis.ipynb: perform UMAP dimensionality reduction, marker gene analysis and further analysis of the single-cell data.
- bulk_analysis.ipynb: analysis of the bulk RNA-seq data, including differential gene analysis, GO enrichment analysis and data processing.
---
