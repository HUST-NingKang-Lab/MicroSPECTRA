# MicroSPECTRA
MicroSPECTRA is a phase–frequency–amplitude spectral framework for longitudinal microbiome dynamics.
It represents longitudinal microbiome profiles as multivariate time signals and enables:
1) perturbation detection via high-frequency (HF) spectral signatures,
2) spectral editing / synthetic trajectory generation in frequency space,
3) phase-delay microbe–metabolite networks to infer lagged associations,
4) baseline spectral fingerprints to quantify inter-individual dynamical phenotypes.
<img width="1496" height="1197" alt="image" src="https://github.com/user-attachments/assets/a3e314da-b0a6-4fff-824d-1aa512ce8f85" />

## Installation

MicroSPECTRA scripts are developed under **Python 3.9+**.

### Option 1: Create a conda environment (recommended)

conda create -n microspectra python=3.10 -y
pip install -r requirements.txt

## Data

MicroSPECTRA expects subject-level abundance tables under `data/`:

```text
data/
  microbiome/
    abundance1.csv
    abundance2.csv
    ...
    abundance34.csv
  metabolome/
    metabolites29.csv
    ...
    metabolites34.csv

### Example
time,Bacteroides,Prevotella,Anaerostipes,Lactobacillus
0,0.12,0.03,0.01,0.00
1,0.10,0.04,0.02,0.00
2,0.08,0.05,0.02,0.01

## Scripts overview

| Script | Purpose | Main outputs |
|---|---|---|
| `01_compute_spectra_and_hf.py` | Sliding-window Fourier spectra + HF ratio | spectra, HF ratio time series |
| `02_fast_slow_taxa.py` | Fast/slow stratification | fast/slow/intermediate labels |
| `03_spectral_generator.py` | Spectral editing generator | synthetic trajectories + plots |
| `04_benchmark_generators.py` | Compare generators | UMAP plots, ACC metrics |
| `05_delay_network.py` | Phase-delay networks | directed edges with delays |
| `06_fingerprint_analysis.py` | Spectral fingerprints & susceptibility | fingerprint vectors, cohort stats |


# Maintainer

| Name | Email | Organization |
|-------|-------|-------|
| Yuli Zhang | yulizhang@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kouyi Zhou | zhoukouyi@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Haohong Zhang | haohongzh@gmail.com | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning  | ningkang@hust.edu.cn | Professor, School of Life Science and Technology, Huazhong University of Science & Technology|










