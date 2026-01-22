# MicroSPECTRA
MicroSPECTRA is a phase–frequency–amplitude spectral framework for longitudinal microbiome dynamics.
It represents longitudinal microbiome profiles as multivariate time signals and enables:
1) perturbation detection via high-frequency (HF) spectral signatures,
2) spectral editing / synthetic trajectory generation in frequency space,
3) phase-delay microbe–metabolite networks to infer lagged associations,
4) baseline spectral fingerprints to quantify inter-individual dynamical phenotypes.
<img width="1496" height="1197" alt="image" src="https://github.com/user-attachments/assets/a3e314da-b0a6-4fff-824d-1aa512ce8f85" />

---

## Installation

MicroSPECTRA scripts are developed under **Python 3.9+** (recommended: Python 3.10).

### Option 1: Create a conda environment (recommended)

```bash
conda create -n microspectra python=3.10 -y
conda activate microspectra
pip install -r requirements.txt
```

### Option 2: Install with pip

```bash
pip install -r requirements.txt
```

> If you do not have `requirements.txt` yet, typical dependencies include:
> `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `umap-learn`, `networkx`.

---

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
```

### Input format (microbiome; example)

- First column: `time` (or `day`), in days (numeric)
- Remaining columns: taxa abundances (numeric)

```csv
time,Bacteroides,Prevotella,Anaerostipes,Lactobacillus
0,0.12,0.03,0.01,0.00
1,0.10,0.04,0.02,0.00
2,0.08,0.05,0.02,0.01
```

### Input format (metabolome; example)

```csv
time,taurochenodeoxycholic_acid,17_hydroxyprogesterone
0,12.3,0.45
1,10.8,0.40
2,11.5,0.43
```

---

## Quickstart

> Replace file paths with your actual filenames.

### 1) Sliding-window Fourier spectra + HF ratio

```bash
python hf_ratio_anomaly.py single \
  --input data/abundance1.csv \
  --time-col time \
  --window "70-122" \
  --out-dir results/anomaly
```

### 2) Fast/slow taxa stratification

```bash
python 02_fast_slow_taxa.py   --subject S01   --microbiome_file data/microbiome/abundance1.csv   --out_dir results/S01/
```

### 3) Spectral generator

```bash
python 03_spectral_generator.py   --subject S01   --microbiome_file data/microbiome/abundance1.csv   --out_dir results/S01/generator/   --n_samples 500   --seed 0
```

### 4) Compare generators (benchmark)

```bash
python 04_benchmark_generators.py   --microbiome_dir data/microbiome/   --subjects S01,S02,S03   --out_dir results/generator_benchmark/   --seed 0
```

### 5) Phase-delay microbe–metabolite network (S29–S34)

```bash
python 05_delay_network.py   --subject S29   --microbiome_file data/microbiome/abundance29.csv   --metabolome_file data/metabolome/metabolites29.csv   --out_dir results/S29/delay_network/
```

### 6) Spectral fingerprints & susceptibility

```bash
python 06_fingerprint_analysis.py   --microbiome_dir data/microbiome/   --subjects S29,S30,S31,S32,S33,S34   --out_dir results/fingerprints/
```

---

## Scripts overview

| Script | Purpose | Main outputs |
|---|---|---|
| `01_compute_spectra_and_hf.py` | Sliding-window Fourier spectra + HF ratio | spectra, HF ratio series |
| `02_fast_slow_taxa.py` | Fast/slow stratification | fast/slow/intermediate labels |
| `03_spectral_generator.py` | Spectral editing generator | synthetic trajectories + plots |
| `04_benchmark_generators.py` | Compare generators | UMAP plots, ACC metrics |
| `05_delay_network.py` | Phase-delay networks | directed edges with delays |
| `06_fingerprint_analysis.py` | Spectral fingerprints & susceptibility | fingerprint vectors, cohort stats |

---

## Outputs

Outputs are written under `results/` by default (or under `--out_dir`).

Typical outputs include:
- `spectra/`: power spectra, dominant timescales, band power fractions  
- `hf_ratio.csv`: HF ratio time series (local anomaly score)  
- `generator/`: synthetic trajectories + embedding plots + metrics  
- `delay_network/`: edge list with predicted delays (days)  
- `fingerprints/`: per-subject spectral fingerprints + cohort summaries  

---


## Maintainers

| Name | Email | Organization |
|---|---|---|
| Yuli Zhang | yulizhang@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kouyi Zhou | zhoukouyi@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Haohong Zhang | haohongzh@gmail.com | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning | ningkang@hust.edu.cn | Professor, School of Life Science and Technology, Huazhong University of Science & Technology |











