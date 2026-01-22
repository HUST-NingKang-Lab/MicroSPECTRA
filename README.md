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

### 2) Fourier generator

```bash
python fourier_ts_generate.py \
  --input data/abundance1.csv \
  --output results/fourier_synth.csv \
  --eval \
  --metrics-out results/fourier_metrics.json \
  --umap-out results/fourier_umap_time_match \
  --umap-n-neighbors 15 \
  --umap-min-dist 0.1

Key generator knobs:
- `--amp-jitter`: amplitude perturbation strength
- `--p-modify`: fraction of frequency bins to modify
- `--keep-k`: only modify bins within [1, keep_k)

python ts_generation_benchmark.py \
  --input data/abundance1.csv \
  --output-dir results/benchmark

Methods included:
- `clr_jitter_repeat_by_time`
- `dirichlet_global`
- `pca_gaussian_softmax`
- `knn_mixup`
```

### 3)  Phase-delay microbe–metabolite network (S29–S34)

```bash
python phase_delay_network.py \
  --microbe-csv data/genus.csv \
  --metabolite-csv data/metabolite.csv \
  --out-dir results/lag_network \
  --time-col time \
  --band-periods "7:0.5" \
  --nperseg 8 \
  --overlap 0.5 \
  --n-perm 200 \
  --phase-r-min 0.30 \
  --xcorr-min-peak 0.15 \
  --q-threshold 0.10
```

### 4) Spectral fingerprints & susceptibility

```bash
# 1) Susceptibility: spectral change magnitude
python spectral_susceptibility.py \
  --root-dir data \
  --out-dir results/susceptibility \
  --time-col time \
  --baseline-day 14 \
  --top-g 20

# 2) Spectral fingerprints: distance + shannon + pre/post stats
python spectral_fingerprints.py \
  --root-dir data \
  --out-dir results/fingerprints \
  --time-col time \
  --baseline-day 14 \
  --top-g 15 \
  --window-len 5

```

---

## Scripts overview

| Script | Purpose | Main outputs |
|---|---|---|
| `hf_ratio_anomaly.py` | Sliding-window Fourier HF ratio anomaly scoring (single table or meta list) :contentReference[oaicite:0]{index=0} | `hf_anomaly_single_summary.csv` / `hf_anomaly_all_tables_summary.csv`, per-table `per_table/*_hf_anomaly_metrics.csv` |
| `fourier_ts_generate.py` | Fourier-based time-series generator (optional evaluation: metrics + UMAP time-match) :contentReference[oaicite:1]{index=1} | synthetic CSV (`--output`), `*_metrics.json`, `*_umap_time_match.png/.pdf` |
| `ts_generation_benchmark.py` | Generator benchmark (kept: `clr_jitter_repeat_by_time`, `dirichlet_global`, `pca_gaussian_softmax`, `knn_mixup`) with metrics + UMAP time-match :contentReference[oaicite:2]{index=2} | per-method: `*_synth.csv`, `*_metrics.json`, `*_umap_time_match.png/.pdf`; summary: `summary_metrics.csv`, `summary_metrics.json` |
| `phase_delay_network.py` | Phase-delay network inference (STFT/CSD + permutation tests + BH-FDR) :contentReference[oaicite:3]{index=3} | `all_results.csv`, `best_per_pair.csv`, `edges_q<=*.csv` |
| `spectral_susceptibility.py` | Spectral susceptibility via pre/post spectral feature shift magnitude :contentReference[oaicite:4]{index=4} | `spectral_change_magnitude_table.csv`, `spectral_change_magnitude.png/.pdf` |
| `spectral_fingerprints.py` | Spectral fingerprints distance (to pre reference) + Shannon index; pre/post stats with BH-FDR :contentReference[oaicite:5]{index=5} | per-subject `*_spectral_distance.png/.pdf`, `*_shannon_index.png/.pdf`; cohort `pre_post_significance_stats.csv` |

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











