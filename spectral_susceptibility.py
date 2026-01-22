from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


EPS = 1e-12


@dataclass(frozen=True)
class SusceptibilityConfig:
    input_filename: str = "genus.csv"
    time_col: str = "time"
    baseline_day: float = 14.0
    top_g: int = 20
    min_mean_abund: float = 1e-4
    low_high_frac: float = 1 / 3


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_plot_style(font_family: str = "DejaVu Sans", font_size: int = 10) -> None:
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [font_family, "Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def read_subject_table(subject_dir: Path, cfg: SusceptibilityConfig) -> pd.DataFrame:
    fp = subject_dir / cfg.input_filename
    df = pd.read_csv(fp)
    if cfg.time_col not in df.columns:
        raise ValueError(f"missing time column: {cfg.time_col} in {fp}")
    df = df.copy()
    df[cfg.time_col] = pd.to_numeric(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).reset_index(drop=True)
    return df


def spectral_fingerprint_features(
    df: pd.DataFrame,
    mask: np.ndarray,
    core_genera: List[str],
    *,
    time_col: str,
    min_mean_abund: float,
    low_high_frac: float,
) -> Optional[Dict[str, object]]:
    sub = df.loc[mask].copy()
    if sub.shape[0] < 3:
        return None

    n = sub.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0)

    genus_power_norm: Dict[str, np.ndarray] = {}
    global_power = np.zeros_like(freqs, dtype=float)

    for g in core_genera:
        if g not in sub.columns:
            continue
        x = pd.to_numeric(sub[g], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).any() or float(np.nanmean(x)) < float(min_mean_abund):
            continue

        x = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)
        x = x - float(np.mean(x))

        w = np.hanning(n)
        fft_vals = np.fft.rfft(x * w)
        power = (np.abs(fft_vals) ** 2).astype(float)

        power[0] = 0.0
        p = power[1:]
        s = float(p.sum())
        if s <= 0:
            continue

        genus_power_norm[g] = p / s
        global_power[1:] += p

    if not genus_power_norm or float(global_power[1:].sum()) <= 0:
        return None

    global_freq_norm = global_power[1:] / float(global_power[1:].sum())

    k = int(max(1, round(len(global_freq_norm) * float(low_high_frac))))
    k = min(k, len(global_freq_norm))

    low_ratio = float(global_freq_norm[:k].sum())
    high_ratio = float(global_freq_norm[-k:].sum())

    prob = global_freq_norm
    entropy = float(-(prob * np.log(prob + 1e-12)).sum() / np.log(len(prob)))

    features = {
        "low_ratio": low_ratio,
        "high_ratio": high_ratio,
        "entropy": entropy,
        "n_genera": int(len(genus_power_norm)),
    }

    return {
        "freqs": freqs[1:],
        "genus_power_norm": genus_power_norm,
        "global_freq_norm": global_freq_norm,
        "features": features,
    }


def change_metrics(base_res: Dict[str, object], post_res: Dict[str, object]) -> Dict[str, float]:
    fb = base_res["features"]
    fp = post_res["features"]

    delta_entropy = float(fp["entropy"] - fb["entropy"])
    delta_low = float(fp["low_ratio"] - fb["low_ratio"])
    delta_high = float(fp["high_ratio"] - fb["high_ratio"])

    change_vec = np.array([delta_entropy, delta_low, delta_high], dtype=float)
    change_mag = float(np.linalg.norm(change_vec))

    return {
        "delta_entropy": delta_entropy,
        "delta_low": delta_low,
        "delta_high": delta_high,
        "change_mag": change_mag,
    }


def plot_change_magnitude(df: pd.DataFrame, out_base: str, *, font_family: str, font_size: int) -> None:
    if df.empty:
        return
    set_plot_style(font_family=font_family, font_size=font_size)

    d = df.sort_values("change_mag", ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(4.2, 3.0))
    ax = plt.gca()
    ax.bar(np.arange(len(d)), d["change_mag"].to_numpy(dtype=float))
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(d["subject"].tolist(), rotation=90)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Spectral change magnitude")
    clean_axes(ax)
    fig.tight_layout()

    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def run(root_dir: str, out_dir: str, cfg: SusceptibilityConfig, *, font_family: str, font_size: int) -> pd.DataFrame:
    root = Path(root_dir)
    out = ensure_dir(out_dir)

    subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
    rows: List[Dict[str, object]] = []

    for sid in subjects:
        df = read_subject_table(root / sid, cfg)
        if df.empty:
            continue

        genera_cols = [c for c in df.columns if c != cfg.time_col]
        if not genera_cols:
            continue

        core = (
            df[genera_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean()
            .sort_values(ascending=False)
            .head(int(cfg.top_g))
            .index.tolist()
        )

        t = pd.to_numeric(df[cfg.time_col], errors="coerce").to_numpy(dtype=float)
        base_mask = np.isfinite(t) & (t < float(cfg.baseline_day))
        post_mask = np.isfinite(t) & (t >= float(cfg.baseline_day))

        base_res = spectral_fingerprint_features(
            df, base_mask, core,
            time_col=cfg.time_col,
            min_mean_abund=cfg.min_mean_abund,
            low_high_frac=cfg.low_high_frac,
        )
        post_res = spectral_fingerprint_features(
            df, post_mask, core,
            time_col=cfg.time_col,
            min_mean_abund=cfg.min_mean_abund,
            low_high_frac=cfg.low_high_frac,
        )

        if base_res is None or post_res is None:
            continue

        ch = change_metrics(base_res, post_res)
        rows.append({"subject": sid, **ch})

    out_df = pd.DataFrame(rows)
    out_csv = out / "spectral_change_magnitude_table.csv"
    out_df.sort_values("change_mag", ascending=False).to_csv(out_csv, index=False)

    plot_change_magnitude(out_df, str(out / "spectral_change_magnitude"), font_family=font_family, font_size=font_size)
    return out_df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spectral_susceptibility.py")
    p.add_argument("--root-dir", required=True)
    p.add_argument("--out-dir", default="results/susceptibility")
    p.add_argument("--input-filename", default="genus.csv")
    p.add_argument("--time-col", default="time")
    p.add_argument("--baseline-day", type=float, default=14.0)
    p.add_argument("--top-g", type=int, default=20)
    p.add_argument("--min-mean-abund", type=float, default=1e-4)
    p.add_argument("--low-high-frac", type=float, default=1 / 3)

    p.add_argument("--font-family", default="DejaVu Sans")
    p.add_argument("--font-size", type=int, default=10)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = SusceptibilityConfig(
        input_filename=str(args.input_filename),
        time_col=str(args.time_col),
        baseline_day=float(args.baseline_day),
        top_g=int(args.top_g),
        min_mean_abund=float(args.min_mean_abund),
        low_high_frac=float(args.low_high_frac),
    )
    df = run(
        root_dir=str(args.root_dir),
        out_dir=str(args.out_dir),
        cfg=cfg,
        font_family=str(args.font_family),
        font_size=int(args.font_size),
    )
    print("saved:", str(Path(args.out_dir) / "spectral_change_magnitude_table.csv"))
    print("saved:", str(Path(args.out_dir) / "spectral_change_magnitude.png"))
    print("saved:", str(Path(args.out_dir) / "spectral_change_magnitude.pdf"))
    print("n_subjects:", len(df))


if __name__ == "__main__":
    main()
