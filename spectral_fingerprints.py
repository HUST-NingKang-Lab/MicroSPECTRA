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
from scipy.stats import mannwhitneyu


EPS = 1e-12


@dataclass(frozen=True)
class FingerprintConfig:
    input_filename: str = "genus.csv"
    time_col: str = "time"
    baseline_day: float = 14.0
    top_g: int = 15
    window_len: int = 5
    min_abund: float = 1e-4
    out_dir: str = "results/fingerprints"


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


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    ok = np.isfinite(p)
    pv = p[ok]
    if pv.size == 0:
        return q

    order = np.argsort(pv)
    pv_sorted = pv[order]
    m = pv_sorted.size
    ranks = np.arange(1, m + 1, dtype=float)

    q_sorted = pv_sorted * m / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q_ok = np.empty_like(q_sorted)
    q_ok[order] = q_sorted
    q[ok] = q_ok
    return q


def format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and (np.isnan(p))):
        return "n.s."
    if p < 1e-4:
        return "p < 1e-4"
    return f"p = {p:.3g}"


def shannon_index(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, 0.0)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())
    if s <= 0:
        return float("nan")
    p = x / s
    return float(-(p * np.log(p + eps)).sum())


def read_subject_table(subject_dir: Path, cfg: FingerprintConfig) -> pd.DataFrame:
    fp = subject_dir / cfg.input_filename
    df = pd.read_csv(fp)
    if cfg.time_col not in df.columns:
        raise ValueError(f"missing time column: {cfg.time_col} in {fp}")
    df = df.copy()
    df[cfg.time_col] = pd.to_numeric(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col]).sort_values(cfg.time_col).reset_index(drop=True)
    return df


def window_fingerprint(window_df: pd.DataFrame, core_genera: List[str], *, min_abund: float) -> Tuple[np.ndarray, np.ndarray]:
    n = window_df.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0)[1:]
    m = len(freqs)
    mat = np.zeros((len(core_genera), m), dtype=float)

    for i, g in enumerate(core_genera):
        if g not in window_df.columns:
            continue
        x = pd.to_numeric(window_df[g], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(x).any() or float(np.nanmean(x)) < float(min_abund):
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
        mat[i, :] = p / s

    return mat, freqs


def mannwhitney_p(pre_vals: np.ndarray, post_vals: np.ndarray) -> float:
    pre = np.asarray(pre_vals, dtype=float)
    post = np.asarray(post_vals, dtype=float)
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    if pre.size < 2 or post.size < 2:
        return float("nan")
    try:
        res = mannwhitneyu(pre, post, alternative="two-sided", method="auto")
        return float(res.pvalue)
    except Exception:
        return float("nan")


def subject_time_series(df: pd.DataFrame, cfg: FingerprintConfig) -> Optional[pd.DataFrame]:
    genera_cols = [c for c in df.columns if c != cfg.time_col]
    if not genera_cols:
        return None

    core_genera = (
        df[genera_cols]
        .apply(pd.to_numeric, errors="coerce")
        .mean()
        .sort_values(ascending=False)
        .head(int(cfg.top_g))
        .index.tolist()
    )

    matrices, times, phases, shannons = [], [], [], []
    w = int(cfg.window_len)
    if df.shape[0] < w:
        return None

    for k in range(w - 1, df.shape[0]):
        wdf = df.iloc[k - w + 1 : k + 1]
        mat, _ = window_fingerprint(wdf, core_genera, min_abund=cfg.min_abund)
        matrices.append(mat)

        t_center = float(pd.to_numeric(wdf[cfg.time_col].iloc[-1], errors="coerce"))
        phase = "pre" if t_center < float(cfg.baseline_day) else "post"
        times.append(t_center)
        phases.append(phase)

        last_abund = wdf[genera_cols].iloc[-1].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        shannons.append(shannon_index(last_abund))

    matrices = np.asarray(matrices, dtype=float)
    times = np.asarray(times, dtype=float)
    phases = np.asarray(phases, dtype=object)
    shannons = np.asarray(shannons, dtype=float)

    pre_idx = np.where(phases == "pre")[0]
    if pre_idx.size == 0:
        return None

    F_ref = matrices[pre_idx].mean(axis=0)
    diffs = matrices - F_ref
    dists = np.sqrt((diffs**2).mean(axis=(1, 2)))

    out_df = pd.DataFrame(
        {"time": times, "phase": phases, "distance": dists, "shannon": shannons}
    ).sort_values("time").reset_index(drop=True)
    return out_df


def plot_box_and_time(subject: str, df: pd.DataFrame, *, ycol: str, ylabel: str, out_base: str,
                      baseline_day: float, font_family: str, font_size: int) -> float:
    set_plot_style(font_family=font_family, font_size=font_size)

    pre_vals = df.loc[df["phase"] == "pre", ycol].to_numpy(dtype=float)
    post_vals = df.loc[df["phase"] == "post", ycol].to_numpy(dtype=float)
    p_val = mannwhitney_p(pre_vals, post_vals)

    all_vals = df[ycol].to_numpy(dtype=float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return float("nan")

    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if vmax - vmin < EPS:
        pad = max(0.05 * abs(vmin), 0.01)
        vmin, vmax = vmin - pad, vmax + pad
    else:
        pad = 0.08 * (vmax - vmin)
        vmin, vmax = vmin - pad, vmax + pad

    fig = plt.figure(figsize=(6.0, 4.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 3.0], hspace=0.10)

    ax_box = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[1, 0])

    ax_box.boxplot(
        [pre_vals[np.isfinite(pre_vals)], post_vals[np.isfinite(post_vals)]],
        positions=[0, 1],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(alpha=0.6),
        medianprops=dict(linewidth=1.2),
    )
    ax_box.set_xticks([0, 1])
    ax_box.set_xticklabels([f"Pre (<{baseline_day:g} d)", f"Post (≥{baseline_day:g} d)"])
    ax_box.set_ylabel(ylabel)
    ax_box.set_title(f"{subject}  pre vs post")
    ax_box.set_ylim(vmin, vmax)
    ax_box.text(0.5, vmax - 0.06 * (vmax - vmin), format_p(p_val), ha="center", va="top")
    clean_axes(ax_box)

    ax_ts.plot(df["time"], df[ycol], marker="o", lw=1.8)
    ax_ts.axvline(float(baseline_day), ls="--", lw=1.2)
    ax_ts.set_xlabel("Day")
    ax_ts.set_ylabel(ylabel)
    clean_axes(ax_ts)

    fig.tight_layout()
    ensure_dir(Path(out_base).parent)
    fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return p_val


def summarize_pre_post(df: pd.DataFrame, ycol: str) -> Dict[str, float]:
    pre = df.loc[df["phase"] == "pre", ycol].to_numpy(dtype=float)
    post = df.loc[df["phase"] == "post", ycol].to_numpy(dtype=float)
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    return {
        "n_pre": float(pre.size),
        "n_post": float(post.size),
        "pre_mean": float(np.mean(pre)) if pre.size else float("nan"),
        "post_mean": float(np.mean(post)) if post.size else float("nan"),
        "delta_post_minus_pre": float(np.mean(post) - np.mean(pre)) if (pre.size and post.size) else float("nan"),
    }


def run(root_dir: str, cfg: FingerprintConfig, *, font_family: str, font_size: int) -> pd.DataFrame:
    root = Path(root_dir)
    out = ensure_dir(cfg.out_dir)

    subjects = sorted([p.name for p in root.iterdir() if p.is_dir()])
    rows: List[Dict[str, object]] = []

    for sid in subjects:
        df = read_subject_table(root / sid, cfg)
        res = subject_time_series(df, cfg)
        if res is None or res.empty:
            continue

        p_dist = plot_box_and_time(
            sid,
            res,
            ycol="distance",
            ylabel="Fingerprint distance",
            out_base=str(out / f"{sid}_spectral_distance"),
            baseline_day=cfg.baseline_day,
            font_family=font_family,
            font_size=font_size,
        )
        dist_sum = summarize_pre_post(res, "distance")

        p_shan = plot_box_and_time(
            sid,
            res,
            ycol="shannon",
            ylabel="Shannon index",
            out_base=str(out / f"{sid}_shannon_index"),
            baseline_day=cfg.baseline_day,
            font_family=font_family,
            font_size=font_size,
        )
        shan_sum = summarize_pre_post(res, "shannon")

        rows.append(
            {
                "subject": sid,
                "distance_p": p_dist,
                **{f"distance_{k}": v for k, v in dist_sum.items()},
                "shannon_p": p_shan,
                **{f"shannon_{k}": v for k, v in shan_sum.items()},
            }
        )

    stats_df = pd.DataFrame(rows)
    if not stats_df.empty:
        pvals_all = np.concatenate(
            [stats_df["distance_p"].to_numpy(dtype=float), stats_df["shannon_p"].to_numpy(dtype=float)]
        )
        qvals_all = bh_fdr(pvals_all)
        n = len(stats_df)
        stats_df["distance_q_bh"] = qvals_all[:n]
        stats_df["shannon_q_bh"] = qvals_all[n:]

    stats_df.to_csv(out / "pre_post_significance_stats.csv", index=False)
    return stats_df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spectral_fingerprints.py")
    p.add_argument("--root-dir", required=True)
    p.add_argument("--out-dir", default="results/fingerprints")
    p.add_argument("--input-filename", default="genus.csv")
    p.add_argument("--time-col", default="time")
    p.add_argument("--baseline-day", type=float, default=14.0)
    p.add_argument("--top-g", type=int, default=15)
    p.add_argument("--window-len", type=int, default=5)
    p.add_argument("--min-abund", type=float, default=1e-4)

    p.add_argument("--font-family", default="DejaVu Sans")
    p.add_argument("--font-size", type=int, default=10)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = FingerprintConfig(
        input_filename=str(args.input_filename),
        time_col=str(args.time_col),
        baseline_day=float(args.baseline_day),
        top_g=int(args.top_g),
        window_len=int(args.window_len),
        min_abund=float(args.min_abund),
        out_dir=str(args.out_dir),
    )
    df = run(
        root_dir=str(args.root_dir),
        cfg=cfg,
        font_family=str(args.font_family),
        font_size=int(args.font_size),
    )
    print("saved:", str(Path(cfg.out_dir) / "pre_post_significance_stats.csv"))
    print("n_subjects:", len(df))


if __name__ == "__main__":
    main()
