from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import umap

from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from scipy import linalg as sla

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


EPS = 1e-12


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def save_figure(out_base: str, transparent: bool = False, save_pdf: bool = True) -> Tuple[str, Optional[str]]:
    png_path = out_base + ".png"
    pdf_path = out_base + ".pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", transparent=transparent)
    if save_pdf:
        plt.savefig(pdf_path, bbox_inches="tight", transparent=transparent)
        return png_path, pdf_path
    return png_path, None


def enforce_simplex(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    s = x.sum(axis=-1, keepdims=True)
    return np.where(s > 0, x / s, np.full_like(x, 1.0 / x.shape[-1]))


def make_sample_ids(n_samples: int, base_m: int) -> List[str]:
    base = [f"S{i}" for i in range(base_m)]
    if n_samples <= base_m:
        return base[:n_samples]
    out: List[str] = []
    times = n_samples // base_m
    rem = n_samples % base_m
    counter = 1
    for _ in range(times):
        for s in base:
            out.append(f"{s}_syn{counter}")
            counter += 1
    for s in base[:rem]:
        out.append(f"{s}_syn{counter}")
        counter += 1
    return out


def generate_fourier_coherent(
    X: np.ndarray,
    n_sets: int,
    keep_k: Optional[int],
    amp_jitter: float,
    p_modify: float,
    rng: np.random.Generator,
    min_eps: float = 1e-6,
) -> np.ndarray:
    m, n = X.shape
    K = m // 2 + 1
    F = np.fft.rfft(X, axis=0)
    x_l2 = np.linalg.norm(X, ord=2, axis=0, keepdims=True) + EPS
    A = np.abs(F) / x_l2
    std = A.std(axis=1)

    kk = K if keep_k is None else int(min(max(1, keep_k), K))
    candidates = np.arange(1, kk, dtype=int)
    if candidates.size == 0:
        candidates = np.array([0], dtype=int)
    n_modify_default = max(1, int(round(p_modify * len(candidates))))

    Ys: List[np.ndarray] = []
    for _ in range(n_sets):
        F_i = np.array(F, copy=True)
        n_modify = min(n_modify_default, len(candidates))
        sel = rng.choice(candidates, size=n_modify, replace=False)
        for k in sel:
            v = F[k, :].astype(np.complex128, copy=False)
            if not np.isfinite(v).all():
                continue
            S_k = np.outer(v, np.conj(v)) / max(n, 1)
            S_k = (S_k + S_k.conj().T) * 0.5
            eps = float(np.linalg.norm(S_k)) * 1e-8 + min_eps
            S_k += eps * np.eye(n, dtype=np.complex128)
            try:
                _, U = np.linalg.eigh(S_k)
            except np.linalg.LinAlgError:
                continue
            sigma_k = (std[k] if K > 1 else 0.1) * max(float(amp_jitter), 1e-6)
            d = rng.normal(loc=0.0, scale=sigma_k, size=n)
            d = np.clip(1.0 + d, 1e-6, None)
            D = np.diag(d.astype(np.float64))
            T = U @ D @ U.conj().T
            F_i[k, :] = (T @ v)

        X_i = np.fft.irfft(F_i, n=m, axis=0).astype(np.float64)
        X_i = enforce_simplex(X_i)
        Ys.append(X_i)

    return np.vstack(Ys)


def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray, max_samples: int = 2000) -> float:
    Z = np.vstack([X, Y])
    if Z.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(Z.shape[0], size=max_samples, replace=False)
        Z = Z[idx]
    d = pdist(Z, metric="euclidean")
    d = d[np.isfinite(d)]
    d = d[d > 0]
    med = np.median(d) if d.size > 0 else 1.0
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return float(1.0 / (2.0 * (med**2)))


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    d = cdist(a, b, metric="sqeuclidean")
    return np.exp(-gamma * d)


def mmd_rbf_unbiased(X: np.ndarray, Y: np.ndarray, gamma: Optional[float] = None) -> float:
    m = X.shape[0]
    n = Y.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    if gamma is None:
        gamma = median_heuristic_gamma(X, Y)
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2.0 * Kxy.mean()
    return float(max(mmd2, 0.0))


def frechet_distance_gaussian(pca_X: np.ndarray, pca_Y: np.ndarray) -> float:
    if pca_X.shape[0] < 2 or pca_Y.shape[0] < 2:
        return float("nan")
    mu1 = pca_X.mean(axis=0)
    mu2 = pca_Y.mean(axis=0)
    c1 = np.cov(pca_X, rowvar=False)
    c2 = np.cov(pca_Y, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = sla.sqrtm(c1.dot(c2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = np.zeros_like(c1)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff.dot(diff) + np.trace(c1 + c2 - 2.0 * covmean))
    return float(max(fid, 0.0))


def feature_wasserstein_stats(X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return float("nan"), float("nan")
    ws = []
    for j in range(X.shape[1]):
        ws.append(wasserstein_distance(X[:, j], Y[:, j]))
    ws = np.asarray(ws, dtype=float)
    ws = ws[np.isfinite(ws)]
    if ws.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(ws)), float(np.median(ws))


def separability_auc(X: np.ndarray, Y: np.ndarray, seed: int) -> float:
    Z = np.vstack([X, Y])
    y = np.hstack([np.zeros(X.shape[0]), np.ones(Y.shape[0])])
    if np.unique(y).size < 2:
        return float("nan")

    n_splits = 5
    min_class = int(min((y == 0).sum(), (y == 1).sum()))
    if min_class < 2:
        return float("nan")
    n_splits = min(n_splits, min_class)
    if n_splits < 2:
        return float("nan")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(Z, y):
        clf = LogisticRegression(max_iter=300, solver="lbfgs")
        clf.fit(Z[tr], y[tr])
        p = clf.predict_proba(Z[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if len(aucs) else float("nan")


def coverage_at_threshold(X: np.ndarray, Y: np.ndarray, quantile: float = 0.95) -> Dict[str, float]:
    if X.shape[0] < 3 or Y.shape[0] < 1:
        return {
            f"coverage@{quantile:.2f}": float("nan"),
            "mean_nn_dist_synth_to_real": float("nan"),
            "median_nn_dist_synth_to_real": float("nan"),
            "threshold_real_real": float("nan"),
        }
    nn_r = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(X)
    dist_rr, _ = nn_r.kneighbors(X, n_neighbors=2, return_distance=True)
    rr = dist_rr[:, 1]
    T_rr = np.quantile(rr, quantile)

    nn_sr = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(X)
    dist_sr, _ = nn_sr.kneighbors(Y, n_neighbors=1, return_distance=True)
    d_sr = dist_sr[:, 0]
    coverage = float(np.mean(d_sr <= T_rr))
    return {
        f"coverage@{quantile:.2f}": coverage,
        "mean_nn_dist_synth_to_real": float(np.mean(d_sr)),
        "median_nn_dist_synth_to_real": float(np.median(d_sr)),
        "threshold_real_real": float(T_rr),
    }


def compute_metrics(X: np.ndarray, Y: np.ndarray, seed: int, pca_components: int) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["separability_auc"] = separability_auc(X, Y, seed=seed)
    metrics["mmd2_rbf_median_heuristic"] = mmd_rbf_unbiased(X, Y, gamma=None)

    n = X.shape[1]
    pca_k = min(pca_components, n, max(2, X.shape[0] - 1), max(2, Y.shape[0] - 1))
    pca_k = int(max(2, pca_k))
    try:
        pca = PCA(n_components=pca_k, random_state=seed)
        Z_all = np.vstack([X, Y])
        pca.fit(Z_all)
        X_p = pca.transform(X)
        Y_p = pca.transform(Y)
        metrics["frechet_distance_pca"] = frechet_distance_gaussian(X_p, Y_p)
        metrics["pca_components_used"] = int(pca_k)
    except Exception:
        metrics["frechet_distance_pca"] = float("nan")
        metrics["pca_components_used"] = int(pca_k)

    ws_mean, ws_median = feature_wasserstein_stats(X, Y)
    metrics["wasserstein_mean"] = ws_mean
    metrics["wasserstein_median"] = ws_median
    metrics.update(coverage_at_threshold(X, Y, quantile=0.95))
    return metrics


def plot_umap_time_match(
    X_real: np.ndarray,
    X_synth: np.ndarray,
    src_time: Optional[List],
    real_time: Optional[List],
    out_base: str,
    title: str,
    standardize: bool,
    n_neighbors: int,
    min_dist: float,
    plot_links: bool,
    link_alpha: float,
    marker_size: float,
    seed: int,
    font_family: str,
    font_size: int,
    transparent: bool,
    save_pdf: bool,
) -> float:
    set_plot_style(font_family=font_family, font_size=font_size)

    Z = np.vstack([X_real, X_synth])
    if standardize:
        Z = StandardScaler().fit_transform(Z)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    emb = reducer.fit_transform(Z)

    E_real = emb[: X_real.shape[0]]
    E_syn = emb[X_real.shape[0] :]

    nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(E_real)
    _, nn_idx = nbrs.kneighbors(E_syn, n_neighbors=1, return_distance=True)
    nn_idx = nn_idx.ravel()

    if src_time is None or real_time is None:
        match = np.ones(E_syn.shape[0], dtype=bool)
    else:
        rt = np.asarray(real_time)
        st = np.asarray(src_time)
        match = (rt[nn_idx] == st)

    acc = float(np.mean(match)) if match.size else float("nan")

    fig = plt.figure(figsize=(3.4, 3.1))
    ax = plt.gca()

    rasterize_points = emb.shape[0] > 5000
    ax.scatter(E_real[:, 0], E_real[:, 1], s=max(6.0, marker_size * 0.45), alpha=0.9, label="Real", rasterized=rasterize_points)

    wrong = ~match
    ax.scatter(E_syn[wrong, 0], E_syn[wrong, 1], s=max(6.0, marker_size * 0.60), marker="x", alpha=0.9, label="Generated (mismatch)", rasterized=rasterize_points, linewidths=1.1)
    ax.scatter(E_syn[match, 0], E_syn[match, 1], s=max(6.0, marker_size * 0.50), marker="x", alpha=0.9, label="Generated (match)", rasterized=rasterize_points, linewidths=1.1)

    if plot_links:
        for i in range(E_syn.shape[0]):
            j = nn_idx[i]
            ax.plot([E_syn[i, 0], E_real[j, 0]], [E_syn[i, 1], E_real[j, 1]], lw=0.4, alpha=link_alpha, c="0.6", rasterized=rasterize_points)

    ax.set_title(f"{title} (acc={acc:.3f})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", frameon=False, handlelength=1.5, handletextpad=0.4)
    clean_axes(ax)
    fig.tight_layout()

    ensure_dir(os.path.dirname(out_base) or ".")
    save_figure(out_base, transparent=transparent, save_pdf=save_pdf)
    plt.close(fig)
    return acc


def load_table(input_csv: str, time_col: str) -> Tuple[List, pd.DataFrame, np.ndarray]:
    df = pd.read_csv(input_csv)
    if time_col not in df.columns:
        raise ValueError(f"time column not found: {time_col}")
    time_vec = df[time_col].tolist()
    feature_df = df.drop(columns=[time_col])
    numeric_df = feature_df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError("no numeric features after dropping time")
    X = numeric_df.to_numpy(dtype=float)
    X = enforce_simplex(np.maximum(X, 0.0))
    return time_vec, numeric_df, X


def main() -> None:
    p = argparse.ArgumentParser(prog="fourier_ts_generate.py")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--time-col", default="time")

    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-k", type=int, default=None)
    p.add_argument("--amp-jitter", type=float, default=0.15)
    p.add_argument("--p-modify", type=float, default=0.10)

    p.add_argument("--eval", action="store_true", help="compute metrics + UMAP time-match plot")
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--umap-out", default=None)
    p.add_argument("--umap-standardize", action="store_true")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.10)
    p.add_argument("--plot-links", action="store_true")
    p.add_argument("--link-alpha", type=float, default=0.15)
    p.add_argument("--marker-size", type=float, default=18.0)
    p.add_argument("--pca-components", type=int, default=10)

    p.add_argument("--font-family", type=str, default="DejaVu Sans")
    p.add_argument("--font-size", type=int, default=10)
    p.add_argument("--transparent", action="store_true")
    p.add_argument("--no-pdf", action="store_true")

    args = p.parse_args()

    time_vec, numeric_df, X = load_table(args.input, time_col=args.time_col)
    m = X.shape[0]
    total_rows = m if args.n_samples is None else int(args.n_samples)
    n_sets = int(np.ceil(total_rows / m))

    rng = np.random.default_rng(args.seed)
    Y_full = generate_fourier_coherent(
        X=X,
        n_sets=n_sets,
        keep_k=args.keep_k,
        amp_jitter=args.amp_jitter,
        p_modify=args.p_modify,
        rng=rng,
    )
    Y = Y_full[:total_rows]

    src_time_vec = (time_vec * n_sets)[:total_rows]
    sample_ids = make_sample_ids(n_samples=total_rows, base_m=m)

    ensure_dir(os.path.dirname(args.output) or ".")
    out_numeric = pd.DataFrame(Y, columns=numeric_df.columns)
    out_df = pd.concat([pd.Series(sample_ids, name="sample_id"), out_numeric], axis=1)
    out_df["source_time"] = src_time_vec
    out_df.to_csv(args.output, index=False)

    if not args.eval:
        print("saved:", args.output)
        return

    metrics_out = args.metrics_out or (os.path.splitext(args.output)[0] + "_metrics.json")
    umap_out = args.umap_out or (os.path.splitext(args.output)[0] + "_umap_time_match")

    ensure_dir(os.path.dirname(metrics_out) or ".")
    metrics = compute_metrics(X, Y, seed=args.seed, pca_components=args.pca_components)
    acc = plot_umap_time_match(
        X_real=X,
        X_synth=Y,
        src_time=src_time_vec,
        real_time=time_vec,
        out_base=umap_out,
        title="Fourier coherent UMAP time-match",
        standardize=bool(args.umap_standardize),
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        plot_links=bool(args.plot_links),
        link_alpha=float(args.link_alpha),
        marker_size=float(args.marker_size),
        seed=int(args.seed),
        font_family=str(args.font_family),
        font_size=int(args.font_size),
        transparent=bool(args.transparent),
        save_pdf=(not bool(args.no_pdf)),
    )
    metrics["time_match_accuracy_umap_nn"] = acc

    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("saved:", args.output)
    print("saved:", metrics_out)
    print("saved:", umap_out + ".png")
    if not args.no_pdf:
        print("saved:", umap_out + ".pdf")
    print("time-match accuracy:", acc)


if __name__ == "__main__":
    main()
