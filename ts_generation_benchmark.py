from __future__ import annotations

import argparse
import json
import os
import warnings
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


def save_figure(out_base: str, transparent: bool = False, save_pdf: bool = True) -> None:
    plt.savefig(out_base + ".png", dpi=300, bbox_inches="tight", transparent=transparent)
    if save_pdf:
        plt.savefig(out_base + ".pdf", bbox_inches="tight", transparent=transparent)


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


def clr_transform(X: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    Xp = np.maximum(X, eps)
    Xp = Xp / Xp.sum(axis=1, keepdims=True)
    L = np.log(Xp)
    return L - L.mean(axis=1, keepdims=True)


def clr_inverse(Y: np.ndarray) -> np.ndarray:
    E = np.exp(Y - Y.max(axis=1, keepdims=True))
    return enforce_simplex(E)


def gen_clr_jitter_repeat_by_time(X: np.ndarray, total_rows: int, m: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    rep = int(np.ceil(total_rows / m))
    Xr = np.vstack([X for _ in range(rep)])[:total_rows].copy()
    C = clr_transform(Xr)
    base_std = np.std(clr_transform(X), axis=0, ddof=1)
    base_std = np.where(np.isfinite(base_std) & (base_std > 0), base_std, 1.0)
    noise = rng.normal(0.0, sigma, size=C.shape) * base_std[None, :]
    return clr_inverse(C + noise)


def fit_dirichlet_moment(X: np.ndarray, eps: float = 1e-10) -> Tuple[np.ndarray, float]:
    Xp = np.maximum(X, eps)
    Xp = Xp / Xp.sum(axis=1, keepdims=True)
    mu = Xp.mean(axis=0)
    var = Xp.var(axis=0, ddof=1)
    var = np.where(np.isfinite(var) & (var > 0), var, np.nan)
    t = (mu * (1.0 - mu)) / var - 1.0
    t = t[np.isfinite(t)]
    alpha0 = float(np.median(t)) if t.size > 0 else 100.0
    if not np.isfinite(alpha0) or alpha0 <= 0:
        alpha0 = 100.0
    alpha = mu * alpha0
    alpha = np.where(np.isfinite(alpha) & (alpha > 1e-6), alpha, 1e-3)
    return alpha.astype(float), float(alpha0)


def gen_dirichlet_global(X: np.ndarray, total_rows: int, rng: np.random.Generator, alpha0_floor: float) -> np.ndarray:
    alpha, _ = fit_dirichlet_moment(X)
    alpha0 = float(alpha.sum())
    if not np.isfinite(alpha0) or alpha0 <= 0:
        alpha = np.full_like(alpha, 1.0, dtype=float)
        alpha0 = float(alpha.sum())
    if alpha0 < alpha0_floor:
        scale = alpha0_floor / alpha0
        alpha = alpha * scale
    Y = rng.dirichlet(alpha, size=total_rows)
    return enforce_simplex(Y.astype(float))


def gen_pca_gaussian_softmax(X: np.ndarray, total_rows: int, rng: np.random.Generator, pca_components: int, standardize: bool) -> np.ndarray:
    Z = X.copy()
    scaler = None
    if standardize:
        scaler = StandardScaler()
        Z = scaler.fit_transform(Z)

    k = int(min(max(2, pca_components), Z.shape[1], max(2, Z.shape[0] - 1)))
    p = PCA(n_components=k, random_state=0)
    Zp = p.fit_transform(Z)

    mu = Zp.mean(axis=0)
    cov = np.cov(Zp, rowvar=False)
    cov = np.atleast_2d(cov).astype(float)
    cov = (cov + cov.T) * 0.5
    cov = cov + (1e-6 * np.eye(cov.shape[0], dtype=float))

    try:
        S = rng.multivariate_normal(mean=mu, cov=cov, size=total_rows, method="cholesky")
    except Exception:
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 1e-8, None)
        A = V @ np.diag(np.sqrt(w)) @ V.T
        S = rng.normal(0.0, 1.0, size=(total_rows, cov.shape[0])) @ A.T + mu[None, :]

    Zs = p.inverse_transform(S)
    if scaler is not None:
        Zs = scaler.inverse_transform(Zs)

    Y = np.exp(Zs - Zs.max(axis=1, keepdims=True))
    return enforce_simplex(Y.astype(float))


def gen_knn_mixup(X: np.ndarray, total_rows: int, rng: np.random.Generator, k: int, beta_a: float, clr_sigma: float) -> np.ndarray:
    m, _ = X.shape
    Xs = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=min(max(2, k), m), metric="euclidean").fit(Xs)

    base_idx = rng.integers(0, m, size=total_rows, endpoint=False)
    neigh_idx = nn.kneighbors(Xs[base_idx], n_neighbors=min(max(2, k), m), return_distance=False)
    pick = rng.integers(1, neigh_idx.shape[1], size=total_rows, endpoint=False)
    j_idx = neigh_idx[np.arange(total_rows), pick]

    lam = rng.beta(beta_a, beta_a, size=total_rows).astype(float)
    Y = lam[:, None] * X[base_idx] + (1.0 - lam)[:, None] * X[j_idx]
    Y = enforce_simplex(Y.astype(float))

    if clr_sigma > 0:
        C = clr_transform(Y)
        std = np.std(clr_transform(X), axis=0, ddof=1)
        std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
        noise = rng.normal(0.0, clr_sigma, size=C.shape) * std[None, :]
        Y = clr_inverse(C + noise)
    return Y


def safe_float(v) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


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
    warnings.filterwarnings("ignore", category=UserWarning)

    p = argparse.ArgumentParser(prog="ts_generation_benchmark.py")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--time-col", default="time")
    p.add_argument("--n-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--clr-sigma", type=float, default=0.10)
    p.add_argument("--dirichlet-alpha0-floor", type=float, default=50.0)

    p.add_argument("--pca-components", type=int, default=10)
    p.add_argument("--pca-gen-components", type=int, default=10)
    p.add_argument("--pca-gen-standardize", action="store_true")

    p.add_argument("--mixup-k", type=int, default=10)
    p.add_argument("--mixup-beta-a", type=float, default=0.6)
    p.add_argument("--mixup-clr-sigma", type=float, default=0.05)

    p.add_argument("--umap-standardize", action="store_true")
    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.10)
    p.add_argument("--plot-links", action="store_true")
    p.add_argument("--link-alpha", type=float, default=0.15)
    p.add_argument("--marker-size", type=float, default=18.0)

    p.add_argument("--font-family", type=str, default="DejaVu Sans")
    p.add_argument("--font-size", type=int, default=10)
    p.add_argument("--transparent", action="store_true")
    p.add_argument("--no-pdf", action="store_true")

    args = p.parse_args()
    ensure_dir(args.output_dir)

    time_vec, numeric_df, X = load_table(args.input, time_col=args.time_col)
    m = X.shape[0]
    total_rows = m if args.n_samples is None else int(args.n_samples)
    n_sets = int(np.ceil(total_rows / m))

    real_time_vec = time_vec
    src_time_vec = (real_time_vec * n_sets)[:total_rows]
    sample_ids = make_sample_ids(n_samples=total_rows, base_m=m)

    rng = np.random.default_rng(args.seed)

    methods = [
        "clr_jitter_repeat_by_time",
        "dirichlet_global",
        "pca_gaussian_softmax",
        "knn_mixup",
    ]

    summary_rows: List[Dict[str, float]] = []

    for method in methods:
        if method == "clr_jitter_repeat_by_time":
            Y = gen_clr_jitter_repeat_by_time(X, total_rows=total_rows, m=m, rng=rng, sigma=args.clr_sigma)
        elif method == "dirichlet_global":
            Y = gen_dirichlet_global(X, total_rows=total_rows, rng=rng, alpha0_floor=args.dirichlet_alpha0_floor)
        elif method == "pca_gaussian_softmax":
            Y = gen_pca_gaussian_softmax(
                X, total_rows=total_rows, rng=rng,
                pca_components=args.pca_gen_components,
                standardize=bool(args.pca_gen_standardize),
            )
        elif method == "knn_mixup":
            Y = gen_knn_mixup(
                X, total_rows=total_rows, rng=rng,
                k=args.mixup_k, beta_a=args.mixup_beta_a, clr_sigma=args.mixup_clr_sigma
            )
        else:
            raise ValueError(f"unknown method: {method}")

        synth_path = os.path.join(args.output_dir, f"{method}_synth.csv")
        metrics_path = os.path.join(args.output_dir, f"{method}_metrics.json")
        umap_base = os.path.join(args.output_dir, f"{method}_umap_time_match")

        out_numeric = pd.DataFrame(Y, columns=numeric_df.columns)
        out_df = pd.concat([pd.Series(sample_ids, name="sample_id"), out_numeric], axis=1)
        out_df["source_time"] = src_time_vec
        out_df.to_csv(synth_path, index=False)

        metrics = compute_metrics(X, Y, seed=args.seed, pca_components=args.pca_components)
        acc = plot_umap_time_match(
            X_real=X,
            X_synth=Y,
            src_time=src_time_vec,
            real_time=real_time_vec,
            out_base=umap_base,
            title=f"{method} UMAP time-match",
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

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        row: Dict[str, float] = {"method": method}
        for k, v in metrics.items():
            row[k] = safe_float(v)
        summary_rows.append(row)

        print("saved:", synth_path)
        print("saved:", umap_base + ".png")
        if not args.no_pdf:
            print("saved:", umap_base + ".pdf")
        print("saved:", metrics_path)
        print("time-match accuracy:", acc)

    summary_df = pd.DataFrame(summary_rows)
    col_order = ["method"] + [c for c in summary_df.columns if c != "method"]
    summary_df = summary_df[col_order]

    summary_csv = os.path.join(args.output_dir, "summary_metrics.csv")
    summary_json = os.path.join(args.output_dir, "summary_metrics.json")
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("saved:", summary_csv)
    print("saved:", summary_json)
    with pd.option_context("display.width", 200, "display.max_columns", 200):
        print(summary_df)


if __name__ == "__main__":
    main()
