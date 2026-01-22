from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class NetworkConfig:
    min_days: int = 8
    apply_clr_microbe: bool = True
    microbe_pseudocount: float = 1e-6
    metabolite_pseudocount: float = 1e-6

    band_periods: Tuple[Tuple[float, float], ...] = ((7.0, 0.5),)  # (center_days, width_frac)
    nperseg: int = 8
    overlap: float = 0.5

    n_perm: int = 200
    phase_r_min: float = 0.30
    xcorr_min_peak: float = 0.15
    max_lag_cap: int = 10

    seed: int = 12345
    q_threshold: float = 0.10

    time_min: Optional[float] = None
    time_max: Optional[float] = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_time_series_table(csv_path: str, *, time_col: str, time_min: Optional[float], time_max: Optional[float]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"time column not found: {time_col}")

    df = df.copy()
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").astype(float)
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()

    if time_min is not None:
        df = df.loc[df.index >= float(time_min)]
    if time_max is not None:
        df = df.loc[df.index <= float(time_max)]

    X = df.apply(pd.to_numeric, errors="coerce")
    X = X.interpolate(limit_direction="both").ffill().bfill()
    return X


def clr_transform(A: np.ndarray, pseudocount: float) -> np.ndarray:
    X = A.astype(float, copy=True)
    X[X < 0] = 0.0
    X += float(pseudocount)
    s = X.sum(axis=1, keepdims=True)
    X = X / np.maximum(s, EPS)
    L = np.log(np.maximum(X, EPS))
    return L - L.mean(axis=1, keepdims=True)


def maybe_log_transform_nonnegative(A: np.ndarray, pseudocount: float) -> np.ndarray:
    Y = A.astype(float, copy=True)
    nonneg_cols = np.where(np.all(Y >= 0, axis=0))[0]
    if nonneg_cols.size > 0:
        Y[:, nonneg_cols] = np.log(Y[:, nonneg_cols] + float(pseudocount))
    return Y


def stft_matrix(A: np.ndarray, *, nperseg: int, noverlap: int) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    n, v = A.shape
    nperseg = int(min(max(2, nperseg), n))
    noverlap = int(min(max(0, noverlap), nperseg - 1))
    step = nperseg - noverlap
    if step <= 0:
        step = 1

    starts = np.arange(0, max(n - nperseg + 1, 1), step)
    if starts.size == 0:
        starts = np.array([0], dtype=int)
        nperseg = n
        noverlap = 0

    w = np.hanning(nperseg).reshape(-1, 1)
    U = float((w[:, 0] ** 2).sum())

    blocks = []
    for s in starts:
        seg = A[s : s + nperseg, :]
        seg = seg - seg.mean(axis=0, keepdims=True)
        seg = seg * w
        Xf = np.fft.rfft(seg, axis=0)
        blocks.append(Xf)

    Xf_all = np.stack(blocks, axis=0)  # (m, f, v)
    f = np.fft.rfftfreq(nperseg, d=1.0)
    return Xf_all, f, U, nperseg, int(len(starts))


def psd_from_stft(Xf_all: np.ndarray, *, U: float, nperseg: int, n_segments: int) -> Tuple[np.ndarray, np.ndarray]:
    Xf2 = np.swapaxes(Xf_all, 0, 1)  # (f, m, v)
    S = (np.einsum("fmp,fmp->fp", np.conj(Xf2), Xf2)).real / (max(U, EPS) * nperseg * max(n_segments, 1))
    return S, Xf2


def csd_from_stft(Xf2: np.ndarray, Yf2: np.ndarray, *, U: float, nperseg: int, n_segments: int) -> np.ndarray:
    return np.einsum("fmp,fmq->fpq", Xf2, np.conj(Yf2)) / (max(U, EPS) * nperseg * max(n_segments, 1))


def band_metrics(
    Sxx: np.ndarray,
    Syy: np.ndarray,
    Sxy: np.ndarray,
    freqs: np.ndarray,
    flo: float,
    fhi: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]]:
    mask = (freqs > 0) & (freqs >= flo) & (freqs <= fhi)
    if mask.sum() == 0:
        return None

    W = Sxx[mask, :, None] * Syy[mask, None, :]  # (k, p, q)
    den = W.sum(axis=0)  # (p, q)
    num = (np.abs(Sxy[mask]) ** 2).sum(axis=0)  # (p, q)

    assoc = np.clip(num / np.maximum(den, 1e-15), 0.0, 1.0)

    Z = (W * np.exp(1j * np.angle(Sxy[mask]))).sum(axis=0)  # (p, q)
    R = np.abs(Z) / np.maximum(den, 1e-15)

    fbar = (W * freqs[mask][:, None, None]).sum(axis=0) / np.maximum(den, 1e-15)
    phi = np.angle(Z)

    delay_phase_days = np.where(fbar > 0, phi / (2 * np.pi * np.maximum(fbar, 1e-15)), 0.0)
    period_days = np.where(fbar > 0, 1.0 / np.maximum(fbar, 1e-15), np.inf)

    return assoc, delay_phase_days, period_days, R, int(mask.sum()), fbar


def bandpass_zscore(A: np.ndarray, flo: float, fhi: float) -> np.ndarray:
    n = A.shape[0]
    F = np.fft.rfftfreq(n, d=1.0)
    mask = (F >= max(0.0, flo)) & (F <= fhi)
    Af = np.fft.rfft(A - A.mean(axis=0, keepdims=True), axis=0)
    Z = np.zeros_like(Af)
    Z[mask, :] = Af[mask, :]
    B = np.fft.irfft(Z, n=n, axis=0).real
    B = (B - B.mean(axis=0, keepdims=True)) / (B.std(axis=0, keepdims=True) + 1e-12)
    return B


def xcorr_peak_delay(Xb: np.ndarray, Yb: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    n = Xb.shape[0]
    max_lag = int(max(1, max_lag))
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)

    C = []
    for L in lags:
        if L >= 0:
            Xs = Xb[L:, :]
            Ys = Yb[: n - L, :]
            d = max(n - L, 1)
        else:
            Xs = Xb[: n + L, :]
            Ys = Yb[-L:, :]
            d = max(n + L, 1)
        C.append((Xs.T @ Ys) / d)
    C = np.stack(C, axis=0)  # (lags, p, q)

    idx = np.argmax(np.abs(C), axis=0)  # (p, q)
    peak = np.take_along_axis(C, idx[None, :, :], axis=0).squeeze(0)
    delay = lags[idx]
    return delay, peak


def phase_randomize(Y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, q = Y.shape
    Yf = np.fft.rfft(Y - Y.mean(axis=0, keepdims=True), axis=0)
    amp = np.abs(Yf)
    ph = np.angle(Yf)
    if Yf.shape[0] > 1:
        ph[1:, :] = rng.uniform(-np.pi, np.pi, size=(Yf.shape[0] - 1, q))
    Yr = amp * np.exp(1j * ph)
    return np.fft.irfft(Yr, n=n, axis=0).real


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, m + 1)

    q = p * m / np.maximum(ranks, 1)
    q[q > 1] = 1.0

    out = np.empty_like(q)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        prev = min(prev, q[idx])
        out[idx] = prev
    return out


def parse_band_periods(s: str) -> Tuple[Tuple[float, float], ...]:
    if not s.strip():
        return ((7.0, 0.5),)
    items = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("band periods must be 'center_days:width_frac' comma-separated")
        c, w = part.split(":", 1)
        items.append((float(c), float(w)))
    if not items:
        items = [(7.0, 0.5)]
    return tuple(items)


def compute_delay_network(
    microbe_csv: str,
    metabolite_csv: str,
    out_dir: str,
    *,
    time_col: str,
    cfg: NetworkConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    outdir = ensure_dir(out_dir)

    M = load_time_series_table(microbe_csv, time_col=time_col, time_min=cfg.time_min, time_max=cfg.time_max)
    T = load_time_series_table(metabolite_csv, time_col=time_col, time_min=cfg.time_min, time_max=cfg.time_max)

    common_time = np.intersect1d(M.index.values, T.index.values)
    M = M.loc[common_time]
    T = T.loc[common_time]

    if len(M) < cfg.min_days:
        raise RuntimeError(f"insufficient overlapping days: {len(M)} < {cfg.min_days}")

    X = M.to_numpy(dtype=float)
    Y = T.to_numpy(dtype=float)

    if cfg.apply_clr_microbe:
        X = clr_transform(X, pseudocount=cfg.microbe_pseudocount)
    Y = maybe_log_transform_nonnegative(Y, pseudocount=cfg.metabolite_pseudocount)

    n, p = X.shape
    q = Y.shape[1]

    noverlap = int(cfg.overlap * min(cfg.nperseg, n))
    nperseg = int(min(cfg.nperseg, n))

    Xf_all, freqs, Ux, nps, nseg_x = stft_matrix(X, nperseg=nperseg, noverlap=noverlap)
    Yf_all, _, Uy, nps2, nseg_y = stft_matrix(Y, nperseg=nperseg, noverlap=noverlap)

    Sxx, Xf2 = psd_from_stft(Xf_all, U=Ux, nperseg=nps, n_segments=nseg_x)
    Syy, Yf2 = psd_from_stft(Yf_all, U=Uy, nperseg=nps2, n_segments=nseg_y)
    Sxy = csd_from_stft(Xf2, Yf2, U=Ux, nperseg=nps, n_segments=nseg_x)

    bands = []
    for center_days, width_frac in cfg.band_periods:
        f0 = 1.0 / float(center_days)
        flo = f0 * (1.0 - float(width_frac))
        fhi = f0 * (1.0 + float(width_frac))
        bands.append((float(center_days), float(flo), float(fhi)))

    assoc_obs, R_obs, dph_obs, per_obs, kpts_list, fbar_obs = [], [], [], [], [], []
    for _, flo, fhi in bands:
        bm = band_metrics(Sxx, Syy, Sxy, freqs, flo, fhi)
        if bm is None:
            assoc = np.zeros((p, q), dtype=float)
            dph = np.zeros((p, q), dtype=float)
            per = np.full((p, q), np.inf, dtype=float)
            R = np.zeros((p, q), dtype=float)
            kpts = 0
            fbar = np.zeros((p, q), dtype=float)
        else:
            assoc, dph, per, R, kpts, fbar = bm
        assoc_obs.append(assoc)
        R_obs.append(R)
        dph_obs.append(dph)
        per_obs.append(per)
        kpts_list.append(int(kpts))
        fbar_obs.append(fbar)

    delay_int_list, xcorr_peak_list = [], []
    for bi, (_, flo, fhi) in enumerate(bands):
        per = per_obs[bi]
        med_per = float(np.nanmedian(np.where(np.isfinite(per), per, np.nan)))
        if not np.isfinite(med_per) or med_per <= 0:
            max_lag = int(cfg.max_lag_cap)
        else:
            max_lag = int(np.clip(np.round(med_per), 1, cfg.max_lag_cap))

        Xb = bandpass_zscore(X, flo, fhi)
        Yb = bandpass_zscore(Y, flo, fhi)
        d_int, peak = xcorr_peak_delay(Xb, Yb, max_lag=max_lag)
        delay_int_list.append(d_int)
        xcorr_peak_list.append(peak)

    cnt_shift = [np.zeros((p, q), dtype=np.int32) for _ in bands]
    cnt_phase = [np.zeros((p, q), dtype=np.int32) for _ in bands]

    for _ in range(cfg.n_perm):
        s = int(rng.integers(0, n))
        Y_roll = np.roll(Y, s, axis=0)
        Yf_roll, _, _, _, nseg_yr = stft_matrix(Y_roll, nperseg=nperseg, noverlap=noverlap)
        Syy_r, Yf2_r = psd_from_stft(Yf_roll, U=Uy, nperseg=nps2, n_segments=nseg_yr)
        Sxy_r = csd_from_stft(Xf2, Yf2_r, U=Ux, nperseg=nps, n_segments=nseg_x)

        for bi, (_, flo, fhi) in enumerate(bands):
            mask = (freqs > 0) & (freqs >= flo) & (freqs <= fhi)
            if mask.sum() == 0:
                continue
            den = (Sxx[mask, :, None] * Syy_r[mask, None, :]).sum(axis=0)
            num = (np.abs(Sxy_r[mask]) ** 2).sum(axis=0)
            assoc_r = np.clip(num / np.maximum(den, 1e-15), 0.0, 1.0)
            cnt_shift[bi] += (assoc_r >= assoc_obs[bi]).astype(np.int32)

    for _ in range(cfg.n_perm):
        Yr = phase_randomize(Y, rng)
        Yfr, _, _, _, nseg_yr = stft_matrix(Yr, nperseg=nperseg, noverlap=noverlap)
        Syy_r, Yf2_r = psd_from_stft(Yfr, U=Uy, nperseg=nps2, n_segments=nseg_yr)
        Sxy_r = csd_from_stft(Xf2, Yf2_r, U=Ux, nperseg=nps, n_segments=nseg_x)

        for bi, (_, flo, fhi) in enumerate(bands):
            mask = (freqs > 0) & (freqs >= flo) & (freqs <= fhi)
            if mask.sum() == 0:
                continue
            den = (Sxx[mask, :, None] * Syy_r[mask, None, :]).sum(axis=0)
            num = (np.abs(Sxy_r[mask]) ** 2).sum(axis=0)
            assoc_r = np.clip(num / np.maximum(den, 1e-15), 0.0, 1.0)
            cnt_phase[bi] += (assoc_r >= assoc_obs[bi]).astype(np.int32)

    rows = []
    for bi, (center_days, flo, fhi) in enumerate(bands):
        p_shift = (1.0 + cnt_shift[bi]) / (1.0 + cfg.n_perm)
        p_phase = (1.0 + cnt_phase[bi]) / (1.0 + cfg.n_perm)
        p_cons = np.maximum(p_shift, p_phase)

        assoc = assoc_obs[bi]
        dph = dph_obs[bi]
        R = R_obs[bi]
        per = per_obs[bi]
        fbar = fbar_obs[bi]
        d_int = delay_int_list[bi]
        peak = xcorr_peak_list[bi]

        delay_use = np.where((R >= cfg.phase_r_min) & (np.abs(peak) >= cfg.xcorr_min_peak), d_int, 0)

        for i, mc in enumerate(M.columns):
            for j, tc in enumerate(T.columns):
                rows.append(
                    [
                        str(mc),
                        str(tc),
                        int(bi),
                        float(center_days),
                        float(flo),
                        float(fhi),
                        float(np.clip(assoc[i, j], 0.0, 1.0)),
                        float(dph[i, j]),
                        int(delay_use[i, j]),
                        float(per[i, j]),
                        float(R[i, j]),
                        int(kpts_list[bi]),
                        int(nseg_x),
                        float(fbar[i, j]),
                        float(peak[i, j]),
                        float(p_shift[i, j]),
                        float(p_phase[i, j]),
                        float(p_cons[i, j]),
                    ]
                )

    df_all = pd.DataFrame(
        rows,
        columns=[
            "microbe",
            "metabolite",
            "band_index",
            "center_days",
            "flo",
            "fhi",
            "assoc",
            "delay_phase_days",
            "delay_days",
            "period_days",
            "phase_r",
            "kpts",
            "n_segments",
            "fbar",
            "xcorr_peak",
            "p_shift",
            "p_phase",
            "p_cons",
        ],
    )

    df_all["q_value"] = bh_fdr(df_all["p_cons"].to_numpy(float))
    df_all = df_all.sort_values(["q_value", "p_cons", "assoc"], ascending=[True, True, False])

    df_best = df_all.loc[df_all.groupby(["microbe", "metabolite"])["q_value"].idxmin()].copy()
    df_best = df_best.sort_values(["q_value", "p_cons", "assoc"], ascending=[True, True, False])

    df_edges = df_best[
        (df_best["q_value"] <= cfg.q_threshold)
        & (df_best["phase_r"] >= cfg.phase_r_min)
        & (np.abs(df_best["xcorr_peak"]) >= cfg.xcorr_min_peak)
    ].copy()

    df_edges["direction"] = np.where(
        df_edges["delay_days"] > 0,
        "microbe_leads",
        np.where(df_edges["delay_days"] < 0, "metabolite_leads", "undirected"),
    )

    df_all.to_csv(outdir / "all_results.csv", index=False)
    df_best.to_csv(outdir / "best_per_pair.csv", index=False)
    df_edges.to_csv(outdir / f"edges_q<={cfg.q_threshold:.2f}.csv", index=False)

    return df_all, df_best, df_edges


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase_delay_network.py")
    p.add_argument("--microbe-csv", required=True)
    p.add_argument("--metabolite-csv", required=True)
    p.add_argument("--out-dir", default="results/lag_network")
    p.add_argument("--time-col", default="time")

    p.add_argument("--time-min", type=float, default=None)
    p.add_argument("--time-max", type=float, default=None)

    p.add_argument("--min-days", type=int, default=8)
    p.add_argument("--no-clr-microbe", dest="apply_clr_microbe", action="store_false")
    p.add_argument("--microbe-pseudocount", type=float, default=1e-6)
    p.add_argument("--metabolite-pseudocount", type=float, default=1e-6)

    p.add_argument("--band-periods", type=str, default="7:0.5", help="comma-separated 'center_days:width_frac', e.g. '7:0.5,14:0.25'")
    p.add_argument("--nperseg", type=int, default=8)
    p.add_argument("--overlap", type=float, default=0.5)

    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--phase-r-min", type=float, default=0.30)
    p.add_argument("--xcorr-min-peak", type=float, default=0.15)
    p.add_argument("--max-lag-cap", type=int, default=10)

    p.add_argument("--q-threshold", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=12345)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = NetworkConfig(
        min_days=int(args.min_days),
        apply_clr_microbe=bool(getattr(args, "apply_clr_microbe", True)),
        microbe_pseudocount=float(args.microbe_pseudocount),
        metabolite_pseudocount=float(args.metabolite_pseudocount),
        band_periods=parse_band_periods(str(args.band_periods)),
        nperseg=int(args.nperseg),
        overlap=float(args.overlap),
        n_perm=int(args.n_perm),
        phase_r_min=float(args.phase_r_min),
        xcorr_min_peak=float(args.xcorr_min_peak),
        max_lag_cap=int(args.max_lag_cap),
        seed=int(args.seed),
        q_threshold=float(args.q_threshold),
        time_min=(float(args.time_min) if args.time_min is not None else None),
        time_max=(float(args.time_max) if args.time_max is not None else None),
    )

    df_all, df_best, df_edges = compute_delay_network(
        microbe_csv=args.microbe_csv,
        metabolite_csv=args.metabolite_csv,
        out_dir=args.out_dir,
        time_col=args.time_col,
        cfg=cfg,
    )

    print("saved:", str(Path(args.out_dir) / "all_results.csv"))
    print("saved:", str(Path(args.out_dir) / "best_per_pair.csv"))
    print("saved:", str(Path(args.out_dir) / f"edges_q<={cfg.q_threshold:.2f}.csv"))
    print("all rows:", len(df_all), "best pairs:", len(df_best), "edges:", len(df_edges))


if __name__ == "__main__":
    main()
