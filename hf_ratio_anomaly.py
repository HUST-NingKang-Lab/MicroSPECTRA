from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12


@dataclass(frozen=True)
class HFAnomalyConfig:
    window_sizes: Tuple[int, ...] = (16, 24, 32)
    hf_cutoffs: Tuple[float, ...] = (0.10, 0.20, 0.30)
    step: int = 1
    zscore: str = "robust"  # "standard" | "robust"
    writeback: str = "spread"  # "spread" | "center"
    fuse: str = "max"  # "max" | "mean"
    postproc_roll_max: int = 0
    clip_neg_to_zero: bool = True
    resample: bool = True
    target_dt: Optional[float] = None
    resample_method: str = "linear"


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_event_windows(window_str: object) -> List[Tuple[float, float]]:
    if window_str is None or (isinstance(window_str, float) and math.isnan(window_str)):
        return []
    s = str(window_str).strip()
    if not s:
        return []
    s = s.replace("，", ",").replace("；", ";")
    pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", s)
    return [(float(a), float(b)) for a, b in pairs]


def make_binary_labels(t: np.ndarray, windows: Sequence[Tuple[float, float]]) -> np.ndarray:
    y = np.zeros(len(t), dtype=int)
    for a, b in windows:
        lo, hi = (a, b) if a <= b else (b, a)
        y |= ((t >= lo) & (t <= hi)).astype(int)
    return y


def safe_auc_pr(y: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if len(y) < 2 or y.sum() == 0 or y.sum() == len(y):
        return float("nan"), float("nan")
    try:
        roc = float(roc_auc_score(y, s))
    except Exception:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y, s))
    except Exception:
        pr = float("nan")
    return roc, pr


def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + EPS
    return (x - med) / (1.4826 * mad)


def standard_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x)) + EPS
    return (x - mu) / sd


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < EPS:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / (mx - mn + EPS)


def resample_uniform(
    t: np.ndarray,
    x: np.ndarray,
    *,
    target_dt: Optional[float],
    method: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]
    if len(t) < 2:
        return t, x, 1.0

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    dt0 = float(np.median(diffs)) if len(diffs) else 1.0
    dt = float(target_dt) if target_dt is not None else dt0
    dt = max(dt, EPS)

    t_u = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    if method != "linear":
        raise ValueError(f"Unsupported resample_method={method}")
    x_u = np.interp(t_u, t, x)
    fs = 1.0 / dt
    return t_u, x_u, fs


def hf_ratio_sliding_windows(
    x: np.ndarray,
    fs: float,
    *,
    window_size: int,
    step: int,
    hf_cutoff_nyquist_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n < window_size or window_size < 4:
        return np.array([], dtype=float), np.array([], dtype=int)

    hann = np.hanning(window_size)
    ratios: List[float] = []
    centers: List[int] = []

    cutoff_hz = (fs / 2.0) * float(hf_cutoff_nyquist_frac)
    for start in range(0, n - window_size + 1, step):
        w = x[start : start + window_size]
        if not np.all(np.isfinite(w)):
            continue
        W = np.fft.rfft(w * hann)
        spec = np.abs(W) ** 2
        freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)
        hf_energy = float(np.sum(spec[freqs >= cutoff_hz]))
        total_energy = float(np.sum(spec)) + EPS
        ratios.append(hf_energy / total_energy)
        centers.append(start + window_size // 2)

    return np.asarray(ratios, dtype=float), np.asarray(centers, dtype=int)


def writeback_window_scores(
    n: int,
    z: np.ndarray,
    centers: np.ndarray,
    *,
    window_size: int,
    mode: str,
) -> np.ndarray:
    if len(z) == 0 or len(centers) == 0:
        return np.zeros(n, dtype=float)

    if mode == "center":
        out = np.full(n, np.nan, dtype=float)
        out[centers] = z
        idx = np.arange(n)
        known = np.isfinite(out)
        if known.sum() < 2:
            return np.zeros(n, dtype=float)
        return np.interp(idx, idx[known], out[known]).astype(float)

    if mode == "spread":
        out = np.full(n, -np.inf, dtype=float)
        half = window_size // 2
        for zi, ci in zip(z, centers):
            a = max(0, int(ci - half))
            b = min(n, a + window_size)
            out[a:b] = np.maximum(out[a:b], float(zi))
        finite = np.isfinite(out)
        if not finite.any():
            return np.zeros(n, dtype=float)
        mn = float(np.min(out[finite]))
        out[~finite] = mn
        return out.astype(float)

    raise ValueError("writeback must be 'spread' or 'center'")


def rolling_max(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.empty_like(x)
    from collections import deque

    dq: deque[Tuple[int, float]] = deque()
    for i in range(n):
        v = float(x[i])
        while dq and dq[-1][1] <= v:
            dq.pop()
        dq.append((i, v))
        while dq and dq[0][0] <= i - k:
            dq.popleft()
        out[i] = dq[0][1]
    return out


def hf_anomaly_score(
    t: np.ndarray,
    x: np.ndarray,
    cfg: HFAnomalyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(t) != len(x):
        raise ValueError("time and value length mismatch")

    x0 = normalize_01(x)
    if np.max(x0) - np.min(x0) < EPS or len(x0) < min(cfg.window_sizes, default=4):
        return np.zeros_like(x0, dtype=float), t, x0

    if cfg.resample:
        t_u, x_u_raw, fs = resample_uniform(
            t, x0, target_dt=cfg.target_dt, method=cfg.resample_method
        )
    else:
        order = np.argsort(t)
        t_u = t[order]
        x_u_raw = x0[order]
        diffs = np.diff(t_u)
        dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        fs = 1.0 / max(dt, EPS)

    x_u = normalize_01(x_u_raw)

    all_scores: List[np.ndarray] = []
    for ws in cfg.window_sizes:
        for hf in cfg.hf_cutoffs:
            ratios, centers = hf_ratio_sliding_windows(
                x_u,
                fs,
                window_size=int(ws),
                step=int(cfg.step),
                hf_cutoff_nyquist_frac=float(hf),
            )
            if cfg.zscore == "robust":
                z = robust_zscore(ratios) if len(ratios) else np.array([], dtype=float)
            else:
                z = standard_zscore(ratios) if len(ratios) else np.array([], dtype=float)
            s = writeback_window_scores(
                len(x_u), z, centers, window_size=int(ws), mode=str(cfg.writeback)
            )
            all_scores.append(s)

    if not all_scores:
        s_u = np.zeros_like(x_u, dtype=float)
    else:
        stack = np.vstack(all_scores)
        s_u = np.max(stack, axis=0) if cfg.fuse == "max" else np.mean(stack, axis=0)

    if cfg.postproc_roll_max and cfg.postproc_roll_max > 1:
        s_u = rolling_max(s_u, int(cfg.postproc_roll_max))

    if cfg.clip_neg_to_zero:
        s_u = np.maximum(s_u, 0.0)

    s_back = np.interp(t, t_u, s_u).astype(float)
    return s_back, t, x0


def run_single_table(
    input_csv: str,
    *,
    time_col: Optional[str],
    window: Optional[str],
    out_dir: str,
    cfg: HFAnomalyConfig,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    df = pd.read_csv(input_csv)

    if time_col is None:
        time_col = df.columns[0]
    if time_col not in df.columns:
        raise ValueError(f"time_col not found: {time_col}")

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    windows = parse_event_windows(window)
    y = make_binary_labels(t, windows) if windows else None

    feature_cols = [c for c in df.columns if c != time_col]
    rows: List[Dict[str, object]] = []

    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        score, _, _ = hf_anomaly_score(t, x, cfg)
        rec: Dict[str, object] = {"input": str(Path(input_csv).name), "feature": str(col)}
        if y is not None:
            roc, pr = safe_auc_pr(y, score)
            rec["roc_auc"] = roc
            rec["pr_auc"] = pr
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(Path(out_dir) / "hf_anomaly_single_summary.csv", index=False)
    return out


def run_from_meta(
    meta_csv: str,
    *,
    base_dir: str,
    out_dir: str,
    time_col: str = "time",
    name_col: str = "name",
    window_col: str = "window",
    cfg: HFAnomalyConfig = HFAnomalyConfig(),
) -> pd.DataFrame:
    ensure_dir(out_dir)
    meta = pd.read_csv(meta_csv)
    for c in (name_col, window_col):
        if c not in meta.columns:
            raise ValueError(f"meta missing column: {c}")

    rows: List[Dict[str, object]] = []
    per_table_dir = Path(out_dir) / "per_table"
    ensure_dir(per_table_dir)

    for _, r in meta.iterrows():
        name = str(r[name_col]).strip()
        csv_path = Path(base_dir) / f"{name}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        if time_col not in df.columns:
            continue

        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
        windows = parse_event_windows(r[window_col])
        y = make_binary_labels(t, windows)

        feature_cols = [c for c in df.columns if c != time_col]
        table_rows: List[Dict[str, object]] = []

        for col in feature_cols:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            score, _, _ = hf_anomaly_score(t, x, cfg)
            roc, pr = safe_auc_pr(y, score)

            rec: Dict[str, object] = {
                "table": name,
                "feature": str(col),
                "window": str(r[window_col]),
                "roc_auc": roc,
                "pr_auc": pr,
            }
            for extra in ("taxa", "event", "doi"):
                if extra in meta.columns:
                    rec[extra] = r.get(extra, "")
            table_rows.append(rec)
            rows.append(rec)

        if table_rows:
            pd.DataFrame(table_rows).to_csv(per_table_dir / f"{name}_hf_anomaly_metrics.csv", index=False)

    out = pd.DataFrame(rows)
    if len(out):
        out.sort_values(["table", "pr_auc"], ascending=[True, False], inplace=True)
    out.to_csv(Path(out_dir) / "hf_anomaly_all_tables_summary.csv", index=False)
    return out


def parse_cfg_json(s: Optional[str]) -> Dict[str, object]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception as e:
        raise ValueError(f"invalid cfg json: {e}") from e


def cfg_from_args(args: argparse.Namespace) -> HFAnomalyConfig:
    override = parse_cfg_json(getattr(args, "cfg_json", None))
    base = HFAnomalyConfig(
        window_sizes=tuple(args.window_sizes),
        hf_cutoffs=tuple(args.hf_cutoffs),
        step=int(args.step),
        zscore=str(args.zscore),
        writeback=str(args.writeback),
        fuse=str(args.fuse),
        postproc_roll_max=int(args.postproc_roll_max),
        clip_neg_to_zero=bool(args.clip_neg_to_zero),
        resample=bool(args.resample),
        target_dt=(float(args.target_dt) if args.target_dt is not None else None),
        resample_method=str(args.resample_method),
    )
    if not override:
        return base

    d = base.__dict__.copy()
    for k, v in override.items():
        if k not in d:
            continue
        if k in ("window_sizes", "hf_cutoffs") and isinstance(v, list):
            d[k] = tuple(v)
        else:
            d[k] = v
    return HFAnomalyConfig(**d)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hf_ratio_anomaly.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_single = sub.add_parser("single", help="run anomaly scoring on a single CSV")
    p_single.add_argument("--input", required=True)
    p_single.add_argument("--time-col", default=None)
    p_single.add_argument("--window", default=None, help="e.g. '70-122' or '70-122;150-159'")
    p_single.add_argument("--out-dir", default="results/anomaly")

    p_meta = sub.add_parser("meta", help="run anomaly scoring for a meta.csv list of tables")
    p_meta.add_argument("--meta-csv", required=True)
    p_meta.add_argument("--base-dir", required=True, help="folder that contains <name>.csv")
    p_meta.add_argument("--out-dir", default="results/anomaly")
    p_meta.add_argument("--time-col", default="time")
    p_meta.add_argument("--name-col", default="name")
    p_meta.add_argument("--window-col", default="window")

    for pp in (p_single, p_meta):
        pp.add_argument("--window-sizes", nargs="+", type=int, default=[16, 24, 32])
        pp.add_argument("--hf-cutoffs", nargs="+", type=float, default=[0.10, 0.20, 0.30])
        pp.add_argument("--step", type=int, default=1)
        pp.add_argument("--zscore", choices=["standard", "robust"], default="robust")
        pp.add_argument("--writeback", choices=["spread", "center"], default="spread")
        pp.add_argument("--fuse", choices=["max", "mean"], default="max")
        pp.add_argument("--postproc-roll-max", type=int, default=0)
        pp.add_argument("--clip-neg-to-zero", action="store_true", default=True)
        pp.add_argument("--no-clip-neg-to-zero", dest="clip_neg_to_zero", action="store_false")
        pp.add_argument("--resample", action="store_true", default=True)
        pp.add_argument("--no-resample", dest="resample", action="store_false")
        pp.add_argument("--target-dt", type=float, default=None)
        pp.add_argument("--resample-method", choices=["linear"], default="linear")
        pp.add_argument("--cfg-json", default=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = cfg_from_args(args)

    if args.cmd == "single":
        df = run_single_table(
            args.input,
            time_col=args.time_col,
            window=args.window,
            out_dir=args.out_dir,
            cfg=cfg,
        )
        cols = [c for c in ["feature", "roc_auc", "pr_auc"] if c in df.columns]
        print(df.sort_values(["pr_auc"], ascending=[False]).head(20)[cols].to_string(index=False))
        return

    if args.cmd == "meta":
        df = run_from_meta(
            args.meta_csv,
            base_dir=args.base_dir,
            out_dir=args.out_dir,
            time_col=args.time_col,
            name_col=args.name_col,
            window_col=args.window_col,
            cfg=cfg,
        )
        cols = [c for c in ["table", "feature", "roc_auc", "pr_auc"] if c in df.columns]
        print(df.sort_values(["pr_auc"], ascending=[False]).head(20)[cols].to_string(index=False))
        return

    raise RuntimeError("unknown command")


if __name__ == "__main__":
    main()
