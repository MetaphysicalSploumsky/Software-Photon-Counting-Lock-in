#!/usr/bin/env python3
"""
Braun & Libchaber photon-counting lock-in (Optics Letters, 2002)

Inputs
------
- PTU (T2) file with:
    CH 0 : marker/sync edges at f_ref (rising edges, one per period)
    CH 1 : photon timestamps
- Uses reader.read_ptu() -> (marker_times_ps, photon_times_ps)

Outputs
-------
- f_ref (Hz), f_sample (Hz), M
- I_dc (counts/bin), bins_per_period (=M), total_bins, total_photons
- A (mod depth), phi_rad, phi_deg

Usage
-----
python3 braun_lockin.py <filename.ptu> --M <int> --trim-ends <int> [--csv]
python3 braun_lockin.py <folder of PTU files> --M <int> --trim-ends <int> [--csv]

Notes
-----
- By default we analyze the span from the first to last marker
- --trim-ends removes that many periods from each end before binning
"""

from __future__ import annotations
import argparse
import numpy as np
import pathlib
import csv
from reader import read_ptu


def fit_period_from_markers(marker_ps: np.ndarray) -> tuple[float, float]:
    if marker_ps.size < 2:
        raise ValueError("Not enough marker edges to estimate period")
    t = marker_ps.astype(np.float64) * 1e-12
    z = np.arange(t.size, dtype=np.float64)
    A = np.vstack([z, np.ones_like(z)]).T
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    T, t0 = float(a), float(b)
    if T <= 0:
        raise ValueError("Non-positive period estimated from markers")
    return T, t0


def make_sampling_grid(marker_ps: np.ndarray, T: float, M: int, trim_ends: int) -> np.ndarray:
    t_mark = marker_ps.astype(np.float64) * 1e-12
    start_idx = trim_ends
    stop_idx  = t_mark.size - trim_ends
    if stop_idx - start_idx < 2:
        raise ValueError("After trimming, not enough periods remain")
    t_start = t_mark[start_idx]
    t_stop  = t_mark[stop_idx] if stop_idx < t_mark.size else t_mark[stop_idx - 1]
    fs = M / T
    dt = 1.0 / fs
    n_edges = int(np.floor((t_stop - t_start) / dt)) + 1
    return t_start + np.arange(n_edges, dtype=np.float64) * dt


def bin_photons(photon_ps: np.ndarray, edges_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if photon_ps.size == 0:
        return np.zeros(edges_s.size - 1, dtype=np.int64), 0.5 * (edges_s[:-1] + edges_s[1:])
    t = photon_ps.astype(np.float64) * 1e-12
    counts, _ = np.histogram(t, bins=edges_s)
    centers = 0.5 * (edges_s[:-1] + edges_s[1:])
    return counts.astype(np.int64), centers


def build_references(t_s: np.ndarray, T: float, t0: float) -> tuple[np.ndarray, np.ndarray]:
    phase = 2.0 * np.pi * (t_s - t0) / T
    return np.sin(phase), np.cos(phase)


def project_small_signal(N_raw: np.ndarray, Rx: np.ndarray, Ry: np.ndarray) -> tuple[float, float, float]:
    N_raw = N_raw.astype(np.float64)
    I_dc = float(N_raw.mean())
    N = N_raw - I_dc
    denom_x, denom_y = float(np.dot(Rx, Rx)), float(np.dot(Ry, Ry))
    if denom_x == 0 or denom_y == 0:
        raise ValueError("Zero reference power in Rx/Ry")
    Xhat = float(np.dot(N, Rx) / denom_x)
    Yhat = float(np.dot(N, Ry) / denom_y)
    A = float(np.hypot(Xhat, Yhat) / (I_dc if I_dc > 0 else np.nan))
    phi = float(np.arctan2(Yhat, Xhat))
    return A, phi, I_dc


def analyze_file(ptu_file: pathlib.Path, M: int, trim_ends: int) -> dict:
    marker_ps, photon_ps = read_ptu(str(ptu_file)) # modify the reader if markers/photons are through different channels
    if marker_ps.size < 3:
        raise ValueError(f"{ptu_file}: Not enough marker edges.")
    if photon_ps.size == 0:
        raise ValueError(f"{ptu_file}: No photons found.")

    T, t0 = fit_period_from_markers(marker_ps)
    f_ref = 1.0 / T
    edges_s = make_sampling_grid(marker_ps, T, M, trim_ends)
    f_sample = M * f_ref

    N_raw, t_s = bin_photons(photon_ps, edges_s)
    Rx, Ry = build_references(t_s, T, t0)
    A, phi, I_dc = project_small_signal(N_raw, Rx, Ry)

    return {
        "file": str(ptu_file),
        "f_ref_Hz": f_ref,
        "f_sample_Hz": f_sample,
        "M": M,
        "bins": N_raw.size,
        "total_photons": int(N_raw.sum()),
        "I_dc_counts_per_bin": I_dc,
        "A": A,
        "phi_rad": phi,
        "phi_deg": np.degrees(phi),
        "bins_per_period": M,
    }


def main():
    ap = argparse.ArgumentParser(description="Braun photon-counting lock-in (A, phi) on multiple PTU files")
    ap.add_argument("path", help="Path to PTU file or folder containing PTU files")
    ap.add_argument("--M", type=int, default=50, help="Oversampling factor (bins per period), default 50")
    ap.add_argument("--trim-ends", type=int, default=1, help="Trim this many periods at each end (default 1)")
    ap.add_argument("--csv", action="store_true", help="Save results to CSV")
    args = ap.parse_args()

    path = pathlib.Path(args.path)
    files = [path] if path.is_file() else sorted(path.glob("*.ptu"))

    if not files:
        raise SystemExit("No PTU files found.")

    results = []
    for f in files:
        try:
            res = analyze_file(f, args.M, args.trim_ends)
            results.append(res)
            if not args.csv:
                print(f"File: {res['file']}")
                print("RESULT — Braun photon-counting lock-in:")
                print(f"  f_ref            : {res['f_ref_Hz']:.6f} Hz")
                print(f"  f_sample         : {res['f_sample_Hz']:.6f} Hz   (M = {res['M']} bins/period)")
                print(f"  bins / photons   : {res['bins']} / {res['total_photons']}")
                print(f"  I_dc (mean/bin)  : {res['I_dc_counts_per_bin']:.6f} counts/bin")
                print(f"  A (mod depth)    : {res['A']:.8f}")
                print(f"  phi              : {res['phi_rad']:.9f} rad  ({res['phi_deg']:.6f} deg)")
                print(f"  bins_per_period  : {res['bins_per_period']}")
                print(f"  trimmed periods  : {args.trim_ends} at start and end\n")
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if args.csv and results:
        csv_file = path.with_suffix(".csv") if path.is_file() else path / "results.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV written to {csv_file}")


if __name__ == "__main__":
    main()
