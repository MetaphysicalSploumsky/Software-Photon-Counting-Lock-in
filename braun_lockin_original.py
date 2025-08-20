#!/usr/bin/env python3
"""
Braun & Libchaber photon-counting lock-in (Optics Letters, 2002):
Oversample -> bin -> project onto sine/cos -> A (modulation depth), phi (phase).

Inputs
------
- PTU (T2) file with:
    CH 0 : marker/sync edges at f_ref (rising edges, one per period)
    CH 1 : photon timestamps
- Uses reader.read_ptu() -> (marker_times_ps, photon_times_ps)

Method (exactly as in the paper)
--------------------------------
1) Pick oversampling factor M (default 50), so f_sample = M * f_ref.
2) Bin photon arrivals N_raw(t_n) on a uniform grid at f_sample.
3) Algorithmic PLL: linear regression of marker times vs. integer edge index
   gives the period T = 1/f_ref and intercept t0 (reference phase).
4) Build references on the same grid:
       R_x(t_n) = sin(2π (t_n - t0) / T),   R_y(t_n) = cos(2π (t_n - t0) / T)
   (zero-mean by construction)
5) Subtract DC: N(t_n) = N_raw(t_n) - ⟨N_raw⟩.
6) Project (least-squares) onto the references:
       X̂ = Σ N R_x / Σ R_x^2,   Ŷ = Σ N R_y / Σ R_y^2
   The complex small signal is S = X̂ + i Ŷ.
7) Relative amplitude and phase:
       I_dc = ⟨N_raw⟩
       A    = |S| / I_dc          (dimensionless modulation depth)
       φ    = atan2(Ŷ, X̂)        (radians)

Outputs
-------
- f_ref (Hz), f_sample (Hz), M
- I_dc (counts/bin), bins_per_period (=M), total_bins, total_photons
- A (mod depth), phi_rad, phi_deg

Usage
-----
python3 braun_lockin_original.py data/august18/Amplitude-Phase-Flatness/500Hz-1mW-1s.ptu --M 50 --trim-ends 1 [--csv]

Notes
-----
- By default we analyze the span from the first to last marker.
- --trim-ends removes that many periods from each end before binning
  (useful if start/stop edges are messy).
"""

from __future__ import annotations
import argparse
import numpy as np
from reader import read_ptu


def fit_period_from_markers(marker_ps: np.ndarray) -> tuple[float, float]:
    """
    Algorithmic PLL via linear regression of t_z vs. z (edge index).
    Returns (T_sec, t0_sec) where T=period=1/f_ref and t0 is the intercept time at z=0.
    """
    if marker_ps.size < 2:
        raise ValueError("Not enough marker edges to estimate period")
    t = marker_ps.astype(np.float64) * 1e-12  # ps -> s
    z = np.arange(t.size, dtype=np.float64)
    # Fit t ≈ a*z + b  (a = T, b = t0)
    A = np.vstack([z, np.ones_like(z)]).T
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    T = float(a)
    t0 = float(b)
    if T <= 0:
        raise ValueError("Non-positive period estimated from markers")
    return T, t0


def make_sampling_grid(marker_ps: np.ndarray, T: float, M: int, trim_ends: int) -> np.ndarray:
    """
    Build bin edges at fsample = M/T across complete periods between markers,
    optionally trimming 'trim_ends' periods at start and end.
    Returns bin edges in seconds (monotonic).
    """
    t_mark = marker_ps.astype(np.float64) * 1e-12
    # Keep only a central slice bounded by full periods
    start_idx = trim_ends
    stop_idx  = t_mark.size - trim_ends
    if stop_idx - start_idx < 2:
        raise ValueError("After trimming, not enough periods remain")
    t_start = t_mark[start_idx]
    t_stop  = t_mark[stop_idx - 1]  # last full period start we keep; we'll stop at its next edge
    # extend to the next marker after t_stop if available
    if stop_idx < t_mark.size:
        t_stop = t_mark[stop_idx]
    fs = M / T
    dt = 1.0 / fs
    # Bin edges from t_start to t_stop (right-open)
    n_edges = int(np.floor((t_stop - t_start) / dt)) + 1
    edges = t_start + np.arange(n_edges, dtype=np.float64) * dt
    return edges


def bin_photons(photon_ps: np.ndarray, edges_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Histogram photon times [ps] into bins defined by 'edges_s' [s].
    Returns (counts_per_bin, bin_centers_s).
    """
    if photon_ps.size == 0:
        return np.zeros(edges_s.size - 1, dtype=np.int64), 0.5 * (edges_s[:-1] + edges_s[1:])
    t = photon_ps.astype(np.float64) * 1e-12
    counts, _ = np.histogram(t, bins=edges_s)
    centers = 0.5 * (edges_s[:-1] + edges_s[1:])
    return counts.astype(np.int64), centers


def build_references(t_s: np.ndarray, T: float, t0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Sinusoidal and cosinusoidal references on the same grid as the counts.
    """
    phase = 2.0 * np.pi * (t_s - t0) / T
    Rx = np.sin(phase)
    Ry = np.cos(phase)
    return Rx, Ry


def project_small_signal(N_raw: np.ndarray, Rx: np.ndarray, Ry: np.ndarray) -> tuple[float, float, float]:
    """
    Mean-subtract counts and project onto Rx, Ry:
      X̂ = Σ N R_x / Σ R_x^2     Ŷ = Σ N R_y / Σ R_y^2
      A  = |X̂ + i Ŷ| / I_dc,  φ = atan2(Ŷ, X̂)
    Returns (A, phi_rad, I_dc).
    """
    N_raw = N_raw.astype(np.float64)
    I_dc = float(N_raw.mean())
    N = N_raw - I_dc
    denom_x = float(np.dot(Rx, Rx))
    denom_y = float(np.dot(Ry, Ry))
    if denom_x == 0 or denom_y == 0:
        raise ValueError("Zero reference power in Rx/Ry")
    Xhat = float(np.dot(N, Rx) / denom_x)
    Yhat = float(np.dot(N, Ry) / denom_y)
    A = float(np.hypot(Xhat, Yhat) / (I_dc if I_dc > 0 else np.nan))
    phi = float(np.arctan2(Yhat, Xhat))
    return A, phi, I_dc


def main():
    ap = argparse.ArgumentParser(description="Braun photon-counting lock-in (A, phi) via oversample/bin/project")
    ap.add_argument("ptu", help="Path to PTU (T2) file")
    ap.add_argument("--M", type=int, default=50, help="Oversampling factor (bins per period), default 50")
    ap.add_argument("--trim-ends", type=int, default=1, help="Trim this many periods at each end (default 1)")
    ap.add_argument("--csv", action="store_true", help="Emit a CSV header+line for easy logging")
    args = ap.parse_args()

    if args.M < 8:
        raise SystemExit("Choose M >= 8 (Braun used ~50).")

    # 1) Read timestamps (picoseconds) from your new reader
    marker_ps, photon_ps = read_ptu(args.ptu)
    if marker_ps.size < 3:
        raise SystemExit("Not enough marker edges in CH0 to estimate f_ref robustly.")
    if photon_ps.size == 0:
        raise SystemExit("No photons found on CH1.")

    # 2) PLL: get T and t0 from marker edges
    T, t0 = fit_period_from_markers(marker_ps)
    f_ref = 1.0 / T

    # 3) Sampling grid at fsample = M * f_ref, inside trimmed marker span
    edges_s = make_sampling_grid(marker_ps, T, args.M, args.trim_ends)
    centers_s = 0.5 * (edges_s[:-1] + edges_s[1:])
    f_sample = args.M * f_ref

    # 4) Bin photons on the grid
    N_raw, t_s = bin_photons(photon_ps, edges_s)

    # 5) Build references on the SAME grid
    Rx, Ry = build_references(t_s, T, t0)

    # 6) Project to get A and phi
    A, phi, I_dc = project_small_signal(N_raw, Rx, Ry)

    # Some useful tallies
    total_bins = N_raw.size
    total_photons = int(N_raw.sum())
    bins_per_period = args.M

    if args.csv:
        # CSV header + single line
        print("file,f_ref_Hz,f_sample_Hz,M,bins,total_photons,I_dc_counts_per_bin,A,phi_rad,phi_deg")
        print(f"{args.ptu},{f_ref:.9f},{f_sample:.9f},{args.M},{total_bins},{total_photons},{I_dc:.6f},{A:.8f},{phi:.9f},{np.degrees(phi):.6f}")
    else:
        print(f"File: {args.ptu}")
        print("RESULT — Braun photon-counting lock-in:")
        print(f"  f_ref            : {f_ref:.6f} Hz")
        print(f"  f_sample         : {f_sample:.6f} Hz   (M = {args.M} bins/period)")
        print(f"  bins / photons   : {total_bins} / {total_photons}")
        print(f"  I_dc (mean/bin)  : {I_dc:.6f} counts/bin")
        print(f"  A (mod depth)    : {A:.8f} (dimensionless)")
        print(f"  phi              : {phi:.9f} rad  ({np.degrees(phi):.6f} deg)")
        print(f"  bins_per_period  : {bins_per_period}")
        print(f"  trimmed periods  : {args.trim_ends} at start and end")

if __name__ == "__main__":
    main()

