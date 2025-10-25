# phase_hist_jakob.py
# CLI: PTU → phases via Jakob dynamic linear fit → histogram (0..2π)
import argparse
from typing import Tuple, Optional
import numpy as np

from reader import read_ptu 

def _correct_phase_offset(phase_fractions: np.ndarray) -> np.ndarray:
    """
    rotate phases so that the complex 1f phasor aligns to angle 0.
    Input/Output phases in [0, 1).
    """
    if phase_fractions.size == 0:
        return phase_fractions
    Z = np.exp(1j * 2 * np.pi * phase_fractions).mean()
    # moves the mean phasor to angle 0
    offset = (np.angle(Z) / (2 * np.pi)) % 1.0
    return (phase_fractions - offset) % 1.0

def calculate_phase_jakob(
    chopper_times: np.ndarray, photon_times: np.ndarray, N: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
   
    if chopper_times.size < N or photon_times.size == 0:
        return np.array([], dtype=float), np.zeros(photon_times.size, dtype=bool)

    indices = np.searchsorted(chopper_times, photon_times, side="right")
    valid_mask = indices >= N
    if not valid_mask.any():
        return np.array([], dtype=float), valid_mask

    valid_indices = indices[valid_mask]
    valid_photons = photon_times[valid_mask].astype(float)

    start_indices = valid_indices - N                                  # shape (M,)
    window_indices = start_indices[:, None] + np.arange(N)[None, :]    # (M, N)
    y = chopper_times[window_indices].astype(float)                     # (M, N)

    
    x = np.arange(N, dtype=float)                                      # (N,)
    x_mean = x.mean()
    x_var = x.var()                                                    # scalar
    y_mean = y.mean(axis=1)                                            # (M,)
    cov = ((y - y_mean[:, None]) * (x - x_mean)).sum(axis=1) / N       # (M,)
    slope = cov / x_var                                                # (M,) -> period (ps)
    intercept = y_mean - slope * x_mean                                # (M,)

    t_last = intercept + slope * x[-1]                                 # (M,)
    phase_frac = (valid_photons - t_last) / slope                      # (M,)
    phase_frac = phase_frac % 1.0
    return phase_frac, valid_mask

def main():
    ap = argparse.ArgumentParser(
        description="Compute per-photon phase (Jakob method) and show histogram."
    )
    ap.add_argument("ptu", type=str)
    ap.add_argument("--ch-ph", type=int, default=1, help="photon channel (default 1)")
    ap.add_argument("--marker-bit", type=int, default=2,
                    help="use markers where (mask & (1<<bit)) != 0 (default 2)")
    ap.add_argument("--ref-every-n", type=int, default=1,
                    help="keep every N-th ref edge (use 2 if both edges present)")
    ap.add_argument("--N", type=int, default=10, help="ticks per local linear fit (default 10)")
    ap.add_argument("--bins", type=int, default=90, help="histogram bins (default 90)")
    ap.add_argument("--ascii", action="store_true", help="print ASCII histogram instead of plotting")
    ap.add_argument("--out", type=str, default="", help="save plot to PNG (implies plotting)")
    args = ap.parse_args()

    # --- Read PTU ---
    header, ch, markers, reader = read_ptu(args.ptu) # type: ignore
    t_ph = ch.get(args.ch_ph, np.array([], dtype=np.int64)) # type: ignore
    mask = 1 << args.marker_bit
    t_ref = np.asarray([tps for (mk, tps) in markers if (mk & mask) != 0], dtype=np.int64)
    if t_ref.size == 0:
        print("No reference markers found with that bit. Try another --marker-bit.")
        return
    if args.ref_every_n > 1 and t_ref.size:
        t_ref = t_ref[::args.ref_every_n].copy()

    # Restrict photons to the ref span (avoid extrapolation at the ends)
    lo, hi = t_ref[0], t_ref[-1]
    keep_ph = (t_ph >= lo) & (t_ph <= hi)
    t_ph = t_ph[keep_ph]

    print(f"Reader: {reader} | Photons CH{args.ch_ph}: {t_ph.size:,} | Ref markers(bit={args.marker_bit}): {t_ref.size:,}")
    if t_ph.size == 0 or t_ref.size < args.N:
        print("Not enough data for phase computation (need photons and ≥N ref edges).")
        return

    phase_fractions, valid_mask = calculate_phase_jakob(t_ref, t_ph, N=args.N)
    if phase_fractions.size == 0:
        print("No photons had ≥N prior reference edges. Increase acquisition or lower N.")
        return

    corrected = _correct_phase_offset(phase_fractions)
    theta = (2 * np.pi * corrected).astype(float)

    used = valid_mask.sum()
    print(f"Photons used (valid windows): {used:,} / {t_ph.size:,}")
    med_dt_ps = float(np.median(np.diff(t_ref))) if t_ref.size > 1 else float("nan")
    if np.isfinite(med_dt_ps) and med_dt_ps > 0:
        f_med = 1e6 / med_dt_ps
        print(f"Median ref period ≈ {med_dt_ps*1e-6:.3f} µs → f ≈ {f_med:.3f} Hz")

    if args.ascii and not args.out:
        counts, edges = np.histogram(theta, bins=args.bins, range=(0, 2*np.pi))
        peak = counts.max() if counts.size else 1
        width = 60
        for i, c in enumerate(counts):
            lo = edges[i]; hi = edges[i+1]
            bar = "#" * int(round(width * (c / peak)))
            print(f"[{lo:6.3f},{hi:6.3f})  {c:7d}  {bar}")
        return

    try:
        import matplotlib.pyplot as plt  
        counts, edges = np.histogram(theta, bins=args.bins, range=(0, 2*np.pi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.figure(figsize=(7.5, 4.5))
        plt.bar(centers, counts, width=(2*np.pi/args.bins), align="center")
        plt.xlabel("Phase θ (rad)")
        plt.ylabel("Counts")
        plt.title(f"Phase histogram (Jakob fit) — {args.ptu}\nN={args.N}, bins={args.bins}")
        plt.xlim(0, 2*np.pi)
        plt.tight_layout()
        if args.out:
            plt.savefig(args.out, dpi=150)
            print(f"Saved plot to {args.out}")
        else:
            plt.show()
    except Exception as e:
        print(f"(Plotting unavailable: {e})  Falling back to ASCII.")
        counts, edges = np.histogram(theta, bins=args.bins, range=(0, 2*np.pi))
        peak = counts.max() if counts.size else 1
        width = 60
        for i, c in enumerate(counts):
            lo = edges[i]; hi = edges[i+1]
            bar = "#" * int(round(width * (c / peak)))
            print(f"[{lo:6.3f},{hi:6.3f})  {c:7d}  {bar}")

if __name__ == "__main__":
    main()
# phase_hist_jakob.py data/laserON_modulated_200s.ptu --ch-ph 1 --marker-bit 1 --N 10