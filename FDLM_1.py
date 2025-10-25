"""
FDLM_1.py — Frequency-Domain Lifetime (single decay) from PTU files using Braun-style photon lock-in

Pipeline:
  - Input: folder with .ptu files, each acquired at a different modulation frequency
  - For each file:
      * Read timestamps and markers (PicoHarp 300 T2, via reader.py)
      * Build modulation reference edge list from a chosen marker bit (ZERO-BASED: 0..3)
      * Compute per-photon phases θ relative to the 1f reference using Jakob dynamic-fit (braun_lockin_jakob.phases_jakob)
        or a robust fallback estimator if that import fails
      * Convert phases → phasor to get relative modulation A and phase φ
      * Estimate the modulation frequency f from marker edges
  - Fit single-exponential lifetime τ using phase data only (default) or jointly with amplitudes:
      * Phase model: φ(ω) = arctan(ω τ)
      * Amplitude model: m(ω) = 1 / sqrt(1 + (ω τ)^2)
  - Plot φ vs f (and A vs f if --use-amplitude), save CSV of results, print τ

Usage:
  python FDLM_1.py <folder> --marker-bit 1 --photon-channel 1 --N 10 --savefig fdlm_fit.png
  python FDLM_1.py <folder> --marker-bit 1 --photon-channel 1 --N 10 --use-amplitude --savefig fdlm_fit.png

Notes:
  * marker-bit is ZERO-BASED (0..3), matching braun_lockin_jakob.py’s convention.
  * Phase-only fit is robust to square vs sine drive (locking at 1f).
"""

import argparse, math, sys, csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

_has_braun = False
try:
    from .fdlm_helper import phases_jakob as jakob_phases  # type: ignore
    from .fdlm_helper import phasor_from_phases
    _has_braun = True
except Exception:
    _has_braun = False

from reader import read_ptu  # returns: header, times_by_channel{1..4}->ps(int64), markers[(mk:int, ps:int64)], reader_name


@dataclass
class FileResult:
    path: Path
    f_Hz: float
    omega: float
    A: float
    phi: float
    Nphot: int


# -------------------- Helpers --------------------

def extract_marker_edges_ps(markers: List[Tuple[int, int]], bit_zero_based: int) -> np.ndarray:
    """
    Get timestamps (ps) for events where the given marker bit is asserted.
    EXPECTS ZERO-BASED BIT INDEX (0..3), i.e., mask = (1 << bit).
    """
    mask = 1 << bit_zero_based
    times = [t for m, t in markers if (m & mask) != 0]
    return np.asarray(times, dtype=np.int64)


def phases_fallback(chopper_ps: np.ndarray, photon_ps: np.ndarray, N: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback per-photon phases if braun_lockin_jakob is unavailable.
    For each photon, linearly interpolate its phase using the last N reference edges to estimate the local period.
    Returns (theta in [0,2π), valid mask).
    """
    if chopper_ps.size < max(2, N) or photon_ps.size == 0:
        return np.array([]), np.zeros(photon_ps.size, dtype=bool)

    # Index of the last chopper edge before each photon
    idx = np.searchsorted(chopper_ps, photon_ps, side="right") - 1
    valid = idx >= (N - 1)
    if not np.any(valid):
        return np.array([]), valid

    idxv = idx[valid]
    t_ph = photon_ps[valid].astype(np.float64)

    s = idxv - (N - 1)  # start index for the N-edge window
    dt_sum = np.zeros_like(t_ph, dtype=float)
    for j in range(N - 1):
        a = chopper_ps[s + j + 1]
        b = chopper_ps[s + j]
        dt_sum += (a - b).astype(np.float64)
    periods = dt_sum / float(N - 1)

    t0s = chopper_ps[idxv].astype(np.float64)
    frac = ((t_ph - t0s) / periods) % 1.0
    theta = (2.0 * np.pi * frac).astype(np.float64)
    return theta, valid


# def phasor_from_phases(theta: np.ndarray) -> Dict[str, float]:
#     """Compute 1f phasor amplitude A and phase phi from per-photon phases (θ in [0,2π))."""
#     if theta.size == 0:
#         return dict(A=np.nan, phi=np.nan, N=0, sigA=np.nan, sigPhi=np.nan)
#     z = np.exp(1j * theta).sum()
#     N = theta.size
#     A = 2.0 * np.abs(z) / N
#     phi = float(np.angle(z))
#     # Simple shot-noise estimates (order-of-magnitude)
#     sigA = 1.0 / math.sqrt(N)
#     sigPhi = 2.0 * math.pi / math.sqrt(N)
#     return dict(A=float(A), phi=phi, N=int(N), sigA=float(sigA), sigPhi=float(sigPhi))


def estimate_frequency_from_edges(chop_ps: np.ndarray) -> float:
    """Estimate Hz from median edge-to-edge period (robust)."""
    if chop_ps.size < 2:
        return float("nan")
    d = np.diff(chop_ps).astype(np.float64)  # ps
    med = np.median(d)
    if med <= 0:
        return float("nan")
    return 1e12 / med  # ps → Hz


# Models
def phi_model(omega: np.ndarray, tau: float) -> np.ndarray:
    return np.arctan(omega * tau)


def m_model(omega: np.ndarray, tau: float) -> np.ndarray:
    return 1.0 / np.sqrt(1.0 + (omega * tau) ** 2)


# -------------------- Core processing --------------------

def process_file(ptu_path: Path, photon_channel: int, marker_bit_zero_based: int, N_fit: int) -> FileResult:
    header, times_by_channel, markers, reader_name = read_ptu(str(ptu_path))

    # Photons:
    if photon_channel not in times_by_channel:
        raise RuntimeError(
            f"No photons on channel {photon_channel} for {ptu_path.name}. "
            f"Channels present: {sorted(times_by_channel.keys())}."
        )
    photon_ps = times_by_channel[photon_channel]
    if photon_ps.size == 0:
        raise RuntimeError(f"No photons found on channel {photon_channel} in {ptu_path.name}.")

    # Marker-derived reference edges:
    chop_ps = extract_marker_edges_ps(markers, marker_bit_zero_based)
    if chop_ps.size < max(2, N_fit):
        raise RuntimeError(
            f"Not enough marker edges on bit {marker_bit_zero_based} in {ptu_path.name}. Found {chop_ps.size}."
        )

    # Limit photons to reference span (avoid edges outside)
    lo, hi = chop_ps[0], chop_ps[-1]
    photon_ps = photon_ps[(photon_ps >= lo) & (photon_ps <= hi)]
    if photon_ps.size == 0:
        raise RuntimeError(f"No photons within reference span in {ptu_path.name}.")

    # Per-photon phases:
    if _has_braun:
        theta, valid = jakob_phases(chop_ps.astype(np.float64), photon_ps.astype(np.float64), N=N_fit)  # type: ignore
        # jakob_phases returns phases for valid photons already; valid mask not re-used here
    else:
        theta, valid = phases_fallback(chop_ps, photon_ps, N=N_fit)
        theta = theta[valid]

    res = phasor_from_phases(theta)

    f_Hz = estimate_frequency_from_edges(chop_ps)
    return FileResult(path=ptu_path, f_Hz=f_Hz, omega=2 * np.pi * f_Hz, A=res["A"], phi=res["phi"], Nphot=res["N"])


# -------------------- Fitting --------------------

def fit_tau_phase_only(omegas: np.ndarray, phis: np.ndarray) -> float:
    """Least-squares fit of φ = arctan(ω τ)."""
    from scipy.optimize import least_squares
    phis_u = np.unwrap(phis.copy())

    def resid(logtau):
        tau = np.exp(logtau[0])
        return phis_u - np.arctan(omegas * tau)

    # initialize near 1 ns; bounds keep τ positive and numerically sane
    sol = least_squares(resid, x0=[math.log(1e-9)], bounds=([-50.0], [50.0]))
    return float(np.exp(sol.x[0]))


def fit_tau_joint(omegas: np.ndarray, phis: np.ndarray, amps: np.ndarray) -> float:
    """Joint fit using phase and amplitude models with roughly balanced residual scales."""
    from scipy.optimize import least_squares
    phis_u = np.unwrap(phis.copy())

    def resid(logtau):
        tau = np.exp(logtau[0])
        r1 = phis_u - np.arctan(omegas * tau)
        r2 = amps - 1.0 / np.sqrt(1.0 + (omegas * tau) ** 2)
        return np.concatenate([r1 / np.pi, r2])

    sol = least_squares(resid, x0=[math.log(1e-9)], bounds=([-50.0], [50.0]))
    return float(np.exp(sol.x[0]))



def make_plot(results: List[FileResult], tau: float, show_amp: bool, outpath: Optional[Path] = None):
    results = sorted(results, key=lambda r: r.f_Hz)
    f = np.array([r.f_Hz for r in results])
    w = 2 * np.pi * f
    phi = np.array([r.phi for r in results])
    A = np.array([r.A for r in results])

    # Phase plot
    fig = plt.figure(figsize=(7, 5))
    ax1 = plt.gca()
    ax1.scatter(f, np.unwrap(phi), label="phase data", s=40)
    fmin = np.maximum(1.0, np.nanmin(f[f > 0]) * 0.7) if np.any(f > 0) else 1.0
    fmax = np.maximum(2.0, np.nanmax(f) * 1.3) if np.size(f) else 10.0
    ff = np.logspace(np.log10(fmin), np.log10(fmax), 400)
    ax1.plot(ff, np.arctan(2 * np.pi * ff * tau), label=f"fit φ(ω)   τ = {tau*1e9:.3f} ns", lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("modulation frequency f (Hz)")
    ax1.set_ylabel("phase φ (rad)")
    ax1.grid(True, which="both", ls=":")
    ax1.legend(loc="best")

    # Amplitude plot (optional)
    if show_amp:
        ax2 = ax1.twinx()
        ax2.scatter(f, A, marker="x", label="mod depth A (rel.)", s=40)
        ax2.plot(ff, 1.0 / np.sqrt(1.0 + (2 * np.pi * ff * tau) ** 2), lw=2, alpha=0.6, label="fit m(ω)")
        ax2.set_ylabel("relative modulation depth")
        ax2.legend(loc="lower right")

    fig.tight_layout()
    if outpath is not None:
        fig.savefig(outpath, dpi=160)
    return fig



def write_csv(results: List[FileResult], csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "f_Hz", "omega_rad_per_s", "A", "phi_rad", "Nphot"])
        for r in results:
            w.writerow([r.path.name, f"{r.f_Hz:.9g}", f"{r.omega:.9g}", f"{r.A:.6g}", f"{r.phi:.9g}", r.Nphot])



def main():
    ap = argparse.ArgumentParser(description="Single-decay FD lifetime from PTU folder using photon lock-in.")
    ap.add_argument("folder", type=str, help="Folder containing .ptu files (one per modulation frequency).")
    ap.add_argument("--photon-channel", type=int, default=1, help="Photon channel index (default 1).")
    ap.add_argument("--marker-bit", type=int, default=1, help="ZERO-BASED marker bit carrying modulation TTL (0..3).")
    ap.add_argument("--N", type=int, default=10, help="# of prior edges for dynamic period fit (Jakob). 4..20 good.")
    ap.add_argument("--use-amplitude", action="store_true", help="Fit τ jointly using φ and A (by default phase only).")
    ap.add_argument("--savefig", type=str, default=None, help="Path to save plot (e.g., fdlm_fit.png).")
    ap.add_argument("--csv", type=str, default="fdlm_results.csv", help="Output CSV filename.")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")

    ptu_files = sorted(folder.glob("*.ptu"))
    if not ptu_files:
        sys.exit(f"No .ptu files found in {folder}")

    results: List[FileResult] = []
    for p in ptu_files:
        try:
            r = process_file(p, photon_channel=args.photon_channel, marker_bit_zero_based=args.marker_bit, N_fit=args.N)
            results.append(r)
            print(f"{p.name:40s}  f={r.f_Hz:.3f} Hz   A={r.A:.4f}   phi={r.phi:.4f} rad   N={r.Nphot}")
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}", file=sys.stderr)

    if len(results) < 2:
        sys.exit("Need results from ≥2 frequencies to fit τ.")

    # Prepare arrays for fit
    omegas = np.array([r.omega for r in results], dtype=float)
    phis = np.array([r.phi for r in results], dtype=float)
    amps = np.array([r.A for r in results], dtype=float)

    if args.use_amplitude:
        tau = fit_tau_joint(omegas, phis, amps)
    else:
        tau = fit_tau_phase_only(omegas, phis)

    # Plot & save CSV
    outpng = Path(args.savefig) if args.savefig else None
    make_plot(results, tau, show_amp=args.use_amplitude, outpath=outpng)

    csv_path = Path(args.csv)
    write_csv(results, csv_path)
    print(f"\nEstimated lifetime τ = {tau:.6e} s  ({tau*1e9:.3f} ns)")
    if outpng:
        print(f"Saved plot → {outpng}")
    print(f"Saved table → {csv_path}")


if __name__ == "__main__":
    main()

# python FDLM_1.py /path/to/ptu_folder --marker-bit 1 --photon-channel 1 --N 10 --savefig fdlm_fit.png