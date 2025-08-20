# braun_lockin_jakob.py — PTU → Jakob dynamic-fit phases → phasor A,phi and contrast
import argparse
import numpy as np
from reader import read_ptu  # PH300 T2 reader that returns ps

def phases_jakob(chopper_ps: np.ndarray, photon_ps: np.ndarray, N: int = 10):
    """
    Jakob dynamic linear fit (vectorized): for each photon, fit the last N edges,
    return phases in [0, 2π) and a valid mask (photons that had ≥N prior edges).
    """
    if chopper_ps.size < N or photon_ps.size == 0:
        return np.array([]), np.zeros(photon_ps.size, dtype=bool)
    idx = np.searchsorted(chopper_ps, photon_ps, side="right")
    valid = idx >= N
    if not valid.any():
        return np.array([]), valid

    k = idx[valid]                           # (M,)
    t_ph = photon_ps[valid].astype(float)    # (M,)
    # windows of last N ticks per photon
    start = k - N                            # (M,)
    W = start[:, None] + np.arange(N)[None, :]      # (M,N)
    y = chopper_ps[W].astype(float)                  # (M,N)

    # Fit y ≈ slope*x + intercept for each row; x = 0..N-1 (so last tick x_last=N-1)
    x = np.arange(N, dtype=float)
    xm = x.mean()
    xv = x.var()
    ym = y.mean(axis=1)
    cov = ((y - ym[:, None]) * (x - xm)).sum(axis=1) / N
    slope = cov / xv                                  # period in ps
    intercept = ym - slope * xm
    t_last = intercept + slope * x[-1]

    # fractional phase since last tick
    frac = ((t_ph - t_last) / slope) % 1.0            # [0,1)
    theta = 2 * np.pi * frac                          # [0,2π)
    return theta, valid

def phasor_from_phases(theta: np.ndarray):
    """Phasor at 1f from per-photon phases in [0,2π)."""
    if theta.size == 0:
        raise ValueError("No phases provided.")
    Z = np.exp(1j * theta).sum()
    N = theta.size
    A = 2 * np.abs(Z) / N
    phi = float(np.angle(Z))
    sigA = 1.0 / np.sqrt(N)
    sigPhi = 2 * np.pi / np.sqrt(N)
    return dict(A=float(A), phi=phi, sigA=float(sigA), sigPhi=float(sigPhi), N=int(N))

def estimate_contrast_from_hist(theta: np.ndarray):
    """
    Quick, model-free contrast estimate for a nominal 50% duty gate:
    rotate so mean phasor is at 0, then split at θ=π.
    contrast = (I_hi - I_lo) / (I_hi + I_lo)
    For an ideal on/off gate with 50% duty and I_lo=0 → contrast ≈ 1.
    """
    if theta.size == 0:
        return np.nan, 0, 0
    # rotate by -phi so that ON window is centered around 0
    Z = np.exp(1j * theta).mean()
    phi_mean = np.angle(Z)
    thetac = (theta - phi_mean) % (2 * np.pi)
    on = np.sum(thetac < np.pi)
    off = theta.size - on
    if on + off == 0:
        return np.nan, on, off
    contrast = (on - off) / (on + off)
    return float(contrast), int(on), int(off)

def main():
    ap = argparse.ArgumentParser(description="Braun lock-in via Jakob dynamic-fit phases")
    ap.add_argument("ptu")
    ap.add_argument("--ch-ph", type=int, default=1)
    ap.add_argument("--marker-bit", type=int, default=2, help="marker bit used as reference")
    ap.add_argument("--ref-every-n", type=int, default=1, help="keep every Nth ref edge (usually 1)")
    ap.add_argument("--N", type=int, default=10, help="edges per local fit (Jakob)")
    ap.add_argument("--bins", type=int, default=90)
    ap.add_argument("--ascii", action="store_true")
    args = ap.parse_args()

    # --- read PTU & build reference from markers ---
    marker_times, photon_times = read_ptu(args.ptu)
    t_ph = photon_times
    # bitmask = 1 << args.marker_bit
    t_ref = marker_times
    if t_ref.size < args.N or t_ph.size == 0:
        print("Not enough reference edges or photons.")
        return
    if args.ref_every_n > 1:
        t_ref = t_ref[::args.ref_every_n].copy()

    # limit photons to ref span
    lo, hi = t_ref[0], t_ref[-1]
    t_ph = t_ph[(t_ph >= lo) & (t_ph <= hi)]

    # --- phases via Jakob ---
    theta, valid = phases_jakob(t_ref, t_ph, N=args.N)
    if theta.size == 0:
        print("No photons had ≥N prior edges. Increase acquisition or reduce N.")
        return

    # --- phasor & contrast ---
    res = phasor_from_phases(theta)
    # contrast, on, off = estimate_contrast_from_hist(theta)

    # frequency sanity from refs
    med_dt_ps = float(np.median(np.diff(t_ref)))
    f_med = 1e12 / med_dt_ps if med_dt_ps > 0 else np.nan

    # print(f"Reader: {reader}")
    print(f"Photons used: {res['N']:,} / total {t_ph.size:,}   Ref edges: {t_ref.size:,}   N_fit={args.N}")
    print(f"Ref f ≈ {f_med:.3f} Hz  (from median period)")
    print(f"A (1f phasor): {res['A']:.6f} ± {res['sigA']:.6f}")
    print(f"phi: {res['phi']:.6f} rad  (± {res['sigPhi']:.6f})")
    # If you expect 50% duty on/off, the 1f amplitude for ideal gating is 4/π ≈ 1.273
    # print(f"Square-wave note: if fully gated D=0.5, expect A≈4/π≈{4/np.pi:.4f}")
    # Convert A → “depth” only for 50% duty using the 1f relation
    # depth_from_A = (np.pi / 4.0) * res['A']
    # print(f"Depth from A (assuming D=0.5): {depth_from_A:.4f} (1.0 would be 100% on/off)")
    # print(f"Depth from histogram split:   {contrast:.4f}   (on={on:,}, off={off:,})")

    # Optional histogram
    if args.ascii:
        counts, edges = np.histogram(theta, bins=args.bins, range=(0, 2*np.pi))
        peak = counts.max() if counts.size else 1
        for i, c in enumerate(counts):
            lo_b, hi_b = edges[i], edges[i+1]
            bar = "#" * int(round(60 * (c / peak)))
            print(f"[{lo_b:6.3f},{hi_b:6.3f}) {c:7d} {bar}")
    else:
        try:
            import matplotlib.pyplot as plt
            counts, edges = np.histogram(theta, bins=args.bins, range=(0, 2*np.pi))
            centers = 0.5*(edges[:-1] + edges[1:])
            plt.figure(figsize=(7.2,4.2))
            plt.bar(centers, counts, width=(2*np.pi/args.bins), align="center")
            plt.xlabel("Phase θ (rad)"); plt.ylabel("Counts")
            plt.title(f"{args.ptu} — Jakob phases, N={args.N}")
            plt.xlim(0, 2*np.pi); plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"(Plotting unavailable: {e})")
            pass

if __name__ == "__main__":
    main()
# python3 braun_lockin_jakob.py data/august18/Amplitude-Phase-Flatness/500Hz-1mW-1s.ptu  --N 10
# python3 braun_lockin_jakob.py data/august18/Amplitude-Phase-Flatness/1000Hz-1mW-1s.ptu  --N 10
# python3 braun_lockin_jakob.py data/august18/Amplitude-Phase-Flatness/2000Hz-1mW-1s.ptu  --N 10
# python3 braun_lockin_jakob.py data/august18/Amplitude-Phase-Flatness/5000Hz-1mW-1s.ptu  --N 10
# python3 braun_lockin_jakob.py data/august18/Amplitude-Phase-Flatness/10000Hz-1mW-1s.ptu --N 10
# python3 braun_lockin_jakob.py data/august18/Amplitude-Phase-Flatness/1500Hz-1mW-1s.ptu  --N 10

# here real and imaginary parts are inverted wrt to Liu lockin: z = Q + iI

# python3 braun_lockin_jakob.py data/background_180s.ptu --ch-ph 1 --marker-bit 1 --N 10

# this one uses a local frequency for each photon. that's the better mehtod -> returns 0.84 (expected ~1)

# PS F:\Projects\Work2025Photonics\Braun> python liu_lockin.py data/laserON_modulated_200s.ptu --ch-ph 1 --marker-bit 2 --N 10 --duty
#  0.5
# File: data/laserON_modulated_200s.ptu
# Photon ch=1, marker-bit=2, N=10, duty=0.5
# RESULT — Liu I/Q lock-in photon counting:
#   I (signed)       : -20652
#   Q (signed)       : 6
#   S = |I| + |Q|    : 20658   (photon-number output)
#   ON_I/OFF_I       : 5337 / 25989
#   ON_Q/OFF_Q       : 15666 / 15660
#   Total photons    : 31326
# (.venv) PS F:\Projects\Work2025Photonics\Braun> python braun_lockin_jakob.py data/laserON_modulated_200s.ptu --ch-ph 1 --marker-bit 1 --N 10
# Reader: PicoHarp 300 T2
# Photons used: 31,326 / total 31,326   Ref edges: 2,001,297   N_fit=10
# Ref f ≈ 10002.631 Hz  (from median period)
# A (1f phasor): 0.844219 ± 0.005650
# phi: -1.567129 rad  (± 0.035500)

# phases match with z = Q + iI. Nice.
# TODO: assess shot noise limit; A = 100%/sqrt(N), sigPhi = 360/sqrt(N)