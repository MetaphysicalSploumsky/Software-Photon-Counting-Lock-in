# messed up my refactoring so now i need this to do fdlm
import argparse
import numpy as np
from reader import read_ptu 

def phases_jakob(chopper_ps: np.ndarray, photon_ps: np.ndarray, N: int = 10):
   
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

    theta, valid = phases_jakob(t_ref, t_ph, N=args.N)
    if theta.size == 0:
        print("No photons had ≥N prior edges. Increase acquisition or reduce N.")
        return

    res = phasor_from_phases(theta)
    # contrast, on, off = estimate_contrast_from_hist(theta)

    med_dt_ps = float(np.median(np.diff(t_ref)))
    f_med = 1e12 / med_dt_ps if med_dt_ps > 0 else np.nan

    print(f"Photons used: {res['N']:,} / total {t_ph.size:,}   Ref edges: {t_ref.size:,}   N_fit={args.N}")
    print(f"Ref f ≈ {f_med:.3f} Hz  (from median period)")
    print(f"A (1f phasor): {res['A']:.6f} ± {res['sigA']:.6f}")
    print(f"phi: {res['phi']:.6f} rad  (± {res['sigPhi']:.6f})")
    

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
