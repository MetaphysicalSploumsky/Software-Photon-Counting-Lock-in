"""
compare_SNR_statistical.py
--------------------------
Compute the *statistical* SNR (mean / stddev across time windows) for a Liu-style
I/Q photon-counting lock-in, on a single PTU file.

We cut the acquisition into N equal contiguous windows (or fixed-length windows),
run the same Liu I/Q demodulation in each window, collect a scalar metric per
window, and report:
    SNR_stat = |mean(metric_k)| / std(metric_k)

Supported per-window metrics (choose with --metric):
  - I   : signed in-phase channel (ON minus OFF for the in-phase square)
  - IQ  : magnitude sqrt(I^2 + Q^2)  (phase-invariant L2)
  - S   : |I| + |Q|                   (Liu L1 “photon number”)
  - XPHI: projection onto a global phase phi_hat computed from the *whole* run,
          i.e., X_k = I_k*cos(phi_hat) + Q_k*sin(phi_hat)

Usage examples:
  python compare_SNR_statistical.py run.ptu --ch-ph 1 --marker-bit 2 --N 10 --duty 0.5 --bins 200
  python compare_SNR_statistical.py run.ptu --bin-sec 1 --metric XPHI --csv

Notes:
- Windows with zero photons or insufficient marker edges ( < N ) are skipped.
- The statistical SNR describes stability over time and includes any slow drift.
- For pure shot noise with stable conditions, SNR_stat increases like sqrt(T).
"""
from __future__ import annotations
import argparse, math, sys, statistics
from typing import Dict, Any, List, Tuple
import numpy as np

from reader import read_ptu  


# ----------------- helpers (Liu I/Q demod) -----------------
def select_marker_times(markers, bit:int) -> np.ndarray:
    mask = 1 << (bit-1)
    ts = [tt for mk, tt in markers if (mk & mask) != 0]
    return np.asarray(ts, dtype=np.int64)

def local_phase_fractions(photon_times_ps: np.ndarray,
                          ref_times_ps: np.ndarray,
                          N:int=10) -> np.ndarray:
    """
    For each photon time t, fit the last N reference edges r[k] with n = a*r + b (least squares).
    Period T = 1/a. Phase = ((t - r_last) / T) mod 1 in [0,1).
    """
    if len(ref_times_ps) < max(N, 2):
        raise ValueError("Not enough reference edges for local phase estimation")
    phases = np.empty(photon_times_ps.shape[0], dtype=np.float64)
    r = ref_times_ps
    j = N-1
    n_idx = np.arange(N, dtype=np.float64)
    n_mean = float(n_idx.mean())
    for i, t in enumerate(photon_times_ps):
        while j+1 < len(r) and r[j+1] <= t:
            j += 1
        j_use = max(j, N-1)
        j0 = j_use-(N-1)
        window = r[j0:j_use+1]
        t_mean = float(window.mean())
        t_center = window - t_mean
        n_center = n_idx - n_mean
        denom = float((t_center**2).sum())
        if denom == 0:
            T = float(window[-1] - window[-2])
        else:
            a = float((t_center * n_center).sum()) / denom  # n per ps
            T = float(window[-1] - window[-2]) if a == 0 else 1.0 / a
        r0 = float(window[-1])
        phases[i] = ((float(t) - r0) / T) % 1.0
    return phases

def weights_square(phi: np.ndarray, duty: float=0.5, phase_shift: float=0.0) -> np.ndarray:
    """
    Square-wave reference weights in {+1, -1}.
    ON in interval [phase_shift, phase_shift + duty) modulo 1, OFF elsewhere.
    """
    x = (phi + phase_shift) % 1.0
    w = np.empty_like(x)
    on = (x < duty)
    w[on]  =  1.0
    w[~on] = -1.0
    return w

def demod_IQ_counts(ph_ps: np.ndarray, ref_ps: np.ndarray, N:int, duty:float) -> Dict[str,int]:
    """
    Compute per-window signed I/Q counts and ON/OFF splits for each quadrature.
    """
    if ph_ps.size == 0:
        return dict(I=0, Q=0, ON_I=0, OFF_I=0, ON_Q=0, OFF_Q=0, TOT=0)
    phi = local_phase_fractions(ph_ps, ref_ps, N=N)
    wI = weights_square(phi, duty=duty, phase_shift=0.0)
    wQ = weights_square(phi, duty=duty, phase_shift=0.25)  # +90°
    I = int(wI.sum())
    Q = int(wQ.sum())
    ON_I  = int((wI > 0).sum())
    OFF_I = int((wI < 0).sum())
    ON_Q  = int((wQ > 0).sum())
    OFF_Q = int((wQ < 0).sum())
    return dict(I=I, Q=Q, ON_I=ON_I, OFF_I=OFF_I, ON_Q=ON_Q, OFF_Q=OFF_Q, TOT=int(len(ph_ps)))


# ----------------- main computation -----------------
def compute_windows(ph: np.ndarray, ref: np.ndarray, N: int, duty: float,
                    edges: np.ndarray) -> Tuple[List[Dict[str,int]], int]:
    rows: List[Dict[str,int]] = []
    used = 0
    for a, b in zip(edges[:-1], edges[1:]):
        ph_bin  = ph[(ph >= a) & (ph < b)]
        ref_bin = ref[(ref >= a) & (ref < b)]
        if ph_bin.size == 0 or ref_bin.size < max(N, 2):
            continue
        row = demod_IQ_counts(ph_bin, ref_bin, N=N, duty=duty)
        rows.append(row); used += 1
    return rows, used

def main():
    ap = argparse.ArgumentParser(description="Statistical SNR (mean/std) from per-window Liu lock-in metrics.")
    ap.add_argument("ptu", help="PicoHarp 300 T2 .ptu file")
    ap.add_argument("--ch-ph", type=int, default=1, help="photon channel (1..4)")
    ap.add_argument("--marker-bit", type=int, default=1, help="marker bit carrying modulation (1..4)")
    ap.add_argument("--N", type=int, default=10, help="# previous marker edges for local period fit")
    ap.add_argument("--duty", type=float, default=0.5, help="square-wave duty cycle (0<duty<1)")
    ap.add_argument("--start-s", type=float, default=0.0, help="analysis window start (s)")
    ap.add_argument("--stop-s", type=float, default=float('inf'), help="analysis window stop (s)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--bins", type=int, default=100, help="split into this many equal windows (default 100)")
    grp.add_argument("--bin-sec", type=float, default=0.0, help="fixed window length in seconds (overrides --bins)")
    ap.add_argument("--metric", choices=["I","IQ","S","XPHI"], default="I",
                    help="per-window metric to average: I, IQ=|vector|, S=|I|+|Q|, XPHI=projection onto global phase")
    ap.add_argument("--csv", action="store_true", help="emit per-window CSV and summary")
    args = ap.parse_args()

    header, times_by_ch, markers, _ = read_ptu(args.ptu)
    ph  = times_by_ch.get(args.ch_ph, np.array([], dtype=np.int64))
    ref = select_marker_times(markers, args.marker_bit)

    if ph.size == 0:
        print("No photons on requested channel.", file=sys.stderr); sys.exit(2)
    if ref.size < max(args.N, 2):
        print("Not enough reference edges.", file=sys.stderr); sys.exit(2)

    start_ps = int(round(args.start_s * 1e12))
    stop_ps  = int(round(args.stop_s  * 1e12)) if math.isfinite(args.stop_s) else np.iinfo(np.int64).max
    ph  = ph[(ph >= start_ps) & (ph < stop_ps)]
    ref = ref[(ref >= start_ps) & (ref < stop_ps)]
    if ph.size == 0:
        print("No photons in selected window.", file=sys.stderr); sys.exit(2)

    # Set edges either by fixed bin length or by equal partitioning
    if args.bin_sec and args.bin_sec > 0:
        bin_ps = int(round(args.bin_sec * 1e12))
        t0 = int(ph.min() // bin_ps * bin_ps)
        t1 = int(ph.max() // bin_ps * bin_ps + bin_ps)
        edges = np.arange(t0, t1+1, bin_ps, dtype=np.int64)
    else:
        # Equal bins from min to max of photons
        t0 = int(ph.min()); t1 = int(ph.max())
        bins = max(args.bins, 1)
        edges = np.linspace(t0, t1, bins+1, dtype=np.int64)

    # Global phase for XPHI if needed: accumulate on whole window
    phi_hat = None
    if args.metric == "XPHI":
        # Compute global I_sum, Q_sum on whole selection (single demod with same N,duty)
        global_row = demod_IQ_counts(ph, ref, N=args.N, duty=args.duty)
        I_sum, Q_sum = global_row['I'], global_row['Q']
        phi_hat = math.atan2(Q_sum, I_sum) if (I_sum != 0 or Q_sum != 0) else 0.0

    rows, used = compute_windows(ph, ref, N=args.N, duty=args.duty, edges=edges)

    # Build metric series
    vals: List[float] = []
    tots: List[int]   = []
    for r in rows:
        I, Q, TOT = r['I'], r['Q'], r['TOT']
        if TOT <= 0:
            continue
        if args.metric == "I":
            x = float(I)
        elif args.metric == "IQ":
            x = float(math.hypot(I, Q))
        elif args.metric == "S":
            x = float(abs(I) + abs(Q))
        elif args.metric == "XPHI":
            c, s = math.cos(phi_hat), math.sin(phi_hat)
            x = float(I*c + Q*s)
        else:
            x = float(I)
        vals.append(x); tots.append(TOT)

    n = len(vals)
    if n == 0:
        print("No valid windows to compute statistics.", file=sys.stderr); sys.exit(3)

    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else float('inf')
    snr_stat = (abs(mu) / sd) if (sd > 0 and np.isfinite(sd)) else float('inf')

    # Also report counts stats for context
    mean_counts = float(np.mean(tots))
    std_counts  = float(np.std(tots, ddof=1)) if n > 1 else 0.0

    if args.csv:
        # Per-window CSV
        print("idx,metric_val,TOT")
        for i,(x,tt) in enumerate(zip(vals, tots), 1):
            print(f"{i},{x:.6g},{tt}")
        # Summary
        print("# SUMMARY")
        print(f"metric,{args.metric}")
        print(f"windows_used,{n}")
        print(f"mean_metric,{mu:.6g}")
        print(f"std_metric,{sd:.6g}")
        print(f"SNR_stat,{'inf' if not np.isfinite(snr_stat) else f'{snr_stat:.6g}'}")
        print(f"mean_counts_per_window,{mean_counts:.6g}")
        print(f"std_counts_per_window,{std_counts:.6g}")
        if phi_hat is not None:
            print(f"phi_hat_rad,{phi_hat:.8f}")
    else:
        print("RESULT — Statistical SNR from windowed lock-in metrics")
        print(f"  Metric            : {args.metric}")
        if phi_hat is not None:
            print(f"  phi_hat (rad)     : {phi_hat:.6f}")
        print(f"  Windows used      : {n} / requested {len(edges)-1}")
        print(f"  mean(metric)      : {mu:.6f}")
        print(f"  std(metric)       : {sd:.6f}")
        print(f"  SNR_stat = |mean|/std : {snr_stat:.3f}")
        print(f"  counts/window     : mean={mean_counts:.1f}  std={std_counts:.1f}")

if __name__ == "__main__":
    main()

# python compare_SNR_statistical.py data/3mW_100kHzSine.ptu --ch-ph 1 --marker-bit 2 --N 10 --duty 0.5 --bins 200 --metric XPHI
# next we will compare to non-interleaved on/off data
# TODO: assess shot noise limit: 4xT -> 2xSNR