#!/usr/bin/env python3
"""
Liu lock-in photon counting with I/Q (quadrature) square-wave demodulation.

Usage:
  python liu_lockin_iq.py <file.ptu> \
      [--ch-ph 1] [--marker-bit 1] [--N 10] [--duty 0.5] \
      [--bin-sec 0] [--start-s 0] [--stop-s inf] [--csv]

Outputs per window/bin:
  I, Q  -> signed demodulated photon counts (square-wave PSDs)
  S     -> |I| + |Q|  (Liu-style photon number, invariant to initial phase)
  ON_I/OFF_I, ON_Q/OFF_Q, TOT for sanity checks
"""
from __future__ import annotations
import argparse, math, sys
import numpy as np
from reader import read_ptu  # your existing PTU reader

# ------------ helpers ------------
def select_marker_times(markers, bit:int) -> np.ndarray:
    mask = 1 << (bit-1)
    ts = [tt for mk, tt in markers if (mk & mask) != 0]
    return np.asarray(ts, dtype=np.int64)

def local_phase_fractions(photon_times_ps: np.ndarray,
                          ref_times_ps: np.ndarray,
                          N:int=10) -> np.ndarray:
    """
    Jakob-style local phase: for each photon time t, fit the last N reference
    edges r[k] with n = a*r + b (least squares). Period T = 1/a. Phase is
    ((t - r_last)/T) mod 1 in [0,1).
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

def demod_IQ(ph_ps: np.ndarray, ref_ps: np.ndarray, N:int, duty:float):
    """
    Compute I and Q demodulated photon counts (signed), and totals.
    Q is 90° phase-shifted square (phase_shift=+0.25 cycles).
    """
    if ph_ps.size == 0:
        return dict(I=0, Q=0, S=0, ON_I=0, OFF_I=0, ON_Q=0, OFF_Q=0, TOT=0)
    phi = local_phase_fractions(ph_ps, ref_ps, N=N)
    wI = weights_square(phi, duty=duty, phase_shift=0.0)
    wQ = weights_square(phi, duty=duty, phase_shift=0.25)  # +90°
    I = int(wI.sum())
    Q = int(wQ.sum())
    S = abs(I) + abs(Q)
    return dict(
        I=I, Q=Q, S=S,
        ON_I=int((wI > 0).sum()), OFF_I=int((wI < 0).sum()),
        ON_Q=int((wQ > 0).sum()), OFF_Q=int((wQ < 0).sum()),
        TOT=int(len(ph_ps))
    )

# ------------ CLI ------------
def main():
    ap = argparse.ArgumentParser(description="Liu I/Q lock-in photon counting from PTU")
    ap.add_argument("ptu")
    ap.add_argument("--ch-ph", type=int, default=1, help="photon channel (1..4)")
    ap.add_argument("--marker-bit", type=int, default=1, help="marker bit carrying modulation (1..4)")
    ap.add_argument("--N", type=int, default=10, help="# previous edges for local fit")
    ap.add_argument("--duty", type=float, default=0.5, help="square-wave duty cycle")
    ap.add_argument("--bin-sec", type=float, default=0.0, help="if >0, output per time bin of this length")
    ap.add_argument("--start-s", type=float, default=0.0, help="analysis window start (s)")
    ap.add_argument("--stop-s", type=float, default=float('inf'), help="analysis window stop (s)")
    ap.add_argument("--csv", action="store_true", help="CSV output")
    args = ap.parse_args()

    header, times_by_ch, markers, _ = read_ptu(args.ptu)
    ph = times_by_ch.get(args.ch_ph, np.array([], dtype=np.int64))
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

    if args.bin_sec and args.bin_sec > 0:
        bin_ps = int(round(args.bin_sec * 1e12))
        t0 = int(ph.min() // bin_ps * bin_ps)
        t1 = int(ph.max() // bin_ps * bin_ps + bin_ps)
        edges = np.arange(t0, t1+1, bin_ps, dtype=np.int64)

        if args.csv:
            print("t_start_s,t_stop_s,I,Q,S,ON_I,OFF_I,ON_Q,OFF_Q,TOT")
        for a, b in zip(edges[:-1], edges[1:]):
            ph_bin  = ph[(ph >= a) & (ph < b)]
            ref_bin = ref[(ref >= a) & (ref < b)]
            if ph_bin.size == 0 or ref_bin.size < max(args.N, 2):
                row = dict(I=0,Q=0,S=0,ON_I=0,OFF_I=0,ON_Q=0,OFF_Q=0,TOT=0)
            else:
                row = demod_IQ(ph_bin, ref_bin, args.N, args.duty)
            if args.csv:
                print(f"{a/1e12:.6f},{b/1e12:.6f},{row['I']},{row['Q']},{row['S']},{row['ON_I']},{row['OFF_I']},{row['ON_Q']},{row['OFF_Q']},{row['TOT']}")
            else:
                print(f"[{a/1e12:.3f}, {b/1e12:.3f}) s  I={row['I']:7d}  Q={row['Q']:7d}  S=|I|+|Q|={row['S']:7d}  TOT={row['TOT']:7d}")

    else:
        row = demod_IQ(ph, ref, args.N, args.duty)
        if args.csv:
            print("I,Q,S,ON_I,OFF_I,ON_Q,OFF_Q,TOT")
            print(f"{row['I']},{row['Q']},{row['S']},{row['ON_I']},{row['OFF_I']},{row['ON_Q']},{row['OFF_Q']},{row['TOT']}")
        else:
            print(f"File: {args.ptu}")
            print(f"Photon ch={args.ch_ph}, marker-bit={args.marker_bit}, N={args.N}, duty={args.duty}")
            print("RESULT — Liu I/Q lock-in photon counting:")
            print(f"  I (signed)       : {row['I']}")
            print(f"  Q (signed)       : {row['Q']}")
            print(f"  S = |I| + |Q|    : {row['S']}   (photon-number output)")
            print(f"  ON_I/OFF_I       : {row['ON_I']} / {row['OFF_I']}")
            print(f"  ON_Q/OFF_Q       : {row['ON_Q']} / {row['OFF_Q']}")
            print(f"  Total photons    : {row['TOT']}")

if __name__ == "__main__":
    main()

# python liu_lockin.py data/laserON_modulated_200s.ptu --ch-ph 1 --marker-bit 2 --N 10 --duty 0.5