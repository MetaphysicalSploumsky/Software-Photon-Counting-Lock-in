#!/usr/bin/env python3
"""
compare_SNR.py — Compare SNR for Liu-style I/Q lock-in vs 'regular' photon counting.

This script reads a PicoHarp 300 T2 .ptu file (PH300) with photon timestamps and marker
timestamps (carrying the modulation), performs square-wave I/Q demodulation à la Liu,
and compares several signal-to-noise estimates computed over the *same* photons:

  (1) Lock-in (I-only):      SNR_I   = |I| / sqrt(TOT)
  (2) Lock-in (I/Q, L2):     SNR_IQ  = sqrt(I^2 + Q^2) / sqrt(TOT)
  (3) Lock-in (Liu L1 sum):  SNR_L1  = (|I| + |Q|) / sqrt(2 * TOT)     [approximate]
  (4) Regular On-Off diff:   SNR_ONOFF = (ON - OFF) / sqrt(ON + OFF)
  (5) Regular BG-sub:        SNR_BG   = (ON - α*OFF) / sqrt(ON + α^2*OFF), α=duty/(1-duty)

Notes on the SNR choices
------------------------
• For square-wave weights w ∈ {+1,-1}, I = (# in + window) - (# in − window).
  If photons are Poisson-distributed, Var(I) ≈ ON + OFF = TOT, hence SNR_I above.
• For two orthogonal square references, Var(I) ≈ Var(Q) ≈ TOT and (approximately)
  Var(I)+Var(Q) ≈ 2*TOT. The Liu "photon number" S = |I| + |Q| is L1-combining the
  quadratures and is robust to an unknown initial phase. We report an *approximate*
  SNR for S by using sqrt(Var(I)+Var(Q)) in the denominator.
• The “regular” methods are the typical count-difference (ON−OFF) and a duty-corrected
  background subtraction. These are statistically equivalent to square-wave lock-in with
  a single phase (I), but we show them explicitly because they are common baselines.

If --bin-sec > 0 is provided, SNRs are computed per bin and also summarized (mean, std).
Otherwise a single set of numbers is computed over the whole selected window.

Example:
  python compare_SNR.py data/example.ptu --ch-ph 1 --marker-bit 2 --N 10 --duty 0.5 --bin-sec 1 --csv

Requirements:
  • reader.py in the same directory (as provided by you).
  • The .ptu file must be PicoHarp 300 T2 (PH300 T2) format.
"""
from __future__ import annotations
import argparse, math, sys
import numpy as np
from typing import Dict, Any

from reader import read_ptu  # your provided minimal PH300 T2 reader


# ----------------- helpers copied/adapted from Liu I/Q implementation -----------------
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
    Compute I and Q signed counts and ON/OFF splits for each quadrature.
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


# ----------------- SNR formulas -----------------
def snr_lockin_I(I:int, TOT:int) -> float:
    # Var(I) ≈ ON+OFF = TOT for square weights
    return (abs(I) / np.sqrt(TOT)) if TOT > 0 else 0.0

def snr_lockin_IQ_L2(I:int, Q:int, TOT:int) -> float:
    # Simple L2 combination; denominator uses sqrt(TOT) as a pragmatic comparator
    amp = math.hypot(I, Q)
    return (amp / np.sqrt(TOT)) if TOT > 0 else 0.0

def snr_lockin_Liu_L1(I:int, Q:int, TOT:int) -> float:
    # Approximate: Var(I)+Var(Q) ≈ 2*TOT
    return ((abs(I) + abs(Q)) / np.sqrt(2.0*TOT)) if TOT > 0 else 0.0

def snr_onoff(ON:int, OFF:int) -> float:
    num = ON - OFF
    den = np.sqrt(max(ON + OFF, 0))
    return (num / den) if den > 0 else 0.0

def snr_bg_sub(ON:int, OFF:int, duty:float) -> float:
    # Scale OFF to match ON window width
    alpha = duty/(1.0 - duty) if duty < 1.0 else 0.0
    num = ON - alpha*OFF
    den = np.sqrt(ON + (alpha**2) * OFF) if (ON + OFF) > 0 else 0.0
    return (num / den) if den > 0 else 0.0


# ----------------- main -----------------
def compute_metrics_for_window(ph_ps, ref_ps, N, duty) -> Dict[str, Any]:
    d = demod_IQ_counts(ph_ps, ref_ps, N=N, duty=duty)
    I,Q, ONI,OFFI, ONQ,OFFQ, TOT = d['I'], d['Q'], d['ON_I'], d['OFF_I'], d['ON_Q'], d['OFF_Q'], d['TOT']

    res = dict(
        TOT=TOT,
        I=I, Q=Q,
        ON_I=ONI, OFF_I=OFFI, ON_Q=ONQ, OFF_Q=OFFQ,
        S_Liu = abs(I) + abs(Q),
        SNR_I    = snr_lockin_I(I, TOT),
        SNR_IQ   = snr_lockin_IQ_L2(I, Q, TOT),
        SNR_L1   = snr_lockin_Liu_L1(I, Q, TOT),
        SNR_ONOFF_I  = snr_onoff(ONI, OFFI),
        SNR_BG_I     = snr_bg_sub(ONI, OFFI, duty),
        SNR_ONOFF_Q  = snr_onoff(ONQ, OFFQ),
        SNR_BG_Q     = snr_bg_sub(ONQ, OFFQ, duty),
    )
    return res

def main():
    ap = argparse.ArgumentParser(description="Compare SNR: Liu I/Q lock-in vs regular counting.")
    ap.add_argument("ptu", help="PicoHarp 300 T2 file")
    ap.add_argument("--ch-ph", type=int, default=1, help="photon channel (1..4)")
    ap.add_argument("--marker-bit", type=int, default=1, help="marker bit carrying the square modulation (1..4)")
    ap.add_argument("--N", type=int, default=10, help="# previous marker edges for local period fit")
    ap.add_argument("--duty", type=float, default=0.5, help="square-wave duty cycle (0<duty<1)")
    ap.add_argument("--start-s", type=float, default=0.0, help="analysis window start (s)")
    ap.add_argument("--stop-s", type=float, default=float('inf'), help="analysis window stop (s)")
    ap.add_argument("--bin-sec", type=float, default=0.0, help="if >0, compute metrics per bin of this length")
    ap.add_argument("--csv", action="store_true", help="CSV output")
    ap.add_argument("--summary-only", action="store_true", help="with --bin-sec, print only summary stats")
    args = ap.parse_args()

    header, times_by_ch, markers, reader_name = read_ptu(args.ptu)
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

        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            ph_bin  = ph[(ph >= a) & (ph < b)]
            ref_bin = ref[(ref >= a) & (ref < b)]
            if ph_bin.size == 0 or ref_bin.size < max(args.N, 2):
                row = dict(TOT=0,I=0,Q=0,ON_I=0,OFF_I=0,ON_Q=0,OFF_Q=0,
                           S_Liu=0,SNR_I=0,SNR_IQ=0,SNR_L1=0,SNR_ONOFF_I=0,SNR_BG_I=0,SNR_ONOFF_Q=0,SNR_BG_Q=0,
                           t_start_s=a/1e12, t_stop_s=b/1e12)
            else:
                row = compute_metrics_for_window(ph_bin, ref_bin, args.N, args.duty)
                row['t_start_s'] = a/1e12
                row['t_stop_s']  = b/1e12
            rows.append(row)

        # Summary
        def summarize(key):
            vals = np.array([r[key] for r in rows if r['TOT']>0], dtype=float)
            return dict(mean=float(np.mean(vals)) if vals.size else 0.0,
                        std=float(np.std(vals, ddof=1)) if vals.size>1 else 0.0,
                        n=int(vals.size))

        summary = {k: summarize(k) for k in [
            'SNR_I','SNR_IQ','SNR_L1','SNR_ONOFF_I','SNR_BG_I','SNR_ONOFF_Q','SNR_BG_Q'
        ]}
        summary['count_rate_cps'] = summarize('TOT')  # not strictly cps; per-bin counts

        if args.csv:
            if not args.summary-only:
                print("t_start_s,t_stop_s,TOT,I,Q,S_Liu,ON_I,OFF_I,ON_Q,OFF_Q,SNR_I,SNR_IQ,SNR_L1,SNR_ONOFF_I,SNR_BG_I,SNR_ONOFF_Q,SNR_BG_Q")
                for r in rows:
                    print(f"{r['t_start_s']:.6f},{r['t_stop_s']:.6f},{r['TOT']},{r['I']},{r['Q']},{r['S_Liu']},{r['ON_I']},{r['OFF_I']},"
                          f"{r['ON_Q']},{r['OFF_Q']},{r['SNR_I']:.6g},{r['SNR_IQ']:.6g},{r['SNR_L1']:.6g},{r['SNR_ONOFF_I']:.6g},"
                          f"{r['SNR_BG_I']:.6g},{r['SNR_ONOFF_Q']:.6g},{r['SNR_BG_Q']:.6g}")
            # summary CSV
            print("# SUMMARY (per-bin metrics)")
            print("metric,mean,std,n")
            for k,v in summary.items():
                if isinstance(v, dict):
                    print(f"{k},{v['mean']:.6g},{v['std']:.6g},{v['n']}")
            # count-rate proxy
            cr = summary['count_rate_cps']
            print(f"counts_per_bin,{cr['mean']:.6g},{cr['std']:.6g},{cr['n']}")
        else:
            if not args.summary-only:
                print(f"Per-bin results (bin={args.bin_sec} s):")
                for r in rows:
                    print(f"[{r['t_start_s']:.3f}, {r['t_stop_s']:.3f})s  TOT={r['TOT']:6d}  I={r['I']:6d}  Q={r['Q']:6d}  "
                          f"SNR_I={r['SNR_I']:.3f}  SNR_IQ={r['SNR_IQ']:.3f}  SNR_L1={r['SNR_L1']:.3f}  "
                          f"ONOFF_I={r['SNR_ONOFF_I']:.3f}  BG_I={r['SNR_BG_I']:.3f}")
            print("\nSUMMARY across non-empty bins:")
            for k,v in summary.items():
                if isinstance(v, dict):
                    print(f"  {k:12s}: mean={v['mean']:.3f}  std={v['std']:.3f}  n={v['n']}")
            cr = summary['count_rate_cps']
            print(f"  counts/bin   : mean={cr['mean']:.1f}  std={cr['std']:.1f}  n={cr['n']}")

    else:
        # Whole-window computation
        row = compute_metrics_for_window(ph, ref, args.N, args.duty)
        if args.csv:
            print("TOT,I,Q,S_Liu,ON_I,OFF_I,ON_Q,OFF_Q,SNR_I,SNR_IQ,SNR_L1,SNR_ONOFF_I,SNR_BG_I,SNR_ONOFF_Q,SNR_BG_Q")
            print(f"{row['TOT']},{row['I']},{row['Q']},{row['S_Liu']},{row['ON_I']},{row['OFF_I']},"
                  f"{row['ON_Q']},{row['OFF_Q']},{row['SNR_I']:.6g},{row['SNR_IQ']:.6g},{row['SNR_L1']:.6g},"
                  f"{row['SNR_ONOFF_I']:.6g},{row['SNR_BG_I']:.6g},{row['SNR_ONOFF_Q']:.6g},{row['SNR_BG_Q']:.6g}")
        else:
            print("RESULT — SNR comparison (whole window):")
            print(f"  TOT photons       : {row['TOT']}")
            print(f"  I, Q (signed)     : {row['I']}, {row['Q']}")
            print(f"  ON_I / OFF_I      : {row['ON_I']} / {row['OFF_I']}")
            print(f"  ON_Q / OFF_Q      : {row['ON_Q']} / {row['OFF_Q']}")
            print("  --- Lock-in SNRs ---")
            print(f"  SNR_I   = |I|/sqrt(TOT)                 : {row['SNR_I']:.4f}")
            print(f"  SNR_IQ  = sqrt(I^2+Q^2)/sqrt(TOT)      : {row['SNR_IQ']:.4f}")
            print(f"  SNR_L1  = (|I|+|Q|)/sqrt(2*TOT)  (≈)   : {row['SNR_L1']:.4f}")
            print("  --- Regular counting baselines ---")
            print(f"  On-Off (I)   : (ON-OFF)/sqrt(ON+OFF)   : {row['SNR_ONOFF_I']:.4f}")
            print(f"  BG-sub (I)   : ON-αOFF / sqrt(ON+α²OFF): {row['SNR_BG_I']:.4f}   (α = duty/(1-duty))")
            print(f"  On-Off (Q)   : {row['SNR_ONOFF_Q']:.4f}")
            print(f"  BG-sub (Q)   : {row['SNR_BG_Q']:.4f}")

if __name__ == "__main__":
    main()

# python compare_SNR.py data/laserON_modulated_200s.ptu --ch-ph 1 --marker-bit 2 --N 10 --duty 0.5 --bin-sec 1