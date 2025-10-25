#!/usr/bin/env python3
"""
Liu lock-in photon counting 

Implements the reference-weighted counting (RWC) scheme with two ±1 square-wave
references I and Q shifted by 90°:
- Two parallel accumulators use I(t) and Q(t) (±1 weights) at each photon time.
- Weights depend on the photon’s phase within the local modulation period.
- Outputs: I, Q (signed), and S = |I| + |Q| (photon-number output).

CLI
---
python3 liu_lockin.py <file.ptu>
  [--N 10]             # edges used per photon for local period fit
  [--duty 0.5]         # square-wave duty cycle (Liu uses 50%)
  [--start-s 0]        # crop start time (seconds)
  [--stop-s inf]       # crop stop time (seconds)
  [--bin-sec 0]        # if >0, output per time bin of this width (seconds)
  [--csv]              # print CSV per bin instead of pretty text
  
Assumptions:
- reader.read_ptu(file) -> (marker_times_ps, photon_times_ps)
  with CH 0 = reference edges (markers/sync), CH 1 = photons.
"""

from __future__ import annotations
import argparse, math, sys
from typing import Tuple, Optional
import numpy as np
from reader import read_ptu


def local_phase_fractions(
    photon_times_ps: np.ndarray,
    ref_edges_ps: np.ndarray,
    N: int = 10,
) -> np.ndarray:
    """
    for each photon at time t, estimate local period T from the last N reference edges
    r[j0 ... j] via linear regression of edge index n on time, then compute
      phi = ((t - r[j]) / T) % 1  in [0,1).
    """
    if ref_edges_ps.size < max(N, 2):
        raise ValueError("Not enough reference edges for local phase estimation")

    r = ref_edges_ps.astype(np.float64)
    phases = np.empty(photon_times_ps.shape[0], dtype=np.float64)

    j = N - 1
    n_idx = np.arange(N, dtype=np.float64)
    n_mean = n_idx.mean()

    for i, t in enumerate(photon_times_ps):
        while j + 1 < r.size and r[j + 1] <= t:
            j += 1
        j_use = max(j, N - 1)
        j0 = j_use - (N - 1)
        window = r[j0 : j_use + 1] 

        # linear reg
        t_mean = window.mean()
        t_center = window - t_mean
        denom = float((t_center**2).sum())
        if denom == 0.0:
            T = float(window[-1] - window[-2])
            if T <= 0:
                T = 1.0
        else:
            a = float(((t_center) * (n_idx - n_mean)).sum()) / denom  
            if a == 0:
                T = float(window[-1] - window[-2])
                if T <= 0:
                    T = 1.0
            else:
                T = 1.0 / a  

        r_last = float(window[-1])
        phases[i] = ((float(t) - r_last) / T) % 1.0

    return phases


def sq_weights(phi: np.ndarray, duty: float = 0.5, phase_shift: float = 0.0) -> np.ndarray:
    """
    bipolar square-wave weights in {+1, -1}
    +1 during ON window, -1 during OFF
    """
    x = (phi + phase_shift) % 1.0
    w = np.empty_like(x)
    on = x < duty
    w[on] = 1.0
    w[~on] = -1.0
    return w


def demod_iq_from_phases(
    phi: np.ndarray,
    duty: float = 0.5,
) -> Tuple[int, int, int, int, int, int]:
    """
    given per-photon phases, compute I and Q via ±1 square weights.
    Returns: (I, Q, on_I, off_I, on_Q, off_Q)
    """
    # I: 0° reference (ON first half)
    wI = sq_weights(phi, duty=duty, phase_shift=0.0)
    # Q: 90° reference (quarter-period shift)
    wQ = sq_weights(phi, duty=duty, phase_shift=0.25)

    I = int(np.round(wI.sum()))
    Q = int(np.round(wQ.sum()))
    on_I = int((wI > 0).sum())
    off_I = int((wI < 0).sum())
    on_Q = int((wQ > 0).sum())
    off_Q = int((wQ < 0).sum())
    return I, Q, on_I, off_I, on_Q, off_Q


def crop_by_time(
    t_ps: np.ndarray, start_s: float, stop_s: float
) -> np.ndarray:
    start_ps = 0 if start_s <= 0 else int(start_s * 1e12)
    stop_ps = np.iinfo(np.int64).max if not np.isfinite(stop_s) else int(stop_s * 1e12)
    return (t_ps >= start_ps) & (t_ps < stop_ps)


def make_bins(start_ps: int, stop_ps: int, bin_sec: float) -> np.ndarray:
    if bin_sec <= 0:
        return np.array([start_ps, stop_ps], dtype=np.int64)
    w_ps = int(bin_sec * 1e12)
    if w_ps <= 0:
        w_ps = 1
    n = max(1, int(math.ceil((stop_ps - start_ps) / w_ps)))
    edges = start_ps + np.arange(n + 1, dtype=np.int64) * w_ps
    edges[-1] = stop_ps
    return edges



def estimate_global_freq(ref_edges_ps: np.ndarray) -> float:
    """ global f_mod [Hz] from median period between consecutive ref edges"""
    if ref_edges_ps.size < 2:
        return float("nan")
    d = np.diff(ref_edges_ps.astype(np.int64))
    d = d[d > 0]
    if d.size == 0:
        return float("nan")
    T_ps = float(np.median(d))
    return 1e12 / T_ps


def run(
    ptu_path: str,
    N: int,
    duty: float,
    start_s: float,
    stop_s: float,
    bin_sec: float,
    csv: bool,
) -> int:
    # read PTU 
    marker_times_ps, photon_times_ps = read_ptu(ptu_path)  # CH0=markers, CH1=photons

    if marker_times_ps.size == 0:
        print("ERROR: No marker/sync edges (CH0) found. Cannot demodulate.", file=sys.stderr)
        return 2
    if photon_times_ps.size == 0:
        print("WARNING: No photons (CH1) found. Nothing to demodulate.", file=sys.stderr)

    # crop interval
    full_start_ps = int(marker_times_ps.min())
    full_stop_ps  = int(max(marker_times_ps.max(), photon_times_ps.max() if photon_times_ps.size else full_start_ps))
    sel_ph = crop_by_time(photon_times_ps, start_s, stop_s)
    sel_mk = crop_by_time(marker_times_ps, start_s, stop_s)

    photons = photon_times_ps[sel_ph]
    refs    = marker_times_ps[sel_mk]

    # basic stats
    f_est = estimate_global_freq(refs)
    total_ph = int(photons.size)
    ref_edges = int(refs.size)

    # binning
    start_ps = int(full_start_ps if start_s <= 0 else max(full_start_ps, int(start_s * 1e12)))
    stop_ps  = int(full_stop_ps  if not np.isfinite(stop_s) else min(full_stop_ps,  int(stop_s  * 1e12)))
    edges = make_bins(start_ps, stop_ps, bin_sec)

    # header
    if not csv:
        print(f"File: {ptu_path}")
        print("Reader: PicoHarp 330 T2 (assumed).")
        print(f"Ref edges: {ref_edges:>d}   Photons used: {total_ph:>d}   N_fit={N}")
        if np.isfinite(f_est):
            print(f"Est. f_mod ≈ {f_est:0.6f} Hz  (from median period)   duty={duty}")
        else:
            print(f"Est. f_mod: n/a   duty={duty}")

    # compute phases, demod, print
    if csv:
        print("bin_start_s,bin_end_s,I,Q,S,on_I,off_I,on_Q,off_Q,total_photons")

    for b in range(edges.size - 1):
        b0, b1 = int(edges[b]), int(edges[b + 1])

        m_ph = (photons >= b0) & (photons < b1)
        m_rf = (refs    >= b0) & (refs    < b1)

        ph_bin = photons[m_ph]
        rf_bin = refs[m_rf]

        if rf_bin.size < N:
        
            i0 = np.searchsorted(refs, b0, side="left")
            i1 = np.searchsorted(refs, b1, side="right")
            left_needed  = max(0, N - (i1 - i0))
            right_needed = max(0, N - (i1 - i0) - left_needed)
            i0 = max(0, i0 - left_needed)
            i1 = min(refs.size, i1 + right_needed)
            rf_use = refs[i0:i1]
        else:
            rf_use = rf_bin

        if ph_bin.size == 0 or rf_use.size < N:
            I = Q = onI = offI = onQ = offQ = 0
        else:
            phi = local_phase_fractions(ph_bin, rf_use, N=N)
            I, Q, onI, offI, onQ, offQ = demod_iq_from_phases(phi, duty=duty)

        S = abs(I) + abs(Q)
        tot = int(ph_bin.size)

        if csv:
            print(f"{(b0/1e12):.6f},{(b1/1e12):.6f},{I},{Q},{S},{onI},{offI},{onQ},{offQ},{tot}")
        else:
            if edges.size == 2:
                print("RESULT — Liu I/Q lock-in (quadrature RWC):")
            else:
                print(f"\nBin {b+1}/{edges.size-1}  [{b0/1e12:.3f}s – {b1/1e12:.3f}s]  N={tot}")
            print(f"  I (signed)       : {I}")
            print(f"  Q (signed)       : {Q}")
            print(f"  S = |I| + |Q|    : {S}   (photon-number output)")
            print(f"  ON_I / OFF_I     : {onI} / {offI}")
            print(f"  ON_Q / OFF_Q     : {onQ} / {offQ}")
            print(f"  Total photons    : {tot}")

    return 0

def run_chunks(
    ptu_path: str,
    N: int = 10,
    duty: float = 0.5,
    n_chunks: int = 40,
) -> list[dict]:
    """
    divide dataset into `n_chunks` equal-duration segments,
    run lock-in analysis on each, and return results as a list of dicts.
    Each dict contains: start_s, stop_s, I, Q, S, on_I, off_I, on_Q, off_Q, total_photons.
    """
    marker_times_ps, photon_times_ps = read_ptu(ptu_path)
    if marker_times_ps.size == 0 or photon_times_ps.size == 0:
        raise ValueError("PTU file has no marker or photon data.")

    t_min = float(photon_times_ps.min())
    t_max = float(photon_times_ps.max())
    edges = np.linspace(t_min, t_max, n_chunks + 1)

    results = []
    for i in range(n_chunks):
        b0, b1 = int(edges[i]), int(edges[i + 1])
        ph_bin = photon_times_ps[(photon_times_ps >= b0) & (photon_times_ps < b1)]
        rf_bin = marker_times_ps[(marker_times_ps >= b0) & (marker_times_ps < b1)]

        # if too few edges, extend selection (like in run())
        if rf_bin.size < N:
            i0 = np.searchsorted(marker_times_ps, b0, side="left")
            i1 = np.searchsorted(marker_times_ps, b1, side="right")
            left_needed = max(0, N - (i1 - i0))
            right_needed = max(0, N - (i1 - i0) - left_needed)
            i0 = max(0, i0 - left_needed)
            i1 = min(marker_times_ps.size, i1 + right_needed)
            rf_use = marker_times_ps[i0:i1]
        else:
            rf_use = rf_bin

        if ph_bin.size == 0 or rf_use.size < N:
            I = Q = onI = offI = onQ = offQ = 0
        else:
            phi = local_phase_fractions(ph_bin, rf_use, N=N)
            I, Q, onI, offI, onQ, offQ = demod_iq_from_phases(phi, duty=duty)

        results.append(dict(
            start_s=b0 / 1e12,
            stop_s=b1 / 1e12,
            I=I,
            Q=Q,
            S=np.sqrt((I**2) + (Q**2)),
            on_I=onI,
            off_I=offI,
            on_Q=onQ,
            off_Q=offQ,
            total_photons=int(ph_bin.size),
        ))

    return results

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Liu lock-in photon counting (I/Q, square-wave RWC).")
    ap.add_argument("ptu", help="PTU file path")
    ap.add_argument("--N", type=int, default=10, help="number of ref edges per photon for local period (default: 10)")
    ap.add_argument("--duty", type=float, default=0.5, help="square-wave duty in [0,1] (Liu uses 0.5)")
    ap.add_argument("--start-s", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--stop-s", type=float, default=float("inf"), help="stop time (s)")
    ap.add_argument("--bin-sec", type=float, default=0.0, help="if >0, output per time bin of this width (s)")
    ap.add_argument("--csv", action="store_true", help="CSV output")
    args = ap.parse_args(argv)

    if not (0.0 < args.duty < 1.0):
        print("ERROR: --duty must be in (0,1).", file=sys.stderr)
        return 2
    if args.N < 2:
        print("ERROR: --N must be >= 2.", file=sys.stderr)
        return 2

    try:
        return run(
            ptu_path=args.ptu,
            N=args.N,
            duty=args.duty,
            start_s=args.start_s,
            stop_s=args.stop_s,
            bin_sec=args.bin_sec,
            csv=args.csv,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
