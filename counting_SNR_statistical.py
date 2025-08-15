
"""
counting_SNR_statistical.py

Compute statistical SNRs for *sequential* ON/OFF photon-counting runs.
Folder must contain paired files like:
  ON_000.ptu, OFF_000.ptu, ON_001.ptu, OFF_001.ptu, ...

We compute, per pair k:
  t_on, t_off     : acquisition durations inferred from photon timestamps (ps -> s)
  C_on, C_off     : photon counts on the chosen channel
  α_k             : exposure ratio = t_on / t_off
  ΔC_k            : C_on - α_k * C_off                (background-subtracted counts)
  Var_k           : C_on + α_k^2 * C_off              (shot-noise variance under Poisson)
  z_k             : ΔC_k / sqrt(Var_k)                (per-pair SNR "z-score")

We then report:
  • Pooled SNR (all pairs merged):
        SNR_pooled_counts = (Σ ΔC_k) / sqrt(Σ Var_k)
    (this is the optimal fixed-α Poisson SNR for the total experiment)

  • Statistical SNRs (mean / std across pairs), for two metrics:
        m_counts_k = ΔC_k
        m_rates_k  = C_on/t_on - C_off/t_off
    and SNR_stat(metric) = |mean| / std, computed on {k} with finite std.

  • Descriptive stats per pair and totals (counts, durations, rates).

Notes / Caveats
---------------
1) Duration inference: We estimate run length as (max_ts - min_ts). If marker edges are available
   you can improve this by taking the first/last marker timestamps instead. For typical continuous
   acquisitions this proxy is fine, especially when runs are ≥100 ms.

2) Empty files: pairs with zero photons in both files give Var_k = 0. Those pairs are skipped
   for z_k and for the statistical mean/std (but still counted in a summary).

3) Different acquisition lengths: handled via α_k. If your lab uses fixed equal times, α_k ≈ 1.

4) Channel selection: use --ch-ph to choose which photon channel (1..4) to analyze.

Usage
-----
  python counting_SNR_statistical.py <folder> \
      [--ch-ph 1] [--max-pairs inf] [--csv out.csv]

Example
-------
  python counting_SNR_statistical.py ./sequential_runs --ch-ph 1 --csv result.csv
"""

from __future__ import annotations
import argparse, math, os, re, sys, csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np

# --- PTU reader (expects reader.py next to this script or on PYTHONPATH) ---
try:
    from reader import read_ptu
except Exception as e:
    sys.stderr.write("ERROR: couldn't import reader.py (read_ptu). Place reader.py next to this script.\n")
    raise

IDX_RE = re.compile(r'_(\d+)\.ptu$', re.IGNORECASE)

def extract_index(p: Path) -> Optional[int]:
    m = IDX_RE.search(p.name)
    return int(m.group(1)) if m else None

def run_duration_seconds(times_ps: np.ndarray) -> float:
    """Estimate acquisition duration from photon timestamps (ps -> s)."""
    if times_ps.size < 2:
        # Fallback: zero-length would nuke α; return 0 to signal unknown
        return 0.0
    t0 = float(times_ps.min())
    t1 = float(times_ps.max())
    dt = max(0.0, (t1 - t0) * 1e-12)  # ps -> s
    return dt

def get_counts_and_duration(ptu_path: Path, ch_ph: int) -> Tuple[int, float]:
    """Return (counts_on_channel, duration_s)."""
    header, times_by_ch, markers, _ = read_ptu(str(ptu_path))
    ph = times_by_ch.get(ch_ph, np.array([], dtype=np.int64))
    C = int(ph.size)
    # Prefer duration from photon timestamps; if empty or degenerate, try header fallback.
    dt = run_duration_seconds(ph)
    if dt <= 0.0:
        # Try to fall back to header tag 'MeasDesc_AcquisitionTime' if present (seconds)
        dt = float(header.get("MeasDesc_AcquisitionTime", 0.0))
    return C, dt

def main():
    ap = argparse.ArgumentParser(description="Statistical SNR for sequential ON/OFF photon-counting runs")
    ap.add_argument("folder", type=str, help="Folder containing paired ON_xxx.ptu / OFF_xxx.ptu files")
    ap.add_argument("--ch-ph", type=int, default=1, help="Photon channel to analyze (1..4)")
    ap.add_argument("--max-pairs", type=int, default=None, help="Analyze at most this many index pairs (sorted)")
    ap.add_argument("--csv", type=str, default=None, help="Optional CSV output path for per-pair metrics")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        sys.exit(2)

    on_files  = sorted([p for p in folder.glob("ON_*.ptu")  if extract_index(p) is not None], key=extract_index)
    off_files = sorted([p for p in folder.glob("OFF_*.ptu") if extract_index(p) is not None], key=extract_index)

    map_on  = {extract_index(p): p for p in on_files}
    map_off = {extract_index(p): p for p in off_files}
    common_idxs = sorted(set(map_on.keys()) & set(map_off.keys()))
    if not common_idxs:
        print("ERROR: No matching ON_xxx/OFF_xxx pairs found.", file=sys.stderr)
        sys.exit(2)

    if args.max_pairs is not None:
        common_idxs = common_idxs[:args.max_pairs]

    rows = []  # per pair metrics
    sum_delta = 0.0
    sum_var   = 0.0
    total_con = 0
    total_coff= 0
    total_ton = 0.0
    total_toff= 0.0
    skipped_z = 0

    for idx in common_idxs:
        p_on  = map_on[idx]
        p_off = map_off[idx]

        C_on,  t_on  = get_counts_and_duration(p_on,  args.ch_ph)
        C_off, t_off = get_counts_and_duration(p_off, args.ch_ph)

        # Guard against zero durations; if both zero, we can't define α; try α=1 but mark as bad
        if t_on <= 0.0 or t_off <= 0.0:
            alpha = 1.0
            alpha_note = "alpha=1.0 (undefined durations)"
        else:
            alpha = t_on / t_off
            alpha_note = ""

        delta = C_on - alpha * C_off
        var   = C_on + (alpha*alpha) * C_off
        z     = (delta / math.sqrt(var)) if var > 0 else float("nan")
        rate_on  = (C_on  / t_on) if t_on > 0 else float("nan")
        rate_off = (C_off / t_off) if t_off > 0 else float("nan")
        metric_rate = (rate_on - rate_off) if (t_on>0 and t_off>0) else float("nan")

        rows.append(dict(
            idx=idx,
            C_on=C_on, C_off=C_off,
            t_on=t_on, t_off=t_off,
            alpha=alpha,
            delta=delta, var=var, z=z,
            rate_on=rate_on, rate_off=rate_off, metric_rate=metric_rate,
            note=alpha_note
        ))

        # pooled tallies
        if var > 0:
            sum_delta += delta
            sum_var   += var
        else:
            skipped_z += 1

        total_con  += C_on
        total_coff += C_off
        total_ton  += t_on if t_on>0 else 0.0
        total_toff += t_off if t_off>0 else 0.0

    # Statistical SNRs (mean/std across pairs) for metrics with finite values
    def mean_std_finite(values: List[float]) -> Tuple[float, float]:
        vals = np.array([v for v in values if np.isfinite(v)])
        if vals.size == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(vals.std(ddof=1) if vals.size > 1 else 0.0)

    m_counts = [r["delta"] for r in rows if np.isfinite(r["delta"])]
    m_rates  = [r["metric_rate"] for r in rows if np.isfinite(r["metric_rate"])]
    mean_counts, std_counts = mean_std_finite(m_counts)
    mean_rates,  std_rates  = mean_std_finite(m_rates)

    snr_stat_counts = (abs(mean_counts) / std_counts) if (std_counts and np.isfinite(std_counts) and std_counts>0) else float("nan")
    snr_stat_rates  = (abs(mean_rates)  / std_rates)  if (std_rates  and np.isfinite(std_rates)  and std_rates>0)  else float("nan")

    snr_pooled = (sum_delta / math.sqrt(sum_var)) if sum_var > 0 else float("nan")

    # Totals summary
    alpha_tot = (total_ton / total_toff) if (total_ton>0 and total_toff>0) else float("nan")
    delta_tot = total_con - (alpha_tot * total_coff if np.isfinite(alpha_tot) else total_coff)
    var_tot   = total_con + (alpha_tot*alpha_tot * total_coff if np.isfinite(alpha_tot) else total_coff)
    rate_on_tot  = (total_con / total_ton) if total_ton > 0 else float("nan")
    rate_off_tot = (total_coff / total_toff) if total_toff > 0 else float("nan")

    # --- Text output ---
    print("RESULT — Statistical SNR for sequential ON/OFF runs")
    print(f"  Folder            : {folder}")
    print(f"  Channel           : {args.ch_ph}")
    print(f"  Pairs found/used  : {len(set(map_on))}/{len(rows)}")
    print()
    print("  --- Per-experiment pooled SNR (optimal Poisson) ---")
    print(f"  SNR_pooled_counts : {snr_pooled:.4f}")
    print(f"    ΣΔC / sqrt(ΣVar) where ΔC=C_on-αC_off, Var=C_on+α²C_off")
    print()
    print("  --- Statistical SNRs across pairs (mean/std) ---")
    print(f"  Metric ΔC (counts): mean={mean_counts:.6g}  std={std_counts:.6g}  SNR_stat={snr_stat_counts:.3f}")
    print(f"  Metric rate (Hz)  : mean={mean_rates:.6g}   std={std_rates:.6g}   SNR_stat={snr_stat_rates:.3f}")
    print()
    print("  --- Totals (descriptive) ---")
    print(f"  Total counts ON/OFF: {total_con} / {total_coff}")
    print(f"  Total time   ON/OFF: {total_ton:.6f} s / {total_toff:.6f} s")
    print(f"  Total rates  ON/OFF: {rate_on_tot:.6g} Hz / {rate_off_tot:.6g} Hz")
    print(f"  α_total (t_on/t_off): {alpha_tot:.6g}")
    print(f"  ΔC_total, Var_total : {delta_tot:.6g}, {var_tot:.6g}")
    if skipped_z:
        print(f"  Note: skipped {skipped_z} pair(s) with zero variance for z_k.")
    print()
    print("Tip: If your ON/OFF durations are equal by design, you should see α≈1 for all pairs.")

    # --- Optional CSV ---
    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx","C_on","C_off","t_on_s","t_off_s","alpha","delta_counts","var_counts","z_pair","rate_on_Hz","rate_off_Hz","metric_rate_Hz","note"])
            for r in rows:
                w.writerow([r["idx"], r["C_on"], r["C_off"], f"{r['t_on']:.9f}", f"{r['t_off']:.9f}", f"{r['alpha']:.9f}",
                            f"{r['delta']:.6f}", f"{r['var']:.6f}", f"{r['z']:.6f}" if np.isfinite(r["z"]) else "",
                            f"{r['rate_on']:.9f}" if np.isfinite(r["rate_on"]) else "",
                            f"{r['rate_off']:.9f}" if np.isfinite(r["rate_off"]) else "",
                            f"{r['metric_rate']:.9f}" if np.isfinite(r["metric_rate"]) else "",
                            r["note"]])
            # Footer with summary rows
            w.writerow([])
            w.writerow(["SUMMARY","--","--",f"{total_ton:.9f}",f"{total_toff:.9f}",f"{alpha_tot:.9f}",
                        f"{delta_tot:.6f}", f"{var_tot:.6f}", f"{snr_pooled:.6f}",
                        f"{rate_on_tot:.9f}", f"{rate_off_tot:.9f}", "", "pooled + totals"])
            w.writerow(["STAT_SNR","metric=ΔC","", "", "", "",
                        f"{mean_counts:.6f}", f"{std_counts:.6f}", f"{snr_stat_counts:.6f}", "", "", "", "across pairs"])
            w.writerow(["STAT_SNR","metric=rate","", "", "", "",
                        f"{mean_rates:.6f}", f"{std_rates:.6f}", f"{snr_stat_rates:.6f}", "", "", "", "across pairs"])
        print(f"\nSaved per-pair metrics to: {csv_path}")

if __name__ == "__main__":
    main()
