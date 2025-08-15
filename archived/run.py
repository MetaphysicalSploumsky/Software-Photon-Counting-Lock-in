# run.py — read → select ref from markers → lock-in
import argparse, numpy as np
from reader import read_ptu
from lockin import phasor_lockin_ps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptu")
    ap.add_argument("--ch-ph", type=int, default=1)
    ap.add_argument("--marker-bit", type=int, default=2, help="keep records where (mask & (1<<bit))!=0")
    ap.add_argument("--ref-every-n", type=int, default=1, help="subsample reference edges")
    args = ap.parse_args()

    header, ch, markers, reader = read_ptu(args.ptu)
    t_ph = ch.get(args.ch_ph, np.array([], dtype=np.int64))
    mask = 1 << args.marker_bit
    t_ref = np.asarray([tps for (mk, tps) in markers if (mk & mask)!=0], dtype=np.int64)
    if args.ref_every_n > 1 and t_ref.size:
        t_ref = t_ref[::args.ref_every_n].copy()

    print(f"Reader: {reader}")
    print(f"Photons (CH{args.ch_ph}): {t_ph.size:,}")
    print(f"Ref markers (bit={args.marker_bit}): {t_ref.size:,}")
    if t_ph.size == 0 or t_ref.size < 2:
        print("Not enough data."); return

    res = phasor_lockin_ps(t_ph, t_ref)
    med_dt_ps = float(np.median(np.diff(t_ref))) if t_ref.size>2 else float('nan')
    print(f"Median period ≈ {med_dt_ps*1e-6:.3f} µs  (f≈{1e6/med_dt_ps:.3f} Hz)")
    print(f"Estimated f_ref: {res['f_ref']:.6f} Hz")
    print(f"A: {res['A']:.6f} ± {res['sigA']:.6f}")
    print(f"phi: {res['phi']:.6f} rad (± {res['sigPhi']:.6f})")
    print(f"N photons: {res['N']:,}")

if __name__ == "__main__":
    main()
