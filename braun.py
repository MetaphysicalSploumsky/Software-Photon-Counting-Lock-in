# braun_cli.py
import argparse
import numpy as np
from typing import Tuple
from reader import read_ptu  # must return (header, times_by_channel, markers, reader_name) with times in ps

# ---------- Ref frequency from markers (ps → Hz) ----------

def compute_ref_freq(modulation_timestamps):
    # we want one timestamp per period
    # linear regression of modulation_timestamps vs [0, 1, ...]
    # timestamps are in picoseconds → slope = period_ps → freq = 1e12 / period_ps (Hz)
    t = np.asarray(modulation_timestamps, dtype=float)
    n = t.size
    if n < 2:
        raise ValueError("compute_ref_freq: need at least two timestamps.")

    z = np.arange(n, dtype=float)

    zc = z - z.mean()
    tc = t - t.mean()
    denom = float(zc @ zc)
    if denom == 0:
        raise ValueError("compute_ref_freq: degenerate reference indices.")
    period_ps = float((zc @ tc) / denom)  # slope (ps per period)

    if not np.isfinite(period_ps) or period_ps <= 0:
        raise ValueError("compute_ref_freq: invalid period estimated.")

    f_hz = 1e12 / period_ps  # ps → s
    return f_hz

# ---------- Braun binning (seconds) ----------

def bin_photons_braun(photon_timestamps_s, ref_freq, acquisition_time_s, oversampling_factor=50):
    """
    Bin photons according to Braun method with proper oversampling (uniform time bins).
    Inputs:
      photon_timestamps_s : array of photon arrival times in **seconds**, relative to start (>=0)
      ref_freq            : reference frequency in Hz
      acquisition_time_s  : total span to cover (seconds)
    Returns:
      binned_counts (int per bin), bin_centers_s (seconds), time_bin_width_s
    """
    sampling_freq = oversampling_factor * ref_freq  # f_sample = M * f_ref
    time_bin_width = 1.0 / sampling_freq           # seconds

    if acquisition_time_s <= 0:
        raise ValueError("bin_photons_braun: acquisition_time must be > 0.")
    if sampling_freq <= 0:
        raise ValueError("bin_photons_braun: sampling frequency must be > 0.")

    # Number of bins (include any partial at the end)
    num_bins = int(np.ceil(acquisition_time_s / time_bin_width))
    # Bin edges (0 .. acquisition_time_s)
    # (avoid building huge arrays for very long runs with big oversampling)
    bin_edges = np.linspace(0.0, acquisition_time_s, num_bins + 1, dtype=float)

    # Histogram photons into uniform bins
    binned_counts, _ = np.histogram(photon_timestamps_s, bins=bin_edges)

    # Bin centers for reference evaluation
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return binned_counts.astype(float), bin_centers, time_bin_width

# ---------- Braun lock-in on evenly sampled counts ----------

def braun_lock_in(binned_counts, bin_centers_s, ref_freq, phase_offset=0.0, basis="sine"):
    """
    Implement Braun lock-in detection (eq. 2 style) on evenly-sampled count trace.
    Inputs:
      binned_counts : counts per bin (float or int)
      bin_centers_s : time of each bin center (seconds)
      ref_freq      : Hz
      basis         : 'sine' or 'square' reference
    Returns:
      amplitude | phase | I_x | I_y | I_dc
    """
    binned_counts = np.asarray(binned_counts, float)
    bin_centers_s = np.asarray(bin_centers_s, float)
    if binned_counts.size != bin_centers_s.size or binned_counts.size < 8:
        raise ValueError("braun_lock_in: mismatched or too few samples.")

    # Remove DC
    I_dc = float(binned_counts.mean())
    N_t = binned_counts - I_dc

    omega = 2.0 * np.pi * float(ref_freq)
    t = bin_centers_s
    phi0 = float(phase_offset)

    if basis == "sine":
        R_x = np.sin(omega * t + phi0)
        R_y = np.cos(omega * t + phi0)
        # Original normalization (Braun): multiply by sqrt(2) / sqrt(mean(R^2))
        numerator_x = float(np.mean(N_t * R_x))
        numerator_y = float(np.mean(N_t * R_y))
        denom_x = float(np.sqrt(np.mean(R_x**2)))
        denom_y = float(np.sqrt(np.mean(R_y**2)))
        I_x = np.sqrt(2.0) * numerator_x / denom_x
        I_y = np.sqrt(2.0) * numerator_y / denom_y

    elif basis == "square":
        # ±1 references from the sign of the sin/cos
        R_x = np.where(np.sin(omega * t + phi0) >= 0.0, 1.0, -1.0)
        R_y = np.where(np.cos(omega * t + phi0) >= 0.0, 1.0, -1.0)
        numerator_x = float(np.mean(N_t * R_x))
        numerator_y = float(np.mean(N_t * R_y))
        # mean(R^2) is 1 for ±1 wave; Braun's scaling factor for 1f of a square is π/2
        I_x = (np.pi / 2.0) * numerator_x
        I_y = (np.pi / 2.0) * numerator_y
    else:
        raise ValueError("braun_lock_in: basis must be 'sine' or 'square'.")

    # Complex demodulated signal
    delta_I_over_I = (I_x + 1j * I_y) / I_dc
    amplitude = float(np.abs(delta_I_over_I))
    phase = float(np.angle(delta_I_over_I))
    return amplitude, phase, float(I_x), float(I_y), float(I_dc)

# ---------- End-to-end pipeline from PTU ----------

def braun_pipeline(
    data_path: str,
    basis: str = "sine",
    oversampling_factor: int = 50,
    ch_ph: int = 1,
    marker_bit: int = 2,
    ref_every_n: int = 1,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Load PTU, build reference from markers, bin photons per Braun, run lock-in.
    Returns:
      amplitude, phase, I_x, I_y, I_dc, f_ref (Hz), time_bin_width (s)
    """
    # Load data (picoseconds)
    header, ch, markers, reader = read_ptu(data_path)
    photon_ps = ch.get(ch_ph, np.array([], dtype=np.int64))
    if photon_ps.size == 0:
        raise ValueError("No photons on the selected channel.")

    # Build reference timestamps from marker bit (picoseconds)
    mask = 1 << int(marker_bit)
    ref_ps = np.asarray([tps for (mk, tps) in markers if (mk & mask) != 0], dtype=np.int64)
    if ref_ps.size < 2:
        raise ValueError("Not enough reference markers with the selected bit.")

    # If your marker stream includes more than one edge per cycle, keep every Nth
    if ref_every_n > 1:
        ref_ps = ref_ps[::int(ref_every_n)]

    # Compute reference frequency (Hz) from markers (expects ps)
    f_ref = compute_ref_freq(ref_ps)

    # Time-zero alignment and unit conversion to seconds
    t0_ps = min(int(photon_ps[0]), int(ref_ps[0]))
    photon_s = (photon_ps - t0_ps) * 1e-12
    # Only use photons within the reference span (optional but robust)
    lo_s = (ref_ps[0] - t0_ps) * 1e-12
    hi_s = (ref_ps[-1] - t0_ps) * 1e-12
    photon_s = photon_s[(photon_s >= lo_s) & (photon_s <= hi_s)]

    if photon_s.size == 0:
        raise ValueError("No photons within the reference time span.")

    acquisition_time_s = float(photon_s.max())  # span from 0 .. max photon time

    # Bin photons with proper oversampling
    binned_counts, bin_centers_s, time_bin_width_s = bin_photons_braun(
        photon_s, f_ref, acquisition_time_s, oversampling_factor=oversampling_factor
    )

    # Braun lock-in on the evenly-sampled trace
    amplitude, phase, I_x, I_y, I_dc = braun_lock_in(
        binned_counts, bin_centers_s, f_ref, basis=basis
    )

    return amplitude, phase, I_x, I_y, I_dc, float(f_ref), float(time_bin_width_s)

# ---------- CLI ----------

def _main():
    ap = argparse.ArgumentParser(
        description="Braun lock-in pipeline: PTU → bins (oversampled) → demodulation."
    )
    ap.add_argument("ptu", type=str, help="Path to .ptu file")
    ap.add_argument("--basis", type=str, default="sine", choices=["sine", "square"],
                    help="Reference basis used for demodulation (default: sine)")
    ap.add_argument("--oversample", type=int, default=50, help="Oversampling factor M (default 50)")
    ap.add_argument("--ch-ph", type=int, default=1, help="Photon channel (default 1)")
    ap.add_argument("--marker-bit", type=int, default=2, help="Marker bit to use for reference (default 2)")
    ap.add_argument("--ref-every-n", type=int, default=1,
                    help="Keep every Nth marker (set 2 if you have both edges)")
    args = ap.parse_args()

    amp, phi, Ix, Iy, Idc, f_ref, dt = braun_pipeline(
        data_path=args.ptu,
        basis=args.basis,
        oversampling_factor=args.oversample,
        ch_ph=args.ch_ph,
        marker_bit=args.marker_bit,
        ref_every_n=args.ref_every_n,
    )

    samples_per_period = (args.oversample,)
    print(f"File: {args.ptu}")
    print(f"Ref frequency: {f_ref:.6f} Hz")
    print(f"Oversampling factor M: {args.oversample}  → f_sample = {args.oversample*f_ref:.3f} Hz")
    print(f"Time-bin width: {dt*1e6:.3f} µs")
    print(f"Demod basis: {args.basis}")
    print(f"I_dc: {Idc:.6f}")
    print(f"I_x:  {Ix:.6f}")
    print(f"I_y:  {Iy:.6f}")
    print(f"Amplitude (|ΔI/I|): {amp:.6f}")
    print(f"Phase (rad):        {phi:.6f}")
    if args.basis == "square":
        # For 50% duty square-wave, contrast ≈ (π/4)*Amplitude
        print(f"Square-wave note: contrast ≈ (π/4)*A = {(np.pi/4)*amp:.6f}")

if __name__ == "__main__":
    _main()
# python braun.py data/laserON_modulated_200s.ptu --ch-ph 1 --marker-bit 1 --basis sine --oversample 50