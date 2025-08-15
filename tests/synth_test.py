import numpy as np
from lockin import phasor_lockin_ps

def make_ref_ps(f_hz=10_000.0, n_edges=20_000, t0_ps=0.0):
    T_ps = 1e12 / f_hz
    return (t0_ps + np.arange(n_edges) * T_ps).astype(np.int64)

def synth_photons_sine(t_ref_ps, N=50_000, A=0.30, phi=0.5, rng=None):
    """I(θ) ∝ 1 + A cos(θ - φ); returns photon times in ps."""
    if rng is None: rng = np.random.default_rng(0)
    T_ps = float(np.median(np.diff(t_ref_ps)))
    # draw uniform phases, accept-reject with weight 1 + A cos(θ-φ)
    theta = 2*np.pi*rng.random(N*2)
    w = 1 + A*np.cos(theta - phi)
    u = rng.random(theta.size)
    theta = theta[u < w / w.max()][:N]
    # map phases to times by picking a random cycle
    k = rng.integers(0, t_ref_ps.size-1, size=theta.size)
    t = t_ref_ps[k] + (theta/(2*np.pi))*T_ps
    return t.astype(np.int64)

def synth_photons_square(t_ref_ps, N=50_000, duty=0.5, phi=0.0, rng=None):
    """On/off gate of width D; A_theory = 2 sin(pi D)/(pi D)."""
    if rng is None: rng = np.random.default_rng(1)
    T_ps = float(np.median(np.diff(t_ref_ps)))
    # phases uniform inside an open window centered at phi with width 2π D
    half = np.pi * duty
    theta = (phi - half) + 2*half*rng.random(N)
    # wrap to [0, 2π)
    theta = (theta + 2*np.pi) % (2*np.pi)
    # map to times
    k = rng.integers(0, t_ref_ps.size-1, size=theta.size)
    t = t_ref_ps[k] + (theta/(2*np.pi))*T_ps
    return t.astype(np.int64)

def check_sine():
    f = 10_000.0
    A_true, phi_true = 0.30, 0.70
    t_ref = make_ref_ps(f, 20_000)
    t_ph  = synth_photons_sine(t_ref, N=80_000, A=A_true, phi=phi_true)
    res = phasor_lockin_ps(t_ph, t_ref)
    print("\nSINE TEST")
    print(f"f_ref: {res['f_ref']:.2f} Hz (true {f:.2f})")
    print(f"A: {res['A']:.4f} (true {A_true:.4f})  |  phi: {res['phi']:.4f} (true {phi_true:.4f})")
    print(f"σA pred: {res['sigA']:.4f}, σφ pred: {res['sigPhi']:.4f} rad")
    assert abs(res["f_ref"] - f)/f < 1e-3
    assert abs(res["A"] - A_true) < 0.03
    assert abs((res["phi"] - phi_true + np.pi)%(2*np.pi) - np.pi) < 0.05

def check_square():
    f = 10_000.0
    D = 0.5; phi_true = 0.25
    A_theory = 2*np.sin(np.pi*D)/(np.pi*D)  # → 4/π ≈ 1.273 for D=0.5
    t_ref = make_ref_ps(f, 20_000)
    t_ph  = synth_photons_square(t_ref, N=80_000, duty=D, phi=phi_true)
    res = phasor_lockin_ps(t_ph, t_ref)
    print("\nSQUARE TEST (D=0.5)")
    print(f"A: {res['A']:.4f} (theory {A_theory:.4f})  |  phi: {res['phi']:.4f} (true {phi_true:.4f})")
    assert abs(res["A"] - A_theory) < 0.03
    assert abs((res["phi"] - phi_true + np.pi)%(2*np.pi) - np.pi) < 0.05

if __name__ == "__main__":
    check_sine()
    check_square()
    print("\nOK ✓")
