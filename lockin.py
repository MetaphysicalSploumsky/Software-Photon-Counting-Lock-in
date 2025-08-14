# (inputs in picoseconds)
import numpy as np
from typing import Dict

def phasor_lockin_ps(t_ph_ps: np.ndarray, t_ref_ps: np.ndarray) -> Dict[str, float]:
    if t_ph_ps.size == 0 or t_ref_ps.size < 2:
        raise ValueError("Not enough photons or reference edges.")
    z = np.arange(t_ref_ps.size)
    A = np.vstack([z, np.ones_like(z)]).T
    T_ps, t0_ps = np.linalg.lstsq(A, t_ref_ps.astype(float), rcond=None)[0]
    theta = 2*np.pi * (((t_ph_ps - t0_ps) / T_ps) % 1.0)
    Z = np.exp(1j*theta).sum()
    N = theta.size
    Arel = 2*np.abs(Z)/N
    phi  = float(np.angle(Z))
    return {
        "A": float(Arel),
        "phi": phi,
        "sigA": float(1/np.sqrt(N)),
        "sigPhi": float(2*np.pi/np.sqrt(N)),
        "N": int(N),
        "f_ref": float(1e12 / T_ps),
        "T_ps": float(T_ps),
    }
