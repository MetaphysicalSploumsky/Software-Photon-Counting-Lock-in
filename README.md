# Single-Photon Software Lock-In & FDLM Pipeline

## Project Overview

This project provides a robust, event-based software implementation of a lock-in amplifier and a Frequency-Domain Lifetime Measurement (FDLM) pipeline tailored for high-speed single-photon counting data (PicoHarp time-tags). It enables the efficient recovery of both **relative intensity modulation (Magnitude)** and **phase shift (Lifetime)** from weak, shot-noise-limited Photoluminescence (PL) signals.

## Features

* **Event-Based Lock-In:** Implements a digital lock-in algorithm directly on PicoHarp time-tag data (PTU format) to extract AC-modulated signal components[cite: 14].
* **Performance Benchmarking:** Includes signal-to-noise ratio (SNR) benchmarks to define regimes where the lock-in approach is superior to simple sequential ON/OFF subtraction[cite: 15, 16].
* **Frequency-Domain Analysis:** Full FDLM pipeline for fitting frequency-domain lifetime data[cite: 17].
* **Data Diagnostics:** Incorporates $\chi^{2}$ diagnostics and per-frequency phase-offset calibration (from fast scatter) to ensure fit quality and accuracy[cite: 17].

## Installation

### Prerequisites

You must have the following installed:

* Python 3.x
* The necessary PicoQuant libraries or utilities to access the PTU file format (if applicable, specify which ones).

### Via PyPI (Recommended - If Packaged)

```bash
pip install photon-lockin-fdlm
```

### From Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/Metaphysical Sploumsky/photon-lockin-fdlm.git
    cd photon-lockin-fdlm
    ```
2.  Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1\. Lock-In Processing

Import the lock-in module and process a PTU file:

```python
from lockin.processor import run_lock_in
import numpy as np

# Load PTU data (e.g., using a custom loader or PicoQuant utility)
time_tags, markers = load_ptu_data('sample_file.ptu') 
[cite_start]modulation_freq = 100000 # 100 kHz [cite: 15]

# Recover magnitude and phase for the specific modulation frequency
magnitude, phase = run_lock_in(time_tags, markers, modulation_freq)

print(f"Recovered Magnitude: {magnitude}")
print(f"Recovered Phase: {phase} radians")
```

### 2\. FDLM Fitting Pipeline

After processing across a range of frequencies, use the FDLM module for lifetime fitting:

```python
from fdlm.fitter import fit_fdlm

# Example data (Frequency, Magnitude, Phase, Error)
freq_data = np.array([...])
# ... load or process all your frequency data ...

# Perform the fit (e.g., to a single-exponential model)
best_fit_params, chi_squared = fit_fdlm(freq_data, model='single_exp')

print(f"Best-fit lifetime: {best_fit_params['tau']} ns")
[cite_start]print(f"Chi-Squared / DoF: {chi_squared}") # The chi-squared diagnostics [cite: 17]
```

## Results & Benchmarks

This section is vital for showing the project's impact. Use an image tag for a visual showcase.

### SNR Performance

The software lock-in was benchmarked against traditional ON/OFF subtraction methods, demonstrating its effectiveness in different signal-to-noise regimes.

| Metric | Lock-In Performance | ON/OFF Subtraction Performance |
| :--- | :--- | :--- |
| **Operating Range** | [cite\_start]1 kHz to 1 MHz modulation [cite: 15] | Low-frequency only |
| **SNR Improvement** | **X% improvement** in shot-noise-limited regimes. (Replace X with actual number) | Baseline |


## License

This project is licensed under the [Choose a License, e.g., MIT License] - see the `LICENSE.md` file for details.

## Contact

[cite\_start]Amine Sahraoui: aminesahraouics@outlook.com [cite: 2]
[cite\_start]GitHub: [github.com/Metaphysical](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/Metaphysical) Sploumsky [cite: 2]