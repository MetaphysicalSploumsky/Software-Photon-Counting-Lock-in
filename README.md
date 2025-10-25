Here's the updated README with the new image structure:

**README.md**

# Photon Counting Lock-in Analysis Tools

This repository contains a collection of Python tools for analyzing photon counting data from PicoHarp TCSPC systems using various lock-in detection methods. These scripts process PTU files containing timestamp data to extract modulated signals and perform frequency-domain lifetime measurements.

## Overview

The tools implement different photon counting lock-in techniques:

- **Braun & Libchaber method** (`braun_lockin.py`) - Traditional lock-in detection using binning and reference signals
- **Liu method** (`liu_lockin.py`) - Reference-weighted counting with quadrature square-wave demodulation  
- **Jakob phase analysis** (`phase_hist_jakob.py`) - Dynamic period fitting for precise phase determination
- **Frequency-domain lifetime** (`FDLM_1.py`) - Single-exponential lifetime fitting from multi-frequency modulation data

## Files

- `braun_lockin.py` - Braun-style lock-in amplitude/phase extraction
- `FDLM_1.py` - Frequency-domain lifetime measurement pipeline
- `liu_lockin.py` - Liu quadrature lock-in with square-wave references
- `phase_hist_jakob.py` - Per-photon phase histogram analysis
- `reader.py` - Unified PTU file reader for PicoHarp 300/330 data
- `requirements.txt` - Python dependencies

## Requirements

```bash
pip install -r requirements.txt
```

## Basic Usage

### Braun Lock-in Analysis
```bash
python braun_lockin.py data.ptu --M 50 --trim-ends 1
```

### Frequency-Domain Lifetime
```bash
python FDLM_1.py /path/to/ptu_folder --marker-bit 1 --photon-channel 1 --N 10
```

### Liu Lock-in Detection  
```bash
python liu_lockin.py data.ptu --N 10 --duty 0.5 --bin-sec 1.0
```

### Phase Histogram
```bash
python phase_hist_jakob.py data.ptu --marker-bit 1 --bins 90
```

## Input Data

All tools expect PicoHarp `.ptu` files (T2 time-tagged time-resolved format) with:
- Channel 0: Reference/marker signals (modulation sync)
- Channel 1: Photon detection events

The PTU reader supports both PicoHarp 300 (legacy T2) and PicoHarp 330 (Generic T2) formats.

## Output

Each tool provides:
- Modulation depth (A) and phase (φ) measurements
- Statistical summaries (photon counts, reference edges)
- CSV export options for further analysis
- Visualization (phase histograms, lifetime fits)

## Demo Results

### Frequency-Domain Lifetime Analysis
![FDLM Analysis](demo/fdlm_curve.png)
*Frequency-domain lifetime measurement showing phase vs modulation frequency. The curve fit demonstrates single-exponential decay behavior with τ = 4.2 ns.*

### Lifetime from Phase Histogram
![Lifetime from Phase](demo/lifetime_phase_histogram.png)
*Fluorescence lifetime extraction from phase histogram data. The modulation depth and phase shift provide complementary lifetime information.*

### Demultiplexing Capabilities
![Demultiplexing](demo/demultiplexing.png)
*Signal demultiplexing performance showing separation of multiple modulated sources using phase-sensitive detection.*

### Shot Noise Limited Performance
![Shot Noise Limit](demo/shot_noise_limit.png)
*System performance reaching the theoretical shot noise limit, demonstrating optimal signal-to-noise ratio for photon counting applications.*

### Noise Rejection
![Noise Rejection](demo/noise_rejection.png)
*Excellent noise rejection capabilities showing clean signal recovery even in low signal-to-noise conditions.*

### System Linearity
![Linearity](demo/linearity.png)
*Linear response of the lock-in system across different signal intensities, confirming quantitative measurement capability.*

### Phase Histogram Analysis
![Phase Histogram](demo/phase_histogram.png)
*High-resolution phase histogram showing photon distribution across modulation cycles with clear sinusoidal modulation pattern.*

## Sources

### Core Methods
- **Braun & Libchaber (2002)** - *"Trapping of DNA by dielectrophoretic cage in microfluidic arrays"* - Optics Letters 27, 1418 (2002)
- **Liu et al.** - Reference-weighted counting method for photon lock-in detection
- **Jakob phase analysis** - Dynamic linear fitting for robust period estimation against frequency drift

### Technical References
- **PicoHarp 300/330** - PicoQuant TCSPC system documentation and file formats
- **Frequency-domain lifetime** - Standard FD fluorescence lifetime measurement principles
- **Photon counting statistics** - Shot noise limits and signal processing in low-light conditions

### Implementation Notes
- PTU file format based on PicoQuant's "Read_PTU.py" reference implementation
- Some algorithms adapted from laboratory implementations and optimized for Python
- Error handling and edge cases tested with experimental data

## Notes

- Marker bits are zero-based (0-3) in all scripts
- The Jakob phase method provides robust period estimation against frequency drift
- Frequency-domain lifetime fitting supports both phase-only and joint amplitude-phase methods
- Liu lock-in works with adjustable duty cycle square waves

These tools were developed for analyzing modulated fluorescence signals in time-correlated single photon counting experiments.

---