# reader.py — Unified reader for PicoHarp 300/330 TTTR T2 (.ptu)
# Returns timestamps (int64 picoseconds) as:
#     marker_times_ps, photon_times_ps = read_ptu(filepath)
#
# For PicoHarp 330 (Generic T2 records):
#   - We treat "sync" events (special=1, channel==0) as markers (your setup feeds markers on SYNC).
#   - We take detector photons from input channel 1 (i.e., special=0 with raw channel==0 -> ch 1).
#
# For PicoHarp 300 (legacy T2 records):
#   - Photons: channel 1
#   - Markers: special records (channel 0xF) with marker bits 1..4 set.
#
# Both return arrays are numpy int64 of absolute picoseconds from the run start.
#
# Based on PicoQuant demo layout "Read_PTU.py" (header/tag constants and record layouts).

from __future__ import annotations
import io, struct
from typing import Dict, Tuple, List
import numpy as np

def _beint(hexs: str) -> int:
    return struct.unpack(">i", bytes.fromhex(hexs))[0]

# Tag types (we only need to skip/parse a subset)
TY_EMPTY8      = _beint("FFFF0008")
TY_BOOL8       = _beint("00000008")
TY_INT8        = _beint("10000008")
TY_BITSET64    = _beint("11000008")
TY_COLOR8      = _beint("12000008")
TY_FLOAT8      = _beint("20000008")
TY_TDATETIME   = _beint("21000008")
TY_FLOAT8ARRAY = _beint("2001FFFF")
TY_ANSISTRING  = _beint("4001FFFF")
TY_WIDESTRING  = _beint("4002FFFF")
TY_BINARYBLOB  = _beint("FFFFFFFF")

# Record types
RT_PH300_T2    = _beint("00010203")
RT_GENERIC_T2  = _beint("00010207")  # MultiHarp + PicoHarp 330

def _read_header(f) -> Dict[str, object]:
    # Check magic
    magic = f.read(8).decode('utf-8', 'ignore')
    if magic != "PQTTTR\0\0":
        raise ValueError("Not a PTU TTTR file (bad magic)")
    version = f.read(8).decode('utf-8', 'ignore')  # e.g. "2.0.0\0\0"
    header: Dict[str, object] = {"__version__": version}
    # Iterate tagged header until Header_End
    while True:
        ident = f.read(32).decode('utf-8', 'ignore').strip('\0')
        idx   = struct.unpack("<i", f.read(4))[0]
        typ   = struct.unpack("<i", f.read(4))[0]
        if typ in (TY_EMPTY8, TY_BOOL8, TY_INT8, TY_BITSET64, TY_COLOR8):
            val = struct.unpack("<q", f.read(8))[0]
        elif typ in (TY_FLOAT8, TY_TDATETIME):
            val = struct.unpack("<d", f.read(8))[0]
        elif typ == TY_ANSISTRING:
            n = struct.unpack("<q", f.read(8))[0]
            val = f.read(n).decode('utf-8', 'ignore').strip('\0')
        elif typ == TY_WIDESTRING:
            n = struct.unpack("<q", f.read(8))[0]
            val = f.read(n).decode('utf-16le', 'ignore').strip('\0')
        elif typ in (TY_FLOAT8ARRAY, TY_BINARYBLOB):
            n = struct.unpack("<q", f.read(8))[0]
            f.seek(n, io.SEEK_CUR)
            val = n  # store length
        else:
            raise ValueError(f"Unknown header tag type 0x{typ:08X}")

        key = ident if idx == -1 else f"{ident}({idx})"
        header[key] = val
        if ident == "Header_End":
            break
    return header

def read_ptu(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a .ptu TTTR (T2) file and return (marker_times_ps, photon_times_ps).

    For PicoHarp 330 Generic T2: markers are taken from SYNC events (channel 0).
    For PicoHarp 300 T2: markers are taken from special records' marker bits.
    In both cases, photons are taken from detector input channel 1.
    """
    with open(filepath, "rb") as f:
        header = _read_header(f)

        rec_type  = int(header["TTResultFormat_TTTRRecType"]) # type: ignore
        n_records = int(header["TTResult_NumberOfRecords"]) # type: ignore
        globRes_s = float(header["MeasDesc_GlobalResolution"]) # type: ignore
        scale_ps  = int(round(globRes_s * 1e12))  # convert time tags to picoseconds

        # Output buffers (python lists for speed, convert at end)
        markers_ps: List[int] = []
        photons_ps: List[int] = []

        # Dispatch by record type
        if rec_type == RT_PH300_T2:
            # Layout: [3:0]=channel (0..4, F=special), [31:4]=time
            T2WRAPAROUND = 210698240  # from PicoQuant demo
            ofl = 0
            for _ in range(n_records):
                try:
                    word = struct.unpack("<I", f.read(4))[0]
                except Exception as e:
                    break
                channel = (word >> 28) & 0xF
                timetag = word & 0x0FFFFFFF
                if channel == 0xF:
                    # Special: overflow or markers
                    markers_bits = (word & 0xF)  # lower 4 bits of time are markers
                    if markers_bits == 0:
                        ofl += T2WRAPAROUND  # single overflow
                    else:
                        # Treat any marker bit (1..4) as "a marker event" on our single timeline.
                        t_ps = (ofl + timetag) * scale_ps
                        markers_ps.append(t_ps)
                else:
                    # Regular photon event; take only detector channel 1
                    if channel == 1:
                        t_ps = (ofl + timetag) * scale_ps
                        photons_ps.append(t_ps)

        elif rec_type == RT_GENERIC_T2:
            # Layout ("HT2 v2"): special(1), channel(6), timetag(25)
            # Overflows: special=1, channel=0x3F, timetag = count (if 0 => single old-style)
            # Markers:   special=1, channel in 1..15  (we'll also accept these as markers)
            # SYNC:      special=1, channel==0        (we treat these as markers in your setup)
            # Photons:   special=0, channel = raw 0..N-1  -> logical channel = raw+1 (take == 1)
            T2WRAP_V1 = 33552000
            T2WRAP_V2 = 33554432
            ofl = 0
            for _ in range(n_records):
                try:
                    word = struct.unpack("<I", f.read(4))[0]
                except Exception:
                    break
                special = (word >> 31) & 0x1
                channel = (word >> 25) & 0x3F
                timetag = (word & 0x01FFFFFF)
                if special == 1:
                    if channel == 0x3F:
                        # Overflow(s)
                        if timetag == 0:
                            ofl += T2WRAP_V2  # old style single overflow (or v1 value)
                        else:
                            ofl += T2WRAP_V2 * timetag
                    else:
                        # Marker or sync -> treat as marker event
                        t_ps = (ofl + timetag) * scale_ps
                        # Only collect SYNC (channel==0) or generic markers (1..15)
                        if channel == 0 or (1 <= channel <= 15):
                            markers_ps.append(t_ps)
                else:
                    # Photon on an input channel: raw 0.. -> logical 1..
                    logical_ch = channel + 1
                    if logical_ch == 1:
                        t_ps = (ofl + timetag) * scale_ps
                        photons_ps.append(t_ps)
        else:
            raise NotImplementedError(
                f"Unsupported TTTR record type 0x{rec_type:08X}. "
                "This reader supports PicoHarp 300 T2 and Generic T2 (PicoHarp 330)."
            )

    # Convert to numpy arrays (sorted order as in file)
    marker_times = np.asarray(markers_ps, dtype=np.int64)
    photon_times = np.asarray(photons_ps, dtype=np.int64)
    return marker_times, photon_times

# Simple CLI for quick testing:
if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="Extract markers (chn0/sync) and photons (chn1) from PTU T2.")
    ap.add_argument("ptu", help="Path to .ptu file")
    args = ap.parse_args()
    m, p = read_ptu(args.ptu)
    print(f"File: {os.path.basename(args.ptu)}")
    print(f"Markers (chn0/sync): {len(m)}   Photons (chn1): {len(p)}")
    if len(m) and len(p):
        dur_ps = max(m[-1], p[-1]) - min(m[0], p[0])
        print(f"Span: {dur_ps/1e12:.3f} s")
