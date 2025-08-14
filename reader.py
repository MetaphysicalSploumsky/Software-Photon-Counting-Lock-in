# reader.py — PicoHarp 300 T2 to (channel_times_ps, markers)
import io, struct
from typing import Dict, List, Tuple
import numpy as np

def _beint(hexs: str) -> int:
    return struct.unpack(">i", bytes.fromhex(hexs))[0]

# Tag / record type ids (only those we need)
TY_EMPTY8=_beint("FFFF0008"); TY_BOOL8=_beint("00000008"); TY_INT8=_beint("10000008")
TY_BITSET64=_beint("11000008"); TY_COLOR8=_beint("12000008"); TY_FLOAT8=_beint("20000008")
TY_TDATETIME=_beint("21000008"); TY_FLOAT8ARRAY=_beint("2001FFFF")
TY_ANSISTRING=_beint("4001FFFF"); TY_WIDESTRING=_beint("4002FFFF"); TY_BINARYBLOB=_beint("FFFFFFFF")
RT_PH300_T2 = _beint("00010203")

def read_ptu(filepath: str):
    """
    Minimal PH300 T2 reader.
    Returns:
      header : dict(tag->value)
      times_by_channel : dict[int -> np.ndarray[int64]]  (ps)
      markers : list[(marker_bits:int, truetime_ps:int)]
      reader_name : str
    """
    f = open(filepath, "rb")
    try:
        if f.read(8).decode("utf-8", "ignore").strip("\0") != "PQTTTR":
            raise ValueError("Not a PTU file")
        _ = f.read(8)  # version (unused)

        # header 
        header = {}
        while True:
            ident = f.read(32).decode("utf-8", "ignore").strip("\0")
            idx   = struct.unpack("<i", f.read(4))[0]
            typ   = struct.unpack("<i", f.read(4))[0]
            if   typ in (TY_EMPTY8, TY_BOOL8, TY_INT8, TY_BITSET64, TY_COLOR8):
                val = struct.unpack("<q", f.read(8))[0]
            elif typ == TY_FLOAT8:
                val = struct.unpack("<d", f.read(8))[0]
            elif typ == TY_TDATETIME:
                val = struct.unpack("<d", f.read(8))[0]
            elif typ == TY_ANSISTRING:
                n = struct.unpack("<q", f.read(8))[0]; val = f.read(n).decode("utf-8","ignore").strip("\0")
            elif typ == TY_WIDESTRING:
                n = struct.unpack("<q", f.read(8))[0]; val = f.read(n).decode("utf-16le","ignore").strip("\0")
            elif typ in (TY_FLOAT8ARRAY, TY_BINARYBLOB):
                n = struct.unpack("<q", f.read(8))[0]; f.seek(n, io.SEEK_CUR); val = n
            else:
                raise ValueError("Unknown header tag")
            key = ident if idx == -1 else f"{ident}({idx})"
            header[key] = val
            if ident == "Header_End": break

        rec_type  = int(header["TTResultFormat_TTTRRecType"])
        n_records = int(header["TTResult_NumberOfRecords"])
        globRes_s = float(header["MeasDesc_GlobalResolution"])
        scale_ps  = int(round(globRes_s * 1e12))  # ~4 ps/tick

        if rec_type != RT_PH300_T2:
            raise NotImplementedError("reader handles PicoHarp 300 T2 only.")

        # data (PH300 T2) 
        T2WRAP = 210_698_240
        ofl = 0
        times_by_channel: Dict[int, List[int]] = {1:[],2:[],3:[],4:[]}
        markers: List[Tuple[int,int]] = []  # (marker_bits, truetime_ps)

        for _ in range(n_records):
            b = f.read(4)
            if len(b) < 4: break
            w = struct.unpack("<I", b)[0]
            chan    = (w >> 28) & 0xF
            timetag =  w & 0x0FFF_FFFF  # 28 bits
            if chan == 0xF:             # special
                mk = timetag & 0xF      # lowest 4 bits = marker bitmask
                if mk == 0:
                    ofl += T2WRAP       # overflow
                else:
                    truetime_ps = (ofl + timetag) * scale_ps
                    markers.append((mk, truetime_ps))
            else:                       # photon
                if 1 <= chan <= 4:
                    truetime_ps = (ofl + timetag) * scale_ps
                    times_by_channel[chan].append(truetime_ps)

        times_by_channel = {k: np.asarray(v, dtype=np.int64) for k,v in times_by_channel.items()}
        return header, times_by_channel, markers, "PicoHarp 300 T2"
    finally:
        f.close()
