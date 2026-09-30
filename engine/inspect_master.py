"""Measure a rendered WAV file. Usage: python3 engine/inspect_master.py FILE"""
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A


def read_wav(path):
    """Read 16- or 24-bit PCM WAV into a float array in [-1, 1]."""
    with wave.open(path, "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        # sign-extend 24-bit little-endian into int32
        ints = (b[:, 0].astype(np.int32)
                | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float64) / 8388607.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    return data.reshape(-1, nch), sr


if __name__ == "__main__":
    path = sys.argv[1]
    x, sr = read_wav(path)
    print(A.report(x, sr, os.path.basename(path)))

    print("\n  short-term loudness by section:")
    import composition as C
    st = dict(A.lufs_short_term(x, sr, 3.0, 0.5))
    times = np.array(sorted(st))
    for s in C.SECTIONS:
        t0 = s.start * 4 * 60 / C.BPM
        t1 = s.end * 4 * 60 / C.BPM
        vals = [st[t] for t in times if t0 + 1.5 <= t <= t1 - 1.5]
        if vals:
            print(f"    {s.name:8s} {int(t0)//60}:{int(t0)%60:02d}-"
                  f"{int(t1)//60}:{int(t1)%60:02d}   "
                  f"{np.mean(vals):6.2f} LUFS  (peak {np.max(vals):6.2f})")
