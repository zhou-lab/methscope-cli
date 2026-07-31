"""Reader for the MSFMAT1 feature artifact written by `classify-featurize`.

Mirrors analysis/msur.py: mmap the file, expose the sections as numpy views,
and decode nothing eagerly. See src/msfm.h for the layout.
"""
import numpy as np

HDR = np.dtype([
    ("magic", "S8"), ("version", "<u4"), ("n_records", "<u4"),
    ("n_patterns", "<u4"), ("n_classes", "<u4"), ("flags", "<u4"),
    ("reserved", "<u4"), ("names_offset", "<u8"), ("rows_offset", "<u8"),
    ("labels_offset", "<u8"), ("levels_offset", "<u8"), ("beta_offset", "<u8"),
    ("count_offset", "<u8"), ("file_bytes", "<u8"),
])

F_COUNTS = 1
NA = 65535          # matches MSFM_NA


def _names(raw, off, n):
    """n NUL-terminated strings starting at off."""
    out, p = [], int(off)
    for _ in range(n):
        e = raw.index(b"\0", p)
        out.append(raw[p:e].decode())
        p = e + 1
    return out


class Msfm:
    def __init__(self, path):
        self.raw = np.memmap(path, dtype="u1", mode="r").tobytes()
        h = np.frombuffer(self.raw, HDR, count=1)[0]
        if not h["magic"].startswith(b"MSFMAT1") or h["version"] != 1:
            raise ValueError(f"{path}: not a MSFMAT1 v1 artifact")
        self.h = h
        nr, np_ = int(h["n_records"]), int(h["n_patterns"])
        self.n_records, self.n_patterns = nr, np_
        self.patterns = _names(self.raw, h["names_offset"], np_)
        self.records = _names(self.raw, h["rows_offset"], nr)
        self.class_id = np.frombuffer(self.raw, "<u2", nr, int(h["labels_offset"]))
        self.classes = _names(self.raw, int(h["labels_offset"]) + nr * 2,
                              int(h["n_classes"]))
        self.levels = np.frombuffer(self.raw, "<u4", nr, int(h["levels_offset"]))
        self._beta = np.frombuffer(self.raw, "<u2", nr * np_,
                                   int(h["beta_offset"])).reshape(nr, np_)
        self.counts = (np.frombuffer(self.raw, "<u4", nr * np_,
                                     int(h["count_offset"])).reshape(nr, np_)
                       if h["flags"] & F_COUNTS else None)

    @property
    def labels(self):
        """One label string per record ([] when the artifact is unlabeled)."""
        return [self.classes[i] for i in self.class_id] if self.classes else []

    def beta(self):
        """Decoded betas as float64 with NA -> NaN (u16 code / 65534)."""
        b = self._beta.astype(np.float64) / 65534.0
        b[self._beta == NA] = np.nan
        return b
