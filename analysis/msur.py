"""MSURAW2/3 reader, mirroring src/msur.h so C and Python cannot drift.

MSURAW2 records are fixed-width: one `sampled_per_cell` and one `record_bytes`
in the header, so a record is `records_offset + row * record_bytes`.  MSURAW3
lets sparsity vary per replicate — every record *within* a replicate is still
the same size, so O(1) addressing survives via a 24-byte-per-replicate table
instead of padding short records out to the widest level.

Use `Msur(path)` and its `.record(rep, cell)` / `.sample_of(rep)`; both formats
answer the same way, so callers need no version branch.
"""
import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8s4IQ2I4Q")      # 72-byte prefix, identical in v2/v3
REP = np.dtype([("sample", "<u4"), ("flags", "<u4"),
                ("record_bytes", "<u8"), ("offset", "<u8")])

F_TRUTH_U16 = 1
F_BINARIZED = 2

## msur_rep_t.flags -- observed-set encoding, chosen per replicate by whichever
## is smaller: a sorted u32 list (4B per observed CpG) or a whole-genome bitmap.
ENC_LIST = 0
ENC_BITMAP = 1
F_MIXED_SAMPLE = 4


class Msur:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            head = f.read(80)
        (magic, self.version, self.n_cells, self.n_reps, self.patterns,
         self.n_cpg, self.sampled_per_cell, self.flags, self.groups_offset,
         self.truth_offset, self.records_offset,
         self.record_bytes) = HEADER.unpack(head[:72])
        if magic == b"MSURAW3\0" and self.version == 3:
            self.v3 = True
        elif magic == b"MSURAW2\0" and self.version == 2:
            self.v3 = False
        else:
            raise ValueError(f"{path}: not a MSURAW2/3 msur")

        self.raw = np.memmap(self.path, "u1", "r")
        self.truth = np.memmap(self.path, "<u2", "r", self.truth_offset,
                               (self.n_cells, self.n_cpg)) \
            if self.truth_offset else None
        self.groups = np.memmap(self.path, "<u2", "r", self.groups_offset,
                                (self.n_cpg,))
        if self.v3:
            rep_table_offset, = struct.unpack("<Q", head[72:80])
            self.reps = np.memmap(self.path, REP, "r", rep_table_offset,
                                  (self.n_reps,))
        else:
            self.reps = None

    @property
    def embedded_truth(self):
        return bool(self.flags & F_TRUTH_U16) and self.truth_offset

    @property
    def binarized(self):
        return bool(self.flags & F_BINARIZED)

    def sample_of(self, rep):
        """Observed CpGs per cell in this replicate."""
        return int(self.reps[rep]["sample"]) if self.v3 \
            else int(self.sampled_per_cell)

    def encoding_of(self, rep):
        """ENC_LIST or ENC_BITMAP -- how this replicate stores its observed
        set.  v2, and v3 files written before the bitmap existed, are lists."""
        return int(self.reps[rep]["flags"]) if self.v3 else ENC_LIST

    def _offset(self, rep, cell):
        if self.v3:
            r = self.reps[rep]
            return int(r["offset"]) + cell * int(r["record_bytes"])
        return self.records_offset + \
            (rep * self.n_cells + cell) * self.record_bytes

    def record(self, rep, cell):
        """(beta, count, selected) views for one (replicate, cell) row.

        `selected` is always a sorted u32 array of observed CpG indices; when
        the replicate is bitmap-encoded it is materialized from the bits, so
        callers see one representation regardless of how it was stored."""
        o = self._offset(rep, cell)
        P, n = self.patterns, self.sample_of(rep)
        beta = np.ndarray((P,), "<f4", self.raw, o)
        count = np.ndarray((P,), "<u4", self.raw, o + P * 4)
        if self.encoding_of(rep) == ENC_BITMAP:
            nb = (self.n_cpg + 7) // 8
            bits = np.ndarray((nb,), "u1", self.raw, o + P * 8)
            selected = np.flatnonzero(
                np.unpackbits(bits, bitorder="little")[:self.n_cpg]).astype("<u4")
        else:
            selected = np.ndarray((n,), "<u4", self.raw, o + P * 8)
        return beta, count, selected

    def levels(self):
        """[(sample, n_replicates)] in replicate order."""
        if not self.v3:
            return [(int(self.sampled_per_cell), int(self.n_reps))]
        out = []
        for s in np.asarray(self.reps["sample"]):
            if out and out[-1][0] == s:
                out[-1] = (out[-1][0], out[-1][1] + 1)
            else:
                out.append((int(s), 1))
        return out
