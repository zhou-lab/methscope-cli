#!/usr/bin/env python3
"""Sampled continuous evaluation for a bare or bundled UPDEC2 model."""
import argparse
import math
import struct
from pathlib import Path

import numpy as np

MASK64 = (1 << 64) - 1
PCG_INC = 1442695040888963407


class Pcg32:
    def __init__(self, seed):
        self.state = 0
        self.inc = PCG_INC
        self.next()
        self.state = (self.state + seed) & MASK64
        self.next()

    def next(self):
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & MASK64
        x = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        r = old >> 59
        return ((x >> r) | (x << ((-r) & 31))) & 0xFFFFFFFF


def source_split(n, seed):
    cells = list(range(n))
    rng = Pcg32(seed)
    for q in range(n, 1, -1):
        j = rng.next() % q
        cells[q - 1], cells[j] = cells[j], cells[q - 1]
    nt, nv = n * 70 // 100, n * 15 // 100
    return cells[:nt], cells[nt:nt + nv], cells[nt + nv:]


def model_section(path):
    size = path.stat().st_size
    with path.open("rb") as f:
        magic = f.read(8)
        if magic == b"UPDEC2\0\0":
            return 0, size
        f.seek(size - 8)
        container, = struct.unpack("<Q", f.read(8))
        f.seek(container)
        if f.read(8) != b"MSBNDL1\0":
            raise ValueError("not UPDEC2 or MSBNDL1")
        n, = struct.unpack("<I", f.read(4))
        for _ in range(n):
            name, off, length = struct.unpack("<16sQQ", f.read(32))
            if name.split(b"\0", 1)[0] == b"model":
                return off, length
    raise ValueError("bundle has no model section")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=("validation", "test", "all"), default="test")
    ap.add_argument("--rows-per-unit", type=int, default=8)
    ap.add_argument("--targets", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--include-observed", action="store_true",
                    help="score sparse input CpGs too (historical Zhou/Hao protocol)")
    ap.add_argument("--shared-rows", action="store_true",
                    help="reuse cell/replicate rows across units for sequential I/O")
    ap.add_argument("--unit-tsv")
    args = ap.parse_args()

    model_path, data_path = Path(args.model), Path(args.data)
    moff, mbytes = model_section(model_path)
    with model_path.open("rb") as f:
        f.seek(moff)
        mh = struct.unpack("<8s8I11Q", f.read(128))
    if mh[0] != b"UPDEC2\0\0" or mh[1] != 2 or mh[16] != mbytes:
        raise ValueError("invalid UPDEC2 header")
    _, _, flags, patterns, input_dim, n_units, n_members, target, activation, n_cpg, \
        mean_off, scale_off, unit_off, cpg_off, member_off, param_off, file_bytes, \
        index_sum, parameter_sum, reserved = mh
    mean = np.memmap(model_path, "<f4", "r", moff + mean_off, (patterns,))
    scale = np.memmap(model_path, "<f4", "r", moff + scale_off, (patterns,))
    unit_dtype = np.dtype([
        ("output", "<u8"), ("param", "<u8"), ("param_bytes", "<u8"),
        ("cpg_count", "<u4"), ("memberships", "<u4"),
        ("mode", "<u2"), ("rank", "<u2"), ("flags", "<u4")])
    units = np.memmap(model_path, unit_dtype, "r", moff + unit_off, (n_units,))
    cpg = np.memmap(model_path, "<u4", "r", moff + cpg_off, (n_cpg,))

    with data_path.open("rb") as f:
        dh = struct.unpack("<8s4IQ2I4Q", f.read(72))
    magic, version, n_cells, n_reps, side_p, side_cpg, sampled, dflags, \
        group_off, truth_off, records_off, record_bytes = dh
    if magic != b"MSURAW2\0" or version != 2 or not dflags & 1:
        raise ValueError("evaluation requires embedded-truth MSURAW2")
    if side_p < patterns or side_cpg != n_cpg:
        raise ValueError("model and sidecar dimensions differ")
    raw = np.memmap(data_path, "u1", "r")
    truth = np.memmap(data_path, "<u2", "r", truth_off, (n_cells, n_cpg))
    _, val_cells, test_cells = source_split(n_cells, args.seed)
    cells = val_cells if args.split == "validation" else \
            test_cells if args.split == "test" else list(range(n_cells))
    shared = []
    if args.shared_rows:
        srng = np.random.default_rng(args.seed + 424242)
        for _ in range(args.rows_per_unit * 32):
            shared.append((int(cells[srng.integers(len(cells))]),
                           int(srng.integers(n_reps))))

    # SSE, SAE, N, sum(y), sum(p), sum(y^2), sum(p^2), sum(y*p)
    totals = {k: [0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0]
              for k in ("all", "pure", "mixed", "PNA")}
    unit_rows = []
    for ui, u in enumerate(units):
        begin, count = int(u["output"]), int(u["cpg_count"])
        positions = np.asarray(cpg[begin:begin + count], dtype=np.uint32)
        par = np.memmap(model_path, "<f4", "r", moff + int(u["param"]),
                        (int(u["param_bytes"]) // 4,))
        rank, mode = int(u["rank"]), int(u["mode"])
        if mode == 1:
            q = rank * input_dim
            A = par[:q].reshape(rank, input_dim)
            a = par[q:q + rank]
            q += rank
            E = par[q:q + count * rank].reshape(count, rank)
            b = par[q + count * rank:q + count * rank + count]
        else:
            W = par[:count * input_dim].reshape(count, input_dim)
            b = par[count * input_dim:count * input_dim + count]
        rng = np.random.default_rng(args.seed + 1000003 * (ui + 1))
        se = ae = 0.0
        nn = 0
        sy = sp = sy2 = sp2 = syp = 0.0
        made = tries = 0
        while made < args.rows_per_unit and tries < args.rows_per_unit * 32:
            tries += 1
            if shared:
                cell, rep = shared[tries - 1]
            else:
                cell = int(cells[rng.integers(len(cells))])
                rep = int(rng.integers(n_reps))
            roff = records_off + (rep * n_cells + cell) * record_bytes
            beta = np.ndarray((patterns,), "<f4", raw, roff)
            x = np.empty(input_dim, np.float32)
            missing = ~np.isfinite(beta)
            x[:patterns] = np.where(missing, 0, (beta - mean) / scale)
            x[patterns:] = missing
            selected = np.ndarray((sampled,), "<u4", raw,
                                  roff + side_p * 8)
            target = np.asarray(truth[cell, positions])
            valid = target != 65535
            at = np.searchsorted(selected, positions)
            in_range = at < sampled
            observed = np.zeros(count, dtype=bool)
            observed[in_range] = selected[at[in_range]] == positions[in_range]
            ids = np.flatnonzero(valid if args.include_observed else valid & ~observed)
            if not len(ids):
                continue
            if len(ids) > args.targets:
                ids = rng.choice(ids, args.targets, replace=False)
            y = target[ids].astype(np.float32) / 65534.0
            if mode == 1:
                z = A @ x + a
                if activation == 1:
                    z = np.where(z >= 0, z, .01 * z)
                pred = sigmoid(E[ids] @ z + b[ids])
            else:
                pred = sigmoid(W[ids] @ x + b[ids])
            delta = pred - y
            se += float(delta @ delta)
            ae += float(np.abs(delta).sum())
            nn += len(ids)
            yd = y.astype(np.float64)
            pd = pred.astype(np.float64)
            sy += float(yd.sum())
            sp += float(pd.sum())
            sy2 += float(yd @ yd)
            sp2 += float(pd @ pd)
            syp += float(yd @ pd)
            made += 1
        cls = "PNA" if int(u["flags"]) & 2 else \
              "pure" if int(u["flags"]) & 1 else "mixed"
        unit_rows.append((ui, cls, count, nn,
                          math.sqrt(se / nn) if nn else math.nan,
                          ae / nn if nn else math.nan))
        for key in ("all", cls):
            totals[key][0] += se
            totals[key][1] += ae
            totals[key][2] += nn
            totals[key][3] += sy
            totals[key][4] += sp
            totals[key][5] += sy2
            totals[key][6] += sp2
            totals[key][7] += syp
        if (ui + 1) % 25 == 0 or ui + 1 == n_units:
            print(f"[updec2-eval] {ui + 1}/{n_units}", flush=True)

    print("stratum\tn\trmse\tmae\tpearson")
    for key in ("all", "pure", "mixed", "PNA"):
        se, ae, n, sy, sp, sy2, sp2, syp = totals[key]
        den = math.sqrt(max(0, n * sy2 - sy * sy) *
                        max(0, n * sp2 - sp * sp))
        corr = (n * syp - sy * sp) / den if den else math.nan
        print(f"{key}\t{n}\t{math.sqrt(se/n):.9g}\t{ae/n:.9g}\t{corr:.9g}")
    if args.unit_tsv:
        with open(args.unit_tsv, "w") as f:
            f.write("unit\tclass\tcpgs\tn\trmse\tmae\n")
            for row in unit_rows:
                f.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
