#!/usr/bin/env python3
"""Paired global-CpG comparison of two UPDEC2 models on identical targets."""
import argparse
import math
import struct
from pathlib import Path

import numpy as np

from updec2_eval import model_section, sigmoid, source_split


class Model:
    def __init__(self, path):
        self.path = Path(path)
        self.offset, _ = model_section(self.path)
        with self.path.open("rb") as f:
            f.seek(self.offset)
            h = struct.unpack("<8s8I11Q", f.read(128))
        self.version, self.flags = h[1], h[2]
        self.patterns, self.input_dim = h[3], h[4]
        self.n_units, self.activation, self.n_cpg = h[5], h[8], h[9]
        prep_n = self.patterns if self.version == 2 else self.input_dim
        self.mean = np.memmap(self.path, "<f4", "r", self.offset + h[10],
                              (prep_n,))
        self.scale = np.memmap(self.path, "<f4", "r", self.offset + h[11],
                               (prep_n,))
        udt = np.dtype([
            ("output", "<u8"), ("param", "<u8"), ("param_bytes", "<u8"),
            ("cpg_count", "<u4"), ("memberships", "<u4"),
            ("mode", "<u2"), ("rank", "<u2"), ("flags", "<u4")])
        self.units = np.memmap(self.path, udt, "r", self.offset + h[12],
                               (self.n_units,))
        self.cpg = np.memmap(self.path, "<u4", "r", self.offset + h[13],
                             (self.n_cpg,))
        self.inverse = np.empty(self.n_cpg, np.uint32)
        self.inverse[self.cpg] = np.arange(self.n_cpg, dtype=np.uint32)
        self.ends = np.asarray(self.units["output"] + self.units["cpg_count"])
        self.hidden = int(h[19]) if self.version >= 3 and self.flags & 4 else 0
        self.unit_input = self.hidden or self.input_dim
        if self.hidden:
            q = self.hidden * self.input_dim
            par = np.memmap(self.path, "<f4", "r", self.offset + h[15],
                            (q + self.hidden + self.hidden ** 2 + self.hidden,))
            self.w1 = par[:q].reshape(self.hidden, self.input_dim)
            self.b1 = par[q:q + self.hidden]
            q += self.hidden
            self.w2 = par[q:q + self.hidden ** 2].reshape(
                self.hidden, self.hidden)
            self.b2 = par[q + self.hidden ** 2:]

    def input(self, beta, count):
        x = np.empty(self.input_dim, np.float32)
        missing = ~np.isfinite(beta)
        if self.version == 2:
            x[:self.patterns] = np.where(
                missing, 0, (beta - self.mean) / self.scale)
            x[self.patterns:] = missing
        elif self.flags & 8:
            x[:] = np.where(missing | (count == 0), 0,
                            (beta - self.mean) / self.scale)
        else:
            x[0::2] = np.where(missing | (count == 0), 0,
                               (beta - self.mean[0::2]) /
                               self.scale[0::2])
            aux = np.log1p(count) if self.flags & 2 else \
                  (missing | (count == 0))
            x[1::2] = (aux - self.mean[1::2]) / self.scale[1::2]
        if self.hidden:
            h1 = self.w1 @ x + self.b1
            h1 = np.where(h1 >= 0, h1, .01 * h1)
            h2 = self.w2 @ h1 + self.b2
            x = h1 + np.where(h2 >= 0, h2, .01 * h2)
        return x

    def predict(self, beta, count, genomic):
        x = self.input(beta, count)
        ordered = self.inverse[genomic]
        unit_id = np.searchsorted(self.ends, ordered, side="right")
        pred = np.empty(len(genomic), np.float32)
        for ui in np.unique(unit_id):
            take = np.flatnonzero(unit_id == ui)
            u = self.units[ui]
            local = ordered[take] - int(u["output"])
            count, rank = int(u["cpg_count"]), int(u["rank"])
            par = np.memmap(self.path, "<f4", "r",
                            self.offset + int(u["param"]),
                            (int(u["param_bytes"]) // 4,))
            if int(u["mode"]) == 1:
                q = rank * self.unit_input
                A = par[:q].reshape(rank, self.unit_input)
                a = par[q:q + rank]
                q += rank
                E = par[q:q + count * rank].reshape(count, rank)
                b = par[q + count * rank:q + count * rank + count]
                z = A @ x + a
                if self.activation == 1:
                    z = np.where(z >= 0, z, .01 * z)
                pred[take] = sigmoid(E[local] @ z + b[local])
            else:
                W = par[:count * self.unit_input].reshape(
                    count, self.unit_input)
                b = par[count * self.unit_input:
                        count * self.unit_input + count]
                pred[take] = sigmoid(W[local] @ x + b[local])
        return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", choices=("validation", "test", "all"),
                    default="validation")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--targets-per-row", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--include-observed", action="store_true")
    ap.add_argument("--row-tsv")
    args = ap.parse_args()
    a, b = Model(args.model_a), Model(args.model_b)
    if a.n_cpg != b.n_cpg or a.patterns != b.patterns:
        raise ValueError("model dimensions differ")

    dp = Path(args.data)
    with dp.open("rb") as f:
        dh = struct.unpack("<8s4IQ2I4Q", f.read(72))
    _, _, n_cells, n_reps, side_p, n_cpg, sampled, flags, _, truth_off, \
        records_off, record_bytes = dh
    if n_cpg != a.n_cpg or side_p < a.patterns or not flags & 1:
        raise ValueError("sidecar dimensions differ")
    raw = np.memmap(dp, "u1", "r")
    truth = np.memmap(dp, "<u2", "r", truth_off, (n_cells, n_cpg))
    _, val, test = source_split(n_cells, args.seed)
    cells = val if args.split == "validation" else \
            test if args.split == "test" else list(range(n_cells))
    rng = np.random.default_rng(args.seed + 987654321)
    rows = []
    all_y, all_a, all_b = [], [], []
    for ri in range(args.rows):
        cell = int(cells[rng.integers(len(cells))])
        rep = int(rng.integers(n_reps))
        roff = records_off + (rep * n_cells + cell) * record_bytes
        beta = np.ndarray((a.patterns,), "<f4", raw, roff)
        count = np.ndarray((a.patterns,), "<u4", raw, roff + side_p * 4)
        selected = np.ndarray((sampled,), "<u4", raw, roff + side_p * 8)
        need = args.targets_per_row
        chosen = np.empty(0, np.uint32)
        while len(chosen) < need:
            candidate = rng.choice(n_cpg, min(n_cpg, 2 * need),
                                   replace=False).astype(np.uint32)
            target = np.asarray(truth[cell, candidate])
            valid = target != 65535
            if not args.include_observed:
                at = np.searchsorted(selected, candidate)
                in_range = at < sampled
                observed = np.zeros(len(candidate), bool)
                observed[in_range] = \
                    selected[at[in_range]] == candidate[in_range]
                valid &= ~observed
            chosen = np.unique(np.concatenate((chosen, candidate[valid])))
        genomic = rng.choice(chosen, need, replace=False).astype(np.uint32)
        y = np.asarray(truth[cell, genomic], np.float32) / 65534.0
        pa, pb = a.predict(beta, count, genomic), \
                 b.predict(beta, count, genomic)
        ma = float(np.abs(pa - y).mean())
        mb = float(np.abs(pb - y).mean())
        rows.append((ri, cell, rep, len(y), ma, mb, mb - ma))
        all_y.append(y); all_a.append(pa); all_b.append(pb)
        print(f"[paired-updec2] {ri + 1}/{args.rows}", flush=True)
    y, pa, pb = map(np.concatenate, (all_y, all_a, all_b))
    print("model\tn\trmse\tmae\tpearson")
    for name, p in (("A", pa), ("B", pb)):
        d = p - y
        print(f"{name}\t{len(y)}\t{math.sqrt(float(d@d)/len(y)):.9g}\t"
              f"{float(np.abs(d).mean()):.9g}\t"
              f"{float(np.corrcoef(y, p)[0,1]):.9g}")
    delta = np.asarray([r[-1] for r in rows])
    brng = np.random.default_rng(args.seed + 5)
    boot = delta[brng.integers(len(delta), size=(10000, len(delta)))].mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    print(f"B_minus_A_mae\t{float(np.abs(pb-y).mean()-np.abs(pa-y).mean()):.9g}")
    print(f"paired_row_bootstrap_95pct\t{lo:.9g}\t{hi:.9g}")
    if args.row_tsv:
        with open(args.row_tsv, "w") as f:
            f.write("row\tcell\treplicate\tn\tmae_A\tmae_B\tB_minus_A\n")
            for r in rows:
                f.write("\t".join(map(str, r)) + "\n")


if __name__ == "__main__":
    main()
