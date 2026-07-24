#!/usr/bin/env python3
"""Select 24 variance-stratified mixed units plus 8 genome-spaced PNA units."""
import argparse
import struct
from pathlib import Path

import numpy as np

from updec2_eval import model_section, source_split


def evenly(items, n):
    if len(items) < n:
        raise ValueError("stratum has too few units")
    q = np.linspace(0, len(items) - 1, n)
    return [items[int(round(x))] for x in q]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="compact baseline; unit layout supplies CpG positions")
    ap.add_argument("--data", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpgs-per-unit", type=int, default=1024)
    args = ap.parse_args()

    mp, dp = Path(args.model), Path(args.data)
    moff, _ = model_section(mp)
    with mp.open("rb") as f:
        f.seek(moff)
        h = struct.unpack("<8s8I11Q", f.read(128))
    n_units, n_cpg, unit_off, cpg_off = h[5], h[9], h[12], h[13]
    unit_dtype = np.dtype([
        ("output", "<u8"), ("param", "<u8"), ("param_bytes", "<u8"),
        ("cpg_count", "<u4"), ("memberships", "<u4"),
        ("mode", "<u2"), ("rank", "<u2"), ("flags", "<u4")])
    units = np.memmap(mp, unit_dtype, "r", moff + unit_off, (n_units,))
    cpg = np.memmap(mp, "<u4", "r", moff + cpg_off, (n_cpg,))

    with dp.open("rb") as f:
        dh = struct.unpack("<8s4IQ2I4Q", f.read(72))
    n_cells, side_cpg, truth_off = dh[2], dh[5], dh[9]
    train, _, _ = source_split(n_cells, args.seed)
    truth = np.memmap(dp, "<u2", "r", truth_off, (n_cells, side_cpg))

    mixed = []
    pna = []
    for ui, u in enumerate(units):
        flags = int(u["flags"])
        if flags & 2:
            pna.append(ui)
            continue
        if flags & 1:
            continue
        begin, count = int(u["output"]), int(u["cpg_count"])
        take = min(count, args.cpgs_per_unit)
        local = np.linspace(0, count - 1, take).astype(np.int64)
        pos = np.asarray(cpg[begin + local])
        v = np.asarray(truth[np.ix_(train, pos)])
        finite = v != 65535
        x = v[finite].astype(np.float64) / 65534.0
        mixed.append((ui, float(x.var()), count))

    mixed.sort(key=lambda x: (x[1], x[0]))
    chosen = []
    for quartile in np.array_split(np.asarray(mixed, dtype=object), 4):
        chosen.extend(evenly(list(quartile), 6))
    pna_chosen = evenly(pna, 8)
    ids = sorted([int(x[0]) for x in chosen] + pna_chosen)
    Path(args.output).write_text("".join(f"{x}\n" for x in ids))
    with open(args.output + ".tsv", "w") as f:
        f.write("unit\tclass\ttraining_target_variance\tcpgs\n")
        lookup = {int(x[0]): (float(x[1]), int(x[2])) for x in mixed}
        for ui in ids:
            if ui in lookup:
                var, count = lookup[ui]
                f.write(f"{ui}\tmixed\t{var:.9g}\t{count}\n")
            else:
                f.write(f"{ui}\tPNA\tNA\t{int(units[ui]['cpg_count'])}\n")
    print(f"selected {len(ids)} units: 24 mixed + 8 PNA")


if __name__ == "__main__":
    main()
