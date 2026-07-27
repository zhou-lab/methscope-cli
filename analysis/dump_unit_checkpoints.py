#!/usr/bin/env python3
"""Dump per-unit best_step / best_mae from an upscale-train --work-dir.

Each unit_%06u.upuck checkpoint (UPUCK1, see src/upunit_cuda.cu CheckHeader)
records the step at which the unit's validation MAE was best and that MAE, so
two runs that share a units index, split and seed can be compared unit by unit:
which units used their step budget, and where the accuracy actually differs.
"""
import argparse
import struct
from pathlib import Path

## UPUCK1 header: 8s magic, 8x uint32, 3x uint64, float best_mae, uint32 pad.
HEADER = struct.Struct("<8s8I3QfI")


def read_checkpoint(path):
    with path.open("rb") as f:
        h = HEADER.unpack(f.read(HEADER.size))
    if h[0][:6] != b"UPUCK1":
        raise ValueError(f"{path}: not a UPUCK1 checkpoint")
    return {
        "unit": h[2], "mode": h[3], "rank": h[4], "activation": h[5],
        "best_step": h[6], "cpg_count": h[7], "input_dim": h[8],
        "param_floats": h[11], "best_mae": h[12],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir", nargs="+",
                    help="one or more upscale-train --work-dir directories")
    ap.add_argument("-o", "--out", help="TSV output (default stdout)")
    args = ap.parse_args()

    rows = []
    for wd in args.work_dir:
        d = Path(wd)
        for ck in sorted(d.glob("unit_*.upuck")):
            r = read_checkpoint(ck)
            r["work_dir"] = d.name
            rows.append(r)

    cols = ["work_dir", "unit", "mode", "rank", "input_dim", "cpg_count",
            "param_floats", "best_step", "best_mae"]
    out = open(args.out, "w") if args.out else None
    try:
        w = (out or __import__("sys").stdout)
        w.write("\t".join(cols) + "\n")
        for r in rows:
            w.write("\t".join(
                f"{r[c]:.9g}" if c == "best_mae" else str(r[c])
                for c in cols) + "\n")
    finally:
        if out:
            out.close()


if __name__ == "__main__":
    main()
