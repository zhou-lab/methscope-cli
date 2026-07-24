#!/usr/bin/env python3
"""Compare matched compact-factor and direct pilot validation metrics."""
import argparse
import csv


def read(path):
    with open(path) as f:
        return {int(r["unit"]): r for r in csv.DictReader(f, delimiter="\t")}


def metric(rows, key):
    n = sum(int(r["validation_n"]) for r in rows)
    return sum(int(r["validation_n"]) * float(r["validation_mae"])
               for r in rows) / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True)
    ap.add_argument("--direct", required=True)
    ap.add_argument("--gate", type=float, default=.005)
    args = ap.parse_args()
    factor, direct = read(args.factor), read(args.direct)
    if factor.keys() != direct.keys():
        raise ValueError("pilot unit sets differ")
    print("stratum\tn\tfactor_mae\tdirect_mae\tdirect_minus_factor")
    all_delta = None
    for cls in ("all", "mixed", "PNA"):
        ids = sorted(factor) if cls == "all" else \
              [u for u in sorted(factor) if factor[u]["class"] == cls]
        fr, dr = [factor[u] for u in ids], [direct[u] for u in ids]
        fm, fn = metric(fr, "validation_mae")
        dm, dn = metric(dr, "validation_mae")
        if fn != dn:
            raise ValueError("paired pilot target counts differ")
        delta = dm - fm
        if cls == "all":
            all_delta = delta
        print(f"{cls}\t{fn}\t{fm:.9g}\t{dm:.9g}\t{delta:.9g}")
    improvement = -all_delta
    print(f"direct_improvement\t{improvement:.9g}")
    print(f"full_direct_gate\t{'PASS' if improvement >= args.gate else 'FAIL'}")


if __name__ == "__main__":
    main()
