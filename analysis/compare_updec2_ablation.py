#!/usr/bin/env python3
"""Matched four-model UPDEC2 pilot comparison with unit bootstrap intervals."""
import argparse
import csv
import random


def read(path):
    with open(path, newline="") as f:
        return {int(r["unit"]): r for r in csv.DictReader(f, delimiter="\t")}


def weighted(rows, ids):
    n = sum(int(rows[u]["validation_n"]) for u in ids)
    return sum(float(rows[u]["validation_mae"]) *
               int(rows[u]["validation_n"]) for u in ids) / n


def percentile(x, p):
    x = sorted(x)
    return x[round((len(x) - 1) * p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta-no-trunk", required=True)
    ap.add_argument("--beta-trunk", required=True)
    for i, name in enumerate(("missing_notrunk", "count_notrunk",
                              "missing_trunk", "count_trunk"), 1):
        ap.add_argument(f"--model{i}", required=True, metavar=name.upper())
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    # Order makes the 3x2 design explicit: beta/missing/count without a trunk,
    # followed by beta/missing/count with the shared trunk.
    models = [read(a.beta_no_trunk), read(a.model1), read(a.model2),
              read(a.beta_trunk), read(a.model3), read(a.model4)]
    ids = sorted(models[0])
    if any(sorted(m) != ids for m in models[1:]):
        raise ValueError("pilot unit IDs differ")
    for u in ids:
        ns = {int(m[u]["validation_n"]) for m in models}
        if len(ns) != 1:
            raise ValueError(f"validation target counts differ for unit {u}")

    names = ("beta/no-trunk", "missing/no-trunk", "count/no-trunk",
             "beta/trunk", "missing/trunk", "count/trunk")
    score = [weighted(m, ids) for m in models]
    print("model\tunits\tvalidation_n\tweighted_mae")
    total_n = sum(int(models[0][u]["validation_n"]) for u in ids)
    for name, value in zip(names, score):
        print(f"{name}\t{len(ids)}\t{total_n}\t{value:.9g}")

    contrasts = (
        ("missing effect, no trunk", 1, 0),
        ("count effect, no trunk", 2, 0),
        ("trunk effect, beta", 3, 0),
        ("trunk effect, missing", 4, 1),
        ("trunk effect, count", 5, 2),
        ("missing effect, trunk", 4, 3),
        ("count effect, trunk", 5, 3),
    )
    rng = random.Random(a.seed)
    boot = [[] for _ in contrasts]
    for _ in range(a.bootstrap):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        values = [weighted(m, sample) for m in models]
        for out, (_, x, y) in zip(boot, contrasts):
            out.append(values[x] - values[y])
    print("\ncontrast\tdelta_mae\tbootstrap_95_low\tbootstrap_95_high")
    for (label, x, y), values in zip(contrasts, boot):
        print(f"{label}\t{score[x]-score[y]:.9g}\t"
              f"{percentile(values, .025):.9g}\t{percentile(values, .975):.9g}")


if __name__ == "__main__":
    main()
