#!/usr/bin/env python3
"""Per-unit MAE decomposition between two UPDEC2 models on identical targets.

Both models share the same processing-unit structure (same index), so each
sampled target maps to the same unit id in both. Reports the units that
contribute most to the overall MAE difference (B - A), with each model's
per-unit rank, so a rank change can be attributed to specific units.
"""
import argparse
import numpy as np

from compare_updec2_models import Model
from compare_upscale_models import sample_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="baseline (e.g. uniform r16)")
    ap.add_argument("--model-b", required=True, help="variant (e.g. adaptive)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--targets-per-row", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    A, B = Model(args.model_a), Model(args.model_b)
    rows = sample_rows(args.data, A, "all", args.rows, args.targets_per_row,
                       args.seed, include_observed=True)
    n = A.n_units
    sumA, sumB, cnt = (np.zeros(n), np.zeros(n), np.zeros(n, np.int64))
    for row in rows:
        g, c = row["beta"][:A.patterns], row["count"][:A.patterns]
        genomic, y = row["genomic"], row["truth"]
        pa = A.predict(g, c, genomic)
        pb = B.predict(g, c, genomic)
        uid = np.searchsorted(A.ends, A.inverse[genomic], side="right")
        np.add.at(sumA, uid, np.abs(pa - y))
        np.add.at(sumB, uid, np.abs(pb - y))
        np.add.at(cnt, uid, 1)

    m = cnt > 0
    u = np.where(m)[0]
    maeA, maeB = sumA[m] / cnt[m], sumB[m] / cnt[m]
    d = maeB - maeA
    contrib = d * cnt[m] / cnt.sum()            # share of the overall MAE delta
    rA = np.asarray(A.units["rank"])[m]
    rB = np.asarray(B.units["rank"])[m]
    cpg = np.asarray(A.units["cpg_count"])[m]
    flg = np.asarray(A.units["flags"])[m]
    cls = np.where(flg & 2, "PNA", np.where(flg & 1, "pure", "mixed"))

    print(f"overall MAE delta (B-A) = {d @ cnt[m] / cnt.sum():+.6f}  "
          f"(sampled {cnt.sum():,} targets over {m.sum()} units)")
    order = np.argsort(-contrib)                 # biggest positive contribution = worst
    def show(idx, title):
        print(f"\n{title}")
        print(f"{'unit':>5} {'class':5} {'cpgs':>10} {'rA':>3} {'rB':>3} "
              f"{'n':>7} {'maeA':>8} {'maeB':>8} {'dMAE':>9} {'contrib':>9}")
        for i in idx:
            print(f"{u[i]:>5} {cls[i]:5} {cpg[i]:>10,} {rA[i]:>3} {rB[i]:>3} "
                  f"{cnt[m][i]:>7} {maeA[i]:>8.5f} {maeB[i]:>8.5f} "
                  f"{d[i]:>+9.5f} {contrib[i]:>+9.6f}")
    show(order[:12], "=== worst 12 units under B (adaptive) ===")
    show(order[-12:][::-1], "=== best 12 units under B (adaptive) ===")
    # roll up by whether the unit's rank changed
    demoted = rB < rA; promoted = rB > rA; same = rB == rA
    for name, msk in (("rank-demoted (e.g. homog->8)", demoted),
                      ("rank-promoted (e.g. ->32)", promoted),
                      ("rank-unchanged", same)):
        if msk.any():
            print(f"{name:32} units={msk.sum():4d} "
                  f"cpg-share={cnt[m][msk].sum()/cnt.sum():.3f} "
                  f"net-contrib={ (d[msk]*cnt[m][msk]).sum()/cnt.sum():+.6f}")


if __name__ == "__main__":
    main()
