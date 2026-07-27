#!/usr/bin/env python3
"""Score a UPDEC2 model across a coverage ladder from ONE dense msur.

Inputs are drawn fresh from each cell's eligible CpGs in the embedded truth, so
any coverage up to the cell's own is reachable regardless of the msur's --sample.
We permute the eligible set once per row and take nested prefixes, so level k's
observed set is contained in level k+1's -- the same relationship real added sequencing
depth would produce, and it removes between-level sampling variance from the
comparison.

Features are recomputed from the msur's per-CpG group map and its embedded
truth, exactly as `upscale-featurize` would: beta[g] = mean over observed CpGs of
pattern g, count[g] = how many were observed, NaN where none. With --depth 1 each
observed CpG contributes a Bernoulli(beta_truth) draw instead of the bulk beta,
which is what a one-read call actually looks like; the prediction TARGETS stay
noiseless truth either way.

Targets are drawn once per row and shared across every level, so the levels form
a paired comparison on identical (cell, replicate, CpG) tuples.
"""
import argparse
import math
import struct
from pathlib import Path

import numpy as np

from compare_updec2_models import Model
from updec2_eval import source_split


def parse_levels(spec, n_cpg):
    """Accept '0.01%,1%,...' fractions or bare CpG counts; return sorted ints."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("%"):
            out.append(int(round(float(tok[:-1]) / 100.0 * n_cpg)))
        else:
            out.append(int(tok))
    if not out:
        raise ValueError("no levels given")
    return sorted(set(out))


def features(observed, groups, truth_row, patterns, depth, rng):
    """Rebuild (beta, count) over P patterns from an observed CpG set."""
    y = truth_row[observed].astype(np.float32) / 65534.0
    if depth:
        ## d reads at each observed CpG: M ~ Binomial(d, beta), feature = M/d.
        y = rng.binomial(depth, np.clip(y, 0.0, 1.0)).astype(np.float32) / depth
    g = groups[observed].astype(np.int64)
    keep = (g > 0) & (g <= patterns)
    g, y = g[keep], y[keep]
    cnt = np.bincount(g, minlength=patterns + 1)[1:]
    tot = np.bincount(g, weights=y.astype(np.float64),
                      minlength=patterns + 1)[1:]
    beta = np.full(patterns, np.nan, np.float32)
    hit = cnt != 0
    beta[hit] = (tot[hit] / cnt[hit]).astype(np.float32)
    return beta, cnt.astype(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True,
                    help="embedded-truth MSURAW2/3 msur (supplies truth + group map)")
    ap.add_argument("--levels",
                    default="0.01%,0.05%,0.1%,0.5%,1%,5%,10%")
    ap.add_argument("--depth", type=int, default=0,
                    help="reads per observed CpG (0 = noise-free bulk beta)")
    ap.add_argument("--split", choices=("validation", "test", "all"),
                    default="all")
    ap.add_argument("--split-file",
                    help="upscale-train --split file the model was trained\n"
                         "with; omit only for a seeded 70/15/15 model")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--targets-per-row", type=int, default=131072)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--target-cpgs",
                    help="file of CpG indices (one per line) to score on, e.g.\n"
                         "SignatureAtlas wg.bed mapped to the CpG universe.\n"
                         "Inputs stay genome-wide; only the TARGETS narrow, so\n"
                         "this isolates cell-type-specific recovery from the\n"
                         "global-methylation-level signal that dominates a\n"
                         "genome-uniform target set.")
    ap.add_argument("--row-tsv")
    args = ap.parse_args()

    model = Model(args.model)
    dp = Path(args.data)
    with dp.open("rb") as f:
        dh = struct.unpack("<8s4IQ2I4Q", f.read(72))
    magic, _, n_cells, n_reps, side_p, n_cpg, sampled, flags, group_off, \
        truth_off, records_off, record_bytes = dh
    if magic not in (b"MSURAW2\0", b"MSURAW3\0") or not flags & 1:
        raise ValueError("need an embedded-truth MSURAW2/3 msur")
    if n_cpg != model.n_cpg or side_p < model.patterns:
        raise ValueError("msur dimensions disagree with the model")

    levels = parse_levels(args.levels, n_cpg)

    raw = np.memmap(dp, "u1", "r")
    truth = np.memmap(dp, "<u2", "r", truth_off, (n_cells, n_cpg))
    groups = np.memmap(dp, "<u2", "r", group_off, (n_cpg,))
    _, val, test = source_split(n_cells, args.seed, args.split_file)
    cells = val if args.split == "validation" else \
        test if args.split == "test" else list(range(n_cells))

    panel = None
    if args.target_cpgs:
        panel = np.loadtxt(args.target_cpgs, dtype=np.uint32, ndmin=1)
        panel = np.unique(panel)
        if panel.max() >= n_cpg:
            raise ValueError("target CpG index out of range")
        print(f"[ladder] scoring on {len(panel):,} supplied target CpGs")

    rng = np.random.default_rng(args.seed + 987654321)
    rows = []
    pooled = {n: ([], []) for n in levels}
    for ri in range(args.rows):
        cell = int(cells[rng.integers(len(cells))])
        rep = int(rng.integers(n_reps))
        ## Inputs are drawn fresh from the cell's eligible CpGs rather than
        ## subsetting the msur's stored `selected`, which would cap the ladder at
        ## whatever --sample the msur was built with.  The embedded truth carries
        ## every cell x every CpG, so any coverage up to the cell's own is
        ## reachable; all that evaluation needs is a uniform draw, not the YAME
        ## dsample protocol (which matters when BUILDING an msur, not scoring one).
        truth_row = np.asarray(truth[cell])
        eligible = np.flatnonzero(truth_row != 65535).astype(np.uint32)
        if levels[-1] > len(eligible):
            raise ValueError(
                f"cell {cell} has {len(eligible)} eligible CpGs, "
                f"below the densest level {levels[-1]}")

        ## Targets: genome-uniform, valid truth, shared across all levels so the
        ## ladder is a paired comparison.  Observed CpGs are NOT excluded -- the
        ## observed set differs per level, and excluding it would change the
        ## target set with it (this matches the --include-observed protocol the
        ## 20260724 external comparison used).
        if panel is not None:
            ## Fixed panel: score every supplied CpG this cell actually covers.
            genomic = panel[np.asarray(truth[cell, panel]) != 65535]
        else:
            need = args.targets_per_row
            chosen = np.empty(0, np.uint32)
            while len(chosen) < need:
                cand = rng.choice(n_cpg, min(n_cpg, 2 * need),
                                  replace=False).astype(np.uint32)
                chosen = np.unique(np.concatenate(
                    (chosen, cand[np.asarray(truth[cell, cand]) != 65535])))
            genomic = rng.choice(chosen, need, replace=False).astype(np.uint32)
        y = np.asarray(truth[cell, genomic], np.float32) / 65534.0

        order = rng.permutation(len(eligible))    # nested prefixes below
        draw = np.random.default_rng(args.seed + 31 * ri)
        row = {"row": ri, "cell": cell, "replicate": rep, "n": len(y)}
        for n in levels:
            obs = np.sort(eligible[order[:n]])
            beta, cnt = features(obs, groups, truth_row, model.patterns,
                                 args.depth, draw)
            pred = model.predict(beta[:model.patterns], cnt[:model.patterns],
                                 genomic)
            row[n] = float(np.abs(pred - y).mean())
            pooled[n][0].append(y)
            pooled[n][1].append(pred)
        rows.append(row)
        print(f"[ladder] rows={ri + 1}/{args.rows}", flush=True)

    print(f"depth\t{args.depth if args.depth else 'noise-free'}")
    print("level_cpgs\tpct_of_genome\tn\trmse\tmae\tpearson")
    for n in levels:
        ys, ps = map(np.concatenate, pooled[n])
        d = ps - ys
        print(f"{n}\t{100.0 * n / n_cpg:.4g}\t{len(ys)}\t"
              f"{math.sqrt(float(d @ d) / len(ys)):.9g}\t"
              f"{float(np.abs(d).mean()):.9g}\t"
              f"{float(np.corrcoef(ys, ps)[0, 1]):.9g}")

    ## Paired row bootstrap of each level against the densest one, so the
    ## degradation curve carries an interval rather than a point.
    ref = levels[-1]
    brng = np.random.default_rng(args.seed + 5)
    idx = brng.integers(len(rows), size=(10000, len(rows)))
    print("level_cpgs\tmae_minus_densest\tboot95_low\tboot95_high")
    for n in levels:
        delta = np.asarray([r[n] - r[ref] for r in rows])
        boot = delta[idx].mean(1)
        lo, hi = np.quantile(boot, [.025, .975])
        print(f"{n}\t{delta.mean():.9g}\t{lo:.9g}\t{hi:.9g}")

    if args.row_tsv:
        with open(args.row_tsv, "w") as f:
            f.write("row\tcell\treplicate\tn\t" +
                    "\t".join(f"mae_{n}" for n in levels) + "\n")
            for r in rows:
                f.write("\t".join(str(r[k]) for k in
                                  ("row", "cell", "replicate", "n")) + "\t" +
                        "\t".join(f"{r[n]:.9g}" for n in levels) + "\n")


if __name__ == "__main__":
    main()
