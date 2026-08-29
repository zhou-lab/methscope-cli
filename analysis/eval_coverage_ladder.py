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

Besides MAE/RMSE/Pearson, each level reports the CALL metrics a binarized
reconstruction is actually read with: accuracy of the 0.5-thresholded call
against the 0.5-thresholded truth, plus AUROC and average precision (AUPRC)
using the predicted beta as the score.  --floor adds the no-information
baseline for every metric -- the per-CpG median (MAE) / majority call
(accuracy) taken across the cohort -- which is the only honest reference line
for a panel that runs down to a handful of observed CpGs.
"""
import argparse
import math
import struct
from pathlib import Path

import numpy as np

from compare_updec2_models import Model
from updec2_eval import source_split


def parse_levels(spec, n_cpg):
    """Accept '0.01%,1%,...' fractions, bare CpG counts, or 'MIN,MAX,N@SKEW'.

    The third form reproduces `upscale-featurize --sample-logrange MIN,MAX
    --sample-skew SKEW --reps N` exactly (upscale_prepare.c:425), so a ladder
    can be scored on the SAME coverage levels the model was trained over
    rather than on a hand-picked subset of them.
    """
    if "@" in spec:
        head, skew = spec.split("@")
        lo_s, hi_s, n_s = head.split(",")
        lo, hi, reps, skew = (float(lo_s), float(hi_s), int(n_s), float(skew))
        base, span = math.log(lo), math.log(hi) - math.log(lo)
        out = []
        for r in range(reps):
            u = r / (reps - 1) if reps > 1 else 1.0
            out.append(max(1, int(math.exp(base + span * u ** skew) + 0.5)))
        return sorted(set(out))
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


def call_metrics(y, p):
    """Accuracy, AUROC and average precision of a 0.5-thresholded call.

    Truth and prediction are both continuous betas; a reconstruction is read
    as a methylated/unmethylated CALL, so binarize both at 0.5.  AP is the
    step-wise sum used by sklearn's average_precision_score, computed here to
    keep this script dependency-free.
    """
    pos = y >= 0.5
    acc = float(((p >= 0.5) == pos).mean())
    n_pos = int(pos.sum())
    if n_pos == 0 or n_pos == len(pos):
        return acc, float("nan"), float("nan")
    o = np.argsort(-p, kind="stable")
    hit = pos[o]
    tp = np.cumsum(hit)
    fp = np.cumsum(~hit)
    ## Collapse runs of equal scores to their last index, so tied predictions
    ## share one threshold -- without this a tie is resolved by sort order and
    ## the curve reads better than it is (matters here: a sparse-input
    ## reconstruction emits the same prior at many CpGs at once).
    keep = np.r_[np.diff(p[o]) != 0, True]
    tp, fp = tp[keep], fp[keep]
    tpr = np.r_[0.0, tp / n_pos]
    fpr = np.r_[0.0, fp / (len(pos) - n_pos)]
    auroc = float(np.trapezoid(tpr, fpr))
    recall, precision = tp / n_pos, tp / (tp + fp)
    ap = float((np.diff(np.r_[0.0, recall]) * precision).sum())
    return acc, auroc, ap


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
                    default="0.01%,0.05%,0.1%,0.5%,1%,5%,10%",
                    help="comma list of CpG counts or percentages, or\n"
                         "MIN,MAX,N@SKEW for the trainer's logrange ladder\n"
                         "(e.g. 71,2940180,100@0.5 = the 100 levels the\n"
                         "shipped human model was trained over)")
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
    ap.add_argument("--metrics-tsv",
                    help="write the per-level metric table here as well")
    ap.add_argument("--floor", action="store_true",
                    help="also score the no-information baseline: per-CpG\n"
                         "median beta (and majority call) across the cohort.\n"
                         "Costs one sequential pass over the embedded truth.")
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

    ## Per-CpG cohort median/majority over the target panel -- the only
    ## defensible reference line once the input drops to a handful of CpGs,
    ## because MAE is minimized by the MEDIAN (not the mean) and a call is
    ## scored against the MAJORITY.  One sequential pass over the truth.
    floor_beta = None
    if args.floor:
        if panel is None:
            raise ValueError("--floor needs a fixed --target-cpgs panel")
        cohort = np.empty((n_cells, len(panel)), np.float32)
        for c in range(n_cells):
            v = np.asarray(truth[c])[panel].astype(np.float32)
            cohort[c] = np.where(v == 65535, np.nan, v / 65534.0)
            if (c + 1) % 100 == 0:
                print(f"[floor] cells={c + 1}/{n_cells}", flush=True)
        ## The per-CpG MEDIAN over the cells that actually cover it, and
        ## nothing else: MAE is minimized by the median (quoting the MEAN is
        ## the error the 20260727 entry caught), and thresholding that median
        ## at 0.5 IS the majority call, so one vector floors both metrics.
        ## Columns no cell covers get 0.5; they are never scored, since each
        ## row keeps only the panel CpGs its own cell covers.
        with np.errstate(invalid="ignore"):
            floor_beta = np.nanmedian(cohort, axis=0).astype(np.float32)
        floor_beta = np.where(np.isnan(floor_beta), 0.5,
                              floor_beta).astype(np.float32)
        del cohort
        idx_of = np.full(n_cpg, -1, np.int64)
        idx_of[panel] = np.arange(len(panel))

    rng = np.random.default_rng(args.seed + 987654321)
    rows = []
    pooled = {n: ([], []) for n in levels}
    floor_pool = ([], [])
    for ri in range(args.rows):
        ## Inputs are drawn fresh from the cell's eligible CpGs rather than
        ## subsetting the msur's stored `selected`, which would cap the ladder at
        ## whatever --sample the msur was built with.  The embedded truth carries
        ## every cell x every CpG, so any coverage up to the cell's own is
        ## reachable; all that evaluation needs is a uniform draw, not the YAME
        ## dsample protocol (which matters when BUILDING an msur, not scoring one).
        ##
        ## A cohort mixes deep and shallow cells, so redraw rather than abort
        ## when the drawn cell cannot reach the densest rung -- at a 10%-of-
        ## genome top rung a handful of shallow cells would otherwise kill the
        ## whole run.  With a top rung every cell reaches, this never fires and
        ## the draw is identical to the historical one.
        for attempt in range(64):
            cell = int(cells[rng.integers(len(cells))])
            truth_row = np.asarray(truth[cell])
            eligible = np.flatnonzero(truth_row != 65535).astype(np.uint32)
            if len(eligible) >= levels[-1]:
                break
            print(f"[ladder] skip cell {cell}: {len(eligible):,} eligible "
                  f"< {levels[-1]:,}", flush=True)
        else:
            raise ValueError(
                f"no cell reaches the densest level {levels[-1]:,}")
        rep = int(rng.integers(n_reps))

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
        if floor_beta is not None:
            floor_pool[0].append(y)
            floor_pool[1].append(floor_beta[idx_of[genomic]])
        rows.append(row)
        print(f"[ladder] rows={ri + 1}/{args.rows}", flush=True)

    def score(ys, ps):
        d = ps - ys
        acc, auroc, ap = call_metrics(ys, ps)
        return (math.sqrt(float(d @ d) / len(ys)), float(np.abs(d).mean()),
                float(np.corrcoef(ys, ps)[0, 1]), acc, auroc, ap)

    head = ("level_cpgs\tpct_of_genome\tn\trmse\tmae\tpearson\t"
            "accuracy\tauroc\tauprc")
    table = []
    for n in levels:
        ys, ps = map(np.concatenate, pooled[n])
        table.append((str(n), f"{100.0 * n / n_cpg:.6g}", str(len(ys)))
                     + tuple(f"{v:.9g}" for v in score(ys, ps)))
    if floor_beta is not None:
        ys, ps = map(np.concatenate, floor_pool)
        table.append(("floor", "NA", str(len(ys)))
                     + tuple(f"{v:.9g}" for v in score(ys, ps)))

    print(f"depth\t{args.depth if args.depth else 'noise-free'}")
    print(head)
    for r in table:
        print("\t".join(r))
    if args.metrics_tsv:
        with open(args.metrics_tsv, "w") as f:
            f.write(head + "\n")
            for r in table:
                f.write("\t".join(r) + "\n")

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
