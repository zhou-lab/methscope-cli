# Handover: reorganising the MRMP build into four subcommands

State as of commit `8766753`. Two of four pieces are done and tested; two are
not started. The design is settled and recorded — this note is about what is
left, what is measured, and the two traps that cost real time.

Full design, help text for all four subcommands, and the measurements behind
every parameter live in
`labjournal/zhouw3/2025/20251216_methscope.org`, under
`** TODO 20260808 encapsulate MRMP build into four subcommands`.

## Target shape

```
mrmp-build           store.cg               -> global.mrmp
mrmp-build-thin      store.cg  global.mrmp  -> thin.mrmp       ┐ independent,
mrmp-build-neighbor  store.cg  global.mrmp  -> neighbor.mrmp   ┘ run in parallel
mrmp-pool --pooled-top 1000  global thin neighbor -> final.mrmp
```

A set is a set. Nothing downstream distinguishes a global from a satellite, and
inputs may come from any generator and from **different stores**, provided they
share a row space. Curated blocks are therefore just `yame subset` + `mrmp-build`
+ `mrmp-pool`, needing no new code at all.

## Done

**`mrmp-build`** — `--qfilter LO,HI`, `--max-frac-na F`, `--min-cg-depth N`,
applied inline as each CpG is resolved. A failing CpG never enters the pattern
hash, so counts and ranks are post-filter by construction. No recount, and one
fewer read of the reference than a separate pass.

**`mrmp-pool`** — pooled cut by CpG count across sets, plus an `n_cpg` equality
check. Verified: 4 sets, `--pooled-top 1000` gave global=990, two thin sets=2
each, one biology set=6, summing to exactly 1000.

**`mrmp_select.c` / `.h`** — the per-binstring union rule
`q_filter_strict UNION (q_filter AND top-N by delta_mean)` plus the relative
depth floor. Compiles, **no caller yet**. It is for `mrmp-build-thin`.

## Not started

### `mrmp-build-thin` — the largest piece

Needs three things that do not exist in C:

1. **Thin-class list** = store labels − `global.mrmp` labels. The artifact
   already carries its sample names, so this is cheap.
2. **Per-class mean reference depth**, to decide which classes are thin
   (`--max-thinclass-depth`, default 20).
3. **Projection distance** — `mean |v − w|` over reference pattern-average
   vectors — to pick the `--n-partner` nearest. This is the awkward one: it
   needs the reference featurized against the global mask, which
   `classify-featurize` already does. Either call it internally or require the
   `.msfm` as an argument. **Requiring it is probably right** — it keeps the
   command pure and the featurizer already exists.

Then emit one **2-class** set per partner (not one joint N-class set) and run
`ms_mrmp_select` over each.

Why pairs, measured: a joint binstring forces the thin class to oppose every
neighbour at once, and that joint requirement is where its noise concentrates.
On DG-po the 3-class set separated at 0.228 against 0.317 and 0.487 for the two
pairs that replaced it.

Partner candidates must include **other thin classes**, not only deep ones: PC
loses more held-out cells to VLMC-Pia (17.1%) than to its deep partner VLMC
(15.9%).

Do **not** use a binstring-Hamming score to rank partners. It was tried and
dropped: it predicts *which* classes need help slightly better (Pearson +0.79
vs +0.70) but its ranks 2–3 are a binstring-centroid artifact — CGE-Vip comes
out second-nearest for 6 of 8 thin classes and PAL-Inh for 4 — and gating on it
gave ANP one partner where three were wanted. Measured end to end: three
A-ranked partners scored identically to one (ANP 0.607 both ways).

### `mrmp-build-neighbor` — but not as a port

WPGMA over CpG-weighted Hamming, two-stage cut (height 0.011, then sub-split to
3). All inputs are already in the `.mrmp`.

**Do not port the partition logic.** It is the wrong shape. Measured on the
10-fold arm: **73.5% of all remaining error sits in confusion pairs no satellite
covers.** WPGMA partitions, so each class lands in at most one block, and the
3-class cap then truncates — IT-L4 and MGE-Pvalb fall out of the vocabulary
entirely, and PAL-Inh ↔ LSX-Inh is split across two blocks. The top ten
uncovered pairs alone carry 35.7% of error, and every one is symmetric.

Implement it as **overlapping closest-N pairs** instead, the same shape
`mrmp-build-thin` uses. Curated blocks (see the org) are the complementary
input path: nine of them took uncovered error from 73.5% to 25.3%.

Keep blocks **small**. Pattern count grows as 2^N−2 while covered error does
not: a 2-class block spends 2 pooled columns for 7.4% of error, a 6-class block
spends 62 for 4.6%. The old objection to large blocks was dilution (median 4
CpGs per pattern at N=8); with a per-binstring CpG floor that no longer binds
and **the pooled column budget does**. Same conclusion, different reason.

## Two traps that cost real time

**1. yame's stat quantiles are 16-bin histogram edges.** `q95_0` / `q05_1` take
only 8 distinct values on a k/16 grid. A cutoff between edges acts as the nearer
one, so the tuned `q05_1 >= 0.60` has always enforced `>= 0.625`. This is
documented in `YAME/docs/llms.txt` — read the whole entry, not the first line.

`mrmp-build` now uses **exact betas**, so `0.25,0.6` there is genuinely more
permissive than the same string was in the shell pipeline. `0.25,0.625` is the
honest translation of what shipped.

**RESOLVED 2026-08-08.** The residual was the low side's edge convention.
`YAME/src/rowstat.c:147-151` uses OPPOSITE edges on the two sides:
`q95_0 = (qb+1)/NB` (upper edge of the low group's bin) but
`q05_1 = qb1/NB` (lower edge of the high group's). So the awk's tests are

    q05_1 >= HI   <=>   min1 >= ceil(16*HI)/16      (0.60 -> 0.625)
    q95_0 <= LO   <=>   max0 <  floor(16*LO)/16     (0.25 -> STRICT < 0.25)

`--qfilter 0.25` tests `<=`, so it admits CpGs whose expected-0 class sits at
beta EXACTLY 0.25. That is 26,530 CpGs, 100% of the difference, and the relation
is one-sided (shell is a strict subset of C, no shell-only CpGs). The beta==0.5
tie hypothesis contributes exactly zero here: with 2 classes a tie empties one
quantile group, yame emits NA, and both filters reject. It only bites at >=3
classes.

Bit-exact equivalent, verified by membership XOR over all 21,867,837 rows:

    methscope mrmp-build --qfilter 0.2499999,0.625 --max-frac-na 0

General rule: `LO_C = nextbelow(floor(16*LO)/16)`, `HI_C = ceil(16*HI)/16`.
There is no single exact value, since the shell's low test is `<` and
`--qfilter`'s is `<=`; any LO in [0.2499, 0.25) works.

The C is the more correct statistic: below 20 samples per side yame's q95/q05
ARE max/min (the skip loop needs n >= 20), so both compute the same thing and
yame merely snaps it to a 1/16 grid, always in the rejecting direction. The one
thing --qfilter cannot reproduce is genuine outlier tolerance, which yame gains
once a group holds >= 20 samples -- irrelevant for satellites, relevant if a
global binstring puts >= 20 classes on one side.

Still worth re-tuning against held-out margin rather than inheriting numbers
fitted to another tool's binning; the 26,530 extra CpGs are by construction the
marginal ones.

**2. An uncovered class is imputed, not skipped.** `resolve_cpg` assigns it the
majority digit, so its binstring bit is not a measurement and the q-filter must
never test it. `--max-frac-na` bounds the imputation. Note `--min-major-fold 10`
already discards most such CpGs as PNA, so the implicit `0` does less work than
it looks.

## Ranking, settled

Rank by `delta_mean`, not `delta_beta` (worst-case margin): the q-filter already
bounds the worst case. The two scored within 0.007 of each other on held-out
margin while picking 17–28% different CpGs, and mean degrades more gracefully as
a set grows. They are identical for 2 classes.

Ranking by delta beats the old ranking by depth — **+0.586 against +0.446** at
the same 10,000-CpG budget on ANP/ASC. Depth and contrast are near-orthogonal,
so depth ranking took an unbiased sample of whatever the filter admitted rather
than concentrating the discriminating positions. That is why it never looked
broken.

## Also outstanding, outside the C

- `featurize_fold.sh` writes `fold<K>/auto/{train,test}.msfm` and `order.txt`
  with **no arm suffix**, so two arms in one workdir silently overwrite each
  other. Worked around with symlinked workdirs; worth fixing.
- `inspect` reports the deep-only global as 34 classes, not 33. Probably the PNA
  column, but confirm rather than assume.
