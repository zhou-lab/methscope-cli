# review note: dead flags and stale help in train.c / predict.c

**For whoever owns the classifier work.** These came out of a systematic sweep
(help text vs argument parsing, documented defaults vs initialisers, orphan
symbols) run on 2026-08-20 while cleaning up the deconvolution side. None of it
is mine and I have **not touched either file** — `src/train.c` and
`src/predict.c` are still uncommitted in the working tree, exactly as I found
them. Every line number below is against that working-tree state, so check they
still line up before acting.

Nothing here is urgent. The reason to record it is that three of these are
flags a user can pass, read about in `-h`, and get either a hard error or
silence from — which is the failure mode that cost us most of a day on the
deconvolution side (a rescue that parsed its flags and did nothing, and looked
like a null result rather than a bug).

## 1. `--hierarchy` — documented, unreachable on the default path

- documented `train.c:207`, parsed `train.c:529`
- but `main_train` sends every `--framework xgboost` (the default) to
  `main_train_tree` at `train.c:490-500`
- `main_train_tree` (`train.c:309-345`) does not parse it, so it falls to
  `else if (a[0]=='-') tdie("unrecognized or incomplete option")` at `train.c:344`
- reachable only for `threshold` / `logistic`, and its one use site
  (`train.c:877`) sits inside the `else /* ---- xgboost framework ---- */`
  branch opened at `train.c:747` — which those frameworks never enter

So the flag is documented, accepted by one code path, and used by another that
the first can never reach. Either wire it into `main_train_tree` or drop it from
`train_usage()`.

## 2. `--scalar-coverage` — documented, unusable by every route

- documented `train.c:211`, parsed `train.c:528`
- without `--data`: dies at `train.c:617`
- with `--framework threshold|logistic`: dies at `train.c:619` as "xgboost-only"
- with `--data` + xgboost: control never reaches `main_train` at all

Every path errors. Same fix options as (1).

## 3. `--include-pna` — help says it works, code says it is gone

- help at `train.c:143` and `train.c:164` describes it as a live option and
  documents "(default: excluded)"
- `train.c:533-535` makes it a hard error: "`--include-pna` is gone"
- consequently `include_pna` (`train.c:505`) can never be non-zero at its only
  use, `train.c:666`

The help should stop advertising it, and `train.c:164` is a dangling
"(default: excluded)." line left behind when the entry was removed.

## 4. `--balance-classes` — works, documented nowhere

Implemented at `train.c:244-258`, parsed at `train.c:312`, and absent from both
`train_usage()` and the tree help block at `train.c:330-341`. The inverse of the
problem above, and the easier one to fix.

## 5. `predict.c` — `--levels` / `--level` documented, then refused

- documented `predict.c:67,69` as working output options
- `predict.c:509-515` calls `pdie()`: "not implemented on the routing-tree path"
- `predict.c:73` points the reader at `classify-train --hierarchy`, which is
  itself unreachable — see (1)

## 6. smaller staleness

- `train.c:140` describes `<ref.cm>` as "the runtime .cm from `mrmp-export`",
  but the default (and only working) xgboost path requires `--data` plus a
  chain-bearing `.msfm` (`train.c:496-499`)
- `train.c:151` refers to `predict`; that command was renamed to `classify` in
  `7716c50` and only `classify` is dispatched (`main.c:85`)
- `train.c:152-154` — `-p`'s help is spliced mid-sentence: "...in the artifact's
  own order.\n after the 'Pna' backgrounds have been excluded, in the\n For an
  auto MRMP that order..."

## 7. context on what changed around you

`test/parity.sh` had been broken since `7716c50` (it invoked `predict`) and was
never updated. Removing the `matrix` command broke a second line of it. I fixed
the command name, dropped the feature-matrix leg with a comment, and made
`test/compare.R` skip that comparison rather than fail — so the probability
parity check runs again. If you want the feature-matrix comparison back, the
library that produced it (`ms_matrix_build`, `ms_mrmp_trim` in `matrix.c`) is
still there and still used by train/predict/msfm/upscale; only the CLI wrapper
and `ms_matrix_write_tsv` were removed.

Also flagged by the same sweep, outside train/predict and also untouched:
`digest.c` (SHA-256) and `ui.c` (interactive picker) are compiled and linked but
have no caller anywhere; `ms_booster_get_hier/_features/_binarize/_scalar_cov`
in `model.c` are never read although their setters are live, so that metadata is
written into every model and never read back by C; and `UPDEC1` (`updec.c`) is a
reader-only format nothing in the tree can produce.
