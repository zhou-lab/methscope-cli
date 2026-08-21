# response to `20260820_train_predict_review.md` — confirmed, with attribution

**From the session that owns the classifier work** (mouse41 + `hg38_celltype.clfx`
shipping, and the 2026-08-15 flat-format removal). Written at WZ's request so
another session can do the fixing; **I have changed nothing in response to this
review** — `src/train.c` and `src/predict.c` are still uncommitted in the working
tree exactly as the reviewer found them.

Short version: **all six findings are correct.** I verified each one, and three
of them are mine. One is a genuine regression that the review understates, and
one pair should *not* get the same fix.

## How I verified

Not by reading alone — findings 1–3 are reproducible from the built binary:

```sh
./methscope classify-train --data /dev/null --hierarchy /dev/null -o /tmp/x.clfx
#   -> unrecognized or incomplete option: --hierarchy
./methscope classify-train --data /dev/null --scalar-coverage -o /tmp/x.clfx
#   -> unrecognized or incomplete option: --scalar-coverage
```

Attribution comes from diffing against the committed baseline
(`git show HEAD:src/train.c`, `HEAD:src/predict.c`), which is what separates
"broken by the 08-15 work" from "broken for a while".

## Attribution

| # | finding | verdict | origin |
|---|---|---|---|
| 1 | `--hierarchy` unreachable | **confirmed, and worse than stated** | **regression, 2026-08-15** |
| 2 | `--scalar-coverage` unusable | confirmed | pre-existing at `HEAD` |
| 3 | `--include-pna` help vs error | confirmed | pre-existing at `HEAD` |
| 4 | `--balance-classes` undocumented | confirmed | uncommitted 2026-08-15 work |
| 5 | `--levels` / `--level` refused | confirmed | 2026-08-15 |
| 6 | staleness (`ref.cm`, `predict`, spliced `-p`) | confirmed | pre-existing at `HEAD` |

## The one correction: (1) and (2) are not the same problem

The review groups them and says "same fix options as (1)". They need opposite
fixes, because only one of them ever worked.

**`--hierarchy` was live last week and I broke it.** At `HEAD` the dispatch was

```c
if (has_data && !strcmp(fw, "xgboost")) return main_train_tree(argc, argv);   /* HEAD:460 */
```

so `--framework xgboost` *without* `--data` fell through to `main_train`, which
both parsed `--hierarchy` (`HEAD:489`) and used it (`HEAD:839`, inside the
`/* ---- xgboost framework ---- */` branch). That was a working route. The
08-15 change made `--data` mandatory for xgboost and sent every xgboost run to
`main_train_tree`, which does not parse the flag — so the only path that reached
it is gone. **Deleting it from `train_usage()` would quietly ratify a
regression.** It should be wired into `main_train_tree` instead.

**`--scalar-coverage` was already dead at `HEAD`**, independently of anything
I did: `HEAD:578` kills the no-`--data` route (`--scalar-coverage needs --data`)
and `HEAD:460` sends the `--data` route to the tree, which never parses it. Both
legs were closed before 08-15. Nothing can depend on it, so dropping it from the
help is the honest fix — or wire it up, but that is a feature request, not a
repair.

## What I would do with the three that are mine

**(1) `--hierarchy` — wire into `main_train_tree`.** This is the one worth real
effort, and it unblocks (5): the taxonomy that `--hierarchy` embeds is exactly
what `--levels`/`--level` print, so they are one feature split across two
commands. Fixing train without predict leaves the metadata written and
unreadable — which is already the complaint in the review's §7 about
`ms_booster_get_hier` never being read.

**(5) `--levels` / `--level` — my `pdie` is half a fix.** Refusing beats the
silent acceptance that was there before (it returned a bare header as if the
level collapse had happened), but I left the help advertising both flags, which
is the exact "documented, then refused" pattern the review objects to. Either
implement them on the tree path once (1) lands, or drop them from
`predict_usage()` — but do not leave help and behaviour disagreeing.

Same applies to `--probs`, which I guarded at the same time. That one is a
genuine design question rather than an oversight: a tree has no single softmax,
because each node scores its **own class subset** and a routed cell is scored by
several. Leaf-only distribution, or product along the path? Someone has to
decide before it can be implemented; the guard is correct until then.

**(4) `--balance-classes` — decide before documenting.** It is implemented
(`train.c:244-258`, weighting each row `nrow/(K*n_c)`) and works. But it was
built during the 2026-08-14 upsampling experiments and **measured neutral** —
loss weighting was one of six arms, none of which beat the 400-cells/class
baseline, and it is not used by either shipped model. So the choice is document
it as an available knob with that null result attached, or drop it. What it
should not do is stay in the tree undocumented, where the next person re-runs
the same experiment to find out what it does. My preference is to keep and
document, with the null result in the help so nobody expects it to help.

The pre-existing four (2, 3, 6, and the §7 orphans) are a separate cleanup and I
have no stake in them beyond agreeing they are real.

## One thing the review gets exactly right

The framing in its opening — that the dangerous ones are flags a user can pass,
read about in `-h`, and get silence or a hard error from. That is the same
failure class as the `.cg` featurization bug this session spent a day on: two
implementations of one rule, both named `P1..PN`, so every by-name guard matched
while the CpGs behind them differed. Right names, wrong behaviour, no error.
Help text that disagrees with the parser is the documentation-shaped version of
it.
