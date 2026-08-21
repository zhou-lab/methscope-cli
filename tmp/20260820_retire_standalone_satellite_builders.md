# proposal: retire `mrmp-build-thin` and `mrmp-build-neighbor`

**issue — dead-but-reachable subcommands.** From the session that owns the
classifier work. WZ raised this on 2026-08-20 and asked for a written opinion;
**I have not removed anything** — this is a proposal for whoever picks it up.

**I agree they should go**, and the case is stronger than "unused". Below is
what I verified, the one real cost, and the thing that must be recorded before
the delete rather than after.

## What they are now

Satellites for the shipped pipeline come from a **single** `mrmp-build` call:

```sh
mrmp-build --qfilter 0.30,0.70 --delta-mean-top 20000 --min-segregating 20000 \
           --satellite-n 5 --force ref.cg chain.mrmp
```

That is how both shipped models were built — `hg38_celltype.clfx` (3 satellites)
and `mm10_celltype_brain.clfx` (134). `--satellite-n` landed 2026-08-13
specifically to fold the step in; the `20260813_sat_pairs.py` +
`20260813_sat_build.sh` shell-outs were dropped from the command list then.

Neither shipped model calls `mrmp-build-thin` or `mrmp-build-neighbor`.

## Why remove rather than leave

**1. They are a third independent implementation, not a wrapper.**
`main_mrmp_build_thin` (`mrmp.c:2016`, ~431 non-blank lines) and
`main_mrmp_build_neighbor` (`mrmp.c:2467`, ~475) make **zero** calls to
`tree_satellites()` — the function the shipped path actually uses. Three
implementations of "pick class pairs and build a 2-class set for them", sharing
nothing.

That is the exact shape of the bug that cost this session a day: the flat and
tree scoring paths were two implementations of one featurization rule, both
naming their columns `P1..PN`, so every by-name guard matched while the CpGs
behind them differed. Right names, wrong behaviour, no error. Duplication that
*looks* consistent is the dangerous kind, and 900 lines of it is a lot of
surface for a divergence nobody is testing.

**2. They lost on measurement, not on neglect.** The 2026-08-14 arms tested
thin-specific satellite rules against uniform `--satellite-n` and they scored
worse — see that RESEARCH_LOG entry. This is not retiring an untried idea.

**3. Reachable and undocumented as deprecated.** They appear in `methscope -h`
with no hint they are superseded, which is the same "flag a user can pass and
read about, that nothing supports" failure the 2026-08-20 review note objects
to. Consistent with the flat single-booster removal on 08-15: one format, one
implementation, state of the art only.

## The one real cost, stated plainly

`--satellite-n` is **not a superset** of `mrmp-build-thin`. Thin selects on
class *thinness* (few reference cells) plus a projection distance; `--satellite-n`
selects nearest-by-Hamming. Removing thin deletes a distinct selection rule from
the CLI, not merely a duplicate spelling of the current one. I still think it
should go — it was measured and rejected, and git history keeps it — but the
commit message should say "removing a rule that lost", not "removing a
duplicate", or the next person reading the log will assume it was redundant.

## Do this before deleting

**Record a reproducibility pin.** Three labjournal scripts still invoke these
subcommands and will stop working against a current binary:

- `zhouw3/2026/20260809_p01_sweep.sh`
- `zhouw3/2026/20260809_p01_bench.sh`
- `zhouw3/2026/20260810_root_filter_sweep.sh`

These are records of how past results were produced, so the lab reproducibility
rule applies: the org must say how to re-run them. Note the removal commit in
`2025/20251216_methscope.org` next to those scripts — "runnable at
methscope-cli <sha>, before `mrmp-build-thin`/`-neighbor` were removed" — so a
reader can check out the pin instead of discovering the command is gone. Doing
this *after* the delete means someone hits an unrecognised-subcommand error
first and has to reconstruct why.

**Update `docs/llms.txt`**, which currently advertises both commands.

## Scope of the delete

- `main.c:117-118` — the two dispatch lines
- `mrmp.c:2016-2466`, `mrmp.c:2467-~2970` — the two implementations
- their declarations in `mrmp.h`, and the two lines in the top-level help
- check for helpers that become orphans once both are gone (the same sweep that
  found `digest.c` / `ui.c` unreferenced would catch these)

Keep `tree_satellites()`, `sat_hamming()`, `sat_tag()` — that is the live path.

## What I would not do in the same commit

Leave `--satellite-n` alone. It is what both shipped models were built with and
what the published accuracy numbers (0.903 human, 0.969 mouse-brain
class-balanced) are attached to. If the thin *rule* is ever wanted back it
should return as another `--satellite-*` option sharing `tree_satellites()`,
not as a fourth standalone implementation.
