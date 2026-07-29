#!/usr/bin/env python3
"""Publish the agent-facing reference as docs/llms.txt.

    ./docs/make_llms.py ./methscope

An agent that scrapes index.html spends most of its tokens on styling and
figure markup it cannot use. This is the same information in a form it can act
on: what the artifacts are, how to get a model, the four commands it will
actually run, and the traps that are not guessable from the help text.

The per-command sections are the binary's real `-h` output, so they cannot
drift from what the tool prints -- the same contract docs/sync_help.py relies
on. The prose around them is maintained here, because none of it is derivable
from a usage string. Re-run after changing a usage string or shipping a model.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "llms.txt"

## Full help for what an agent actually runs. The training and artifact-build
## commands (upscale-train, upscale-featurize, classify-train, mrmp-build, ...)
## are deliberately summarized instead: together they are ~11 KB of help for
## work that needs a GPU, a truth atlas and hours, and an agent that truly needs
## them can run `methscope <cmd> -h` itself.
FULL = ["upscale", "classify", "deconv", "inspect"]

PREAMBLE = """\
# methscope
# Agent-facing reference for the methscope CLI (DNA methylation inference).
# Canonical: https://zhou-lab.github.io/methscope-cli/llms.txt
# Human docs: https://zhou-lab.github.io/methscope-cli/
# Source of truth: docs/make_llms.py + the binary's own --help.

## What it is

methscope infers cell-type composition, sample labels and CpG-level
methylation from SPARSE methylomes -- the regime where a sample carries a few
thousand to a few hundred thousand observed CpGs rather than whole-genome
coverage. Pure C, no Python or R at inference, ~2 s per sample.

Input is a YAME `.cg` store (see the yame tool). Models are self-contained
bundles that carry their own feature definition:
  .updecx  upscale decoder   -- impute genome-wide CpG methylation
  .ubjx    classifier        -- cell type, sex
  .refx    deconvolution ref -- cell-type proportions

## Install

  conda install -c zhou-lab -c conda-forge methscope
  conda install -c zhou-lab -c conda-forge methscope-cuda   # linux-64, only
                                                            # upscale-train
                                                            # needs a GPU

## Getting models

Models live on HuggingFace (zhou-lab/methscope) and are fetched through YAME's
shared registry, which verifies each file against a pinned digest:

  yame fetch methscope/hg38/models      # or mm10/models, hg38/data

`methscope fetch` was RETIRED -- do not use it. Available:
  hg38_wg.updecx        whole-genome upscale decoder (human)
  mm10_wg.updecx        whole-genome upscale decoder (mouse)
  hg38_celltype.ubjx    cell-type classifier (human, 62 types)
  mm10_celltype.ubjx    cell-type classifier (mouse brain, 41 types)
  hg38_sex.ubjx         sex classifier
  hg38_65celltypes.refx deconvolution reference (65 cell types)
  hg38_10k1.updecx      upscale decoder for one 10k-CpG block (legacy, small)

## Traps

These are the mistakes that are not visible in the usage strings.

1. `upscale` writes CONTINUOUS methylation fractions (YAME format 4). It used
   to threshold at 0.5 into 0/1 calls (format 6); `--binary` still does, but it
   is lossy and only appropriate when a downstream tool demands format 6.
   `--probs` emits a TSV of the same values.

2. One training msur serves MANY models. `upscale-featurize` stores the raw
   per-pattern summary (beta, covered count, observed set), not encoder input;
   `--features` and `--patterns` are chosen at TRAINING time and are just
   projections of it. So do not rebuild a msur to change either. `--patterns P`
   may be narrowed below the msur's pattern count but never widened past it, so
   featurize at the widest vocabulary you might want.

3. `inspect` describes ANY artifact -- bundle, .mrmp, .msui, .msur -- and is
   the fastest way to find out what an unlabeled file is and what it expects.

4. A bundle records what is needed to RUN a model, nothing about how it was
   trained. Provenance for the shipped models lives in the lab journal, not in
   the file.

## Commands
"""


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else str(HERE.parent / "methscope")

    def help_of(*args):
        r = subprocess.run([binary, *args], capture_output=True, text=True)
        return (r.stdout + r.stderr).rstrip()

    top = help_of("-h")
    ## The one-line summaries in the top-level help are the summary list; keep
    ## them so an agent knows a command exists before deciding to ask for its -h.
    summary = [l for l in top.splitlines() if l.startswith("  ") and l[2:3].isalpha()]

    parts = [PREAMBLE, "\n".join(summary), ""]
    for cmd in FULL:
        parts += [f"\n### methscope {cmd}\n", help_of(cmd, "-h"), ""]
    parts += ["\n### Other commands\n",
              "Run `methscope <command> -h` for these; they build or train\n"
              "artifacts and need a truth atlas, and upscale-train needs a GPU.\n"]

    OUT.write_text("\n".join(parts).rstrip() + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
