#!/usr/bin/env python3
"""Publish the agent-facing reference as docs/llms.txt.

    ./docs/make_llms.py ./methscope

An agent that scrapes index.html spends most of its tokens on styling and
figure markup it cannot use. This is the same information in a form it can act
on: what the artifacts are, how to get a model, the four commands it will
actually run, and the traps that are not guessable from the help text.

The per-command sections are the binary's real `-h` output, so they cannot
drift from what the tool prints. The prose around them is maintained here,
because none of it is derivable from a usage string. Re-run after changing a
usage string or shipping a model.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpdump import banner_sections, help_of   # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "llms.txt"


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
  .clfx    classifier        -- cell type, sex (formerly .ubjx, still accepted)
  .msdref  deconvolution ref -- cell-type proportions

## Install

  conda install -c zhou-lab -c conda-forge methscope
  conda install -c zhou-lab -c conda-forge methscope-cuda   # linux-64; only
                                                            # upscale-train
                                                            # uses a GPU, and
                                                            # only to go faster

## Getting models

Models live on HuggingFace (zhou-lab/methscope) and are fetched through the
registry compiled into this binary, which verifies each file against a pinned
digest. `methscope fetch` needs no other tool installed:

  methscope fetch -c hg38/models/hg38_celltype_lite.clfx   # one file
  methscope fetch -y hg38/models                           # the whole directory

A directory is gigabytes, so `fetch` asks before it starts and refuses outright
when nothing can answer -- a script, a batch job -- which is what `-y` settles.
`-c` fetches into the current directory instead of the store. `methscope fetch`
with no argument browses; `methscope fetch -l` dumps the catalogue as TSV, which
is the machine-readable form to read before fetching anything.

MODEL_TABLE

Example queries with KNOWN answers live beside the models in hg38/data:
`human_hg38_immune_mixture.cg` holds nine mixtures whose truths are exact by
construction, named by composition and sparsity (mac70_mono30_2pow22 is
macrophage 0.70 / monocyte 0.30 over 2^22 binarized CpGs). Fetch its
`.cg.idx` too -- the record names are in it, and without them the file is
nine anonymous records.

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


def model_table(ms):
    """One line per catalogued model: the file, and its title from the suite
    file table. Both come from where the docs page's model cards get them --
    `fetch -l` for what this binary actually pins, files.tsv for the prose --
    so this list cannot name a withdrawn file or miss a new one. It rotted
    once: llms.txt sat at the v9 set, naming two files v10 had withdrawn and
    none of the four bank classifiers."""
    import csv, os, subprocess, tempfile
    ## The suite file table replaced YAME/data/assets.tsv at YAME v1.50: one
    ## row per file, no header line, fixed columns. `fetch -l` carries no
    ## title column, which is why the prose still comes from the table.
    tsv = os.path.join(HERE.parent, "YAME", "tools", "registry", "files.tsv")
    title = {}
    with open(tsv) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) >= 7:
                title[os.path.basename(f[1])] = f[6]
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, METHSCOPE_DATA_HOME=d)
        out = subprocess.run([ms, "fetch", "-l"], env=env, capture_output=True,
                             text=True, check=True).stdout
    lines = []
    for line in out.splitlines()[1:]:
        t = line.split("\t")
        if not t[0].endswith("/models"):
            continue
        lines.append("  %-24s %s" % (t[4], title.get(t[4], "")))
    return "\n".join(lines)

def main():
    argv = [a for a in sys.argv[1:] if a != "--check"]
    binary = argv[0] if argv else str(HERE.parent / "methscope")

    top = help_of(binary, "-h")
    ## The one-line summaries in the top-level help are the summary list; keep
    ## them so an agent knows a command exists before deciding to ask for its -h.
    summary = [l for l in top.splitlines() if l.startswith("  ") and l[2:3].isalpha()]

    ## EVERY subcommand, in banner order, from the same -h dump the docs page's
    ## Reference tab is built from (docs/helpdump.py). This file used to carry
    ## four in full and send the reader back to `-h` for the rest, which made it
    ## a different document rather than the same one in another format -- and
    ## where the two overlapped they had drifted apart by 0.10. An agent cannot
    ## run `-h` on a machine it is only reading docs for, so "ask the tool" was
    ## never an answer here anyway.
    parts = [PREAMBLE.replace("MODEL_TABLE", model_table(binary)),
             "\n".join(summary), ""]
    for heading, cmds in banner_sections(binary):
        parts.append("\n## %s\n" % heading)
        for name, desc in cmds:
            parts += ["\n### methscope %s -- %s\n" % (name, desc),
                      help_of(binary, name, "-h"), ""]

    text = "\n".join(parts).rstrip() + "\n"
    ## --check is what keeps this file honest. It shipped stale for three weeks
    ## -- a retirement notice for a command that works, the withdrawn v9 model
    ## list, and a --flat flag the binary refuses -- because nothing compared it
    ## with the binary. `make test` runs this.
    if "--check" in sys.argv:
        have = OUT.read_text() if OUT.exists() else ""
        if have != text:
            print("llms.txt is behind the binary; run docs/make_llms.py ./methscope",
                  file=sys.stderr)
            sys.exit(1)
        print("llms.txt is current")
        return
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
