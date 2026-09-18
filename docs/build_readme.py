#!/usr/bin/env python3
"""Keep the README's version example equal to what the binary prints.

    ./docs/build_readme.py ./methscope [--check]

The README quotes `methscope --version` to explain that the catalogue and the
model tag are compiled in. That quote is three numbers -- the methscope version,
the bundled YAME tag, the models tag -- and every one of them moves on its own
schedule, so a hand-typed copy is stale the first time any of them does. It sat
at `(yame v1.43, models v9)` through two YAME releases and a model tag while the
binary printed v1.45 / v10.

Only the parenthesised part is generated, between the markers; the sentence
around it is prose and stays hand-written.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
README = HERE.parent / "README.md"
BEGIN, END = "<!-- version:begin -->", "<!-- version:end -->"

def main():
    argv = [a for a in sys.argv[1:] if a != "--check"]
    binary = argv[0] if argv else str(HERE.parent / "methscope")
    out = subprocess.run([binary, "--version"], capture_output=True, text=True,
                         check=True).stdout.strip()
    m = re.search(r"\(([^)]*)\)", out)          # "methscope X (yame vY, models vZ)"
    if not m:
        sys.exit("build_readme.py: cannot find the (yame ..., models ...) part of %r" % out)
    page = README.read_text(encoding="utf-8")
    i, j = page.find(BEGIN), page.find(END)
    if i < 0 or j < 0:
        sys.exit("build_readme.py: version markers not found in README.md")
    new = page[:i] + BEGIN + "`(%s)`" % m.group(1) + page[j:]
    if "--check" in sys.argv:
        if new != page:
            sys.exit("README.md quotes a stale --version: run `make docs`")
        print("README version is current"); return
    if new != page:
        README.write_text(new, encoding="utf-8"); print("rewrote the README version quote")
    else:
        print("README version unchanged")

if __name__ == "__main__":
    main()
