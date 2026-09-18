#!/usr/bin/env python3
"""The ONE place `methscope <sub> -h` is collected.

Both references are built from this: the docs page's Reference tab
(build_help.py) and the agent-facing llms.txt (make_llms.py). They used to run
their own subprocesses with their own environments and their own ideas of which
subcommands to include -- the page rendered all seventeen, llms.txt carried four
in full -- so where they overlapped they could disagree, and on 2026-09-17 they
did. Same dump, same environment, no drift possible.

HOME and TERM are pinned because `fetch -h` prints the resolved store path and
the help printer adapts to a terminal; without that the output depends on who
runs `make docs`.
"""
import os
import re
import subprocess

ENV = {"HOME": "/home/user", "PATH": os.environ.get("PATH", ""),
       "LC_ALL": "C", "TERM": "dumb"}

def banner_sections(ms):
    """The bare banner, section by section: [(heading, [(name, description)])].
    A heading has no leading space; a command line is two spaces, the name, then
    a capitalised description (the shape test/t_usage.sh also parses)."""
    out = subprocess.run([ms], env=ENV, capture_output=True, text=True).stderr
    sections, heading = [], None
    for line in out.splitlines():
        m = re.match(r"^  ([a-z][a-z0-9-]+) +([A-Z].*)$", line)
        if m and heading is not None:
            sections[-1][1].append((m.group(1), m.group(2).strip()))
        elif line and not line.startswith(" ") and not line.startswith("Usage") \
                and not line.startswith("methscope"):
            heading = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
            sections.append((heading, []))
    return [(h, cmds) for h, cmds in sections if cmds]

def help_of(ms, *args):
    """`methscope <args> -h` as the tool prints it, stdout or stderr."""
    r = subprocess.run([ms, *args], env=ENV, capture_output=True, text=True)
    return (r.stdout or r.stderr).rstrip("\n")

def subcommands(ms):
    """Every subcommand the banner advertises, in banner order."""
    return [name for _, cmds in banner_sections(ms) for name, _ in cmds]
