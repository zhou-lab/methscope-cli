#!/usr/bin/env python3
"""Render docs/examples/*.sh into docs/index.html.

The runnable code on the page is written ONCE, as scripts under
docs/examples/. Each has a `## title:` line and, for a block the reader is
not meant to run here, a `## norun: <reason>` line; the reason is what the
page shows beside the block and what test/docs_gate.py reads to skip it. The
page carries one marker pair per script:

    <!-- example:25_classify -->
    <!-- /example -->

and this fills the pair with the script's body as a <pre><code> block,
re-applying the page's markup: comments, the subcommand after `methscope`,
and the training-artifact extensions that carry a tooltip.

    docs/build_examples.py            rewrite docs/index.html in place
    docs/build_examples.py --check    exit 1 if the page is behind the scripts,
                                      or shows a runnable block the gate would
                                      not run (a bare <pre><code> outside a
                                      marker pair)

`make docs` runs it; `make test` runs --check.
"""
import glob, html, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "docs", "index.html")
EXAMPLES = os.path.join(ROOT, "docs", "examples")
TIP_EXT = ("msur", "msui", "updecx")            # formats with a data-tip-src figure on the page

def load(name):
    path = os.path.join(EXAMPLES, name + ".sh")
    if not os.path.exists(path): sys.exit("build_examples.py: no script for marker example:%s" % name)
    title, norun, body = None, None, []
    for line in open(path, encoding="utf-8"):
        if line.startswith("#!"): continue
        if line.startswith("## title:"): title = line[9:].strip(); continue
        if line.startswith("## norun:"): norun = line[9:].strip(); continue
        body.append(line.rstrip("\n"))
    return title, norun, "\n".join(body).strip("\n")

def markup(text):
    """Escape, then the page's three spans. Order matters: comments first so a
    `#` inside one is not re-marked, then subcommands, then tooltips."""
    out = []
    for line in text.split("\n"):
        line = html.escape(line, quote=False)
        m = re.match(r"^(\s*)(#.*)$", line) or re.match(r"^(.*?\S\s+)(#.*)$", line)
        if m:
            code, cm = m.group(1), '<span class="cm">%s</span>' % m.group(2)
        else:
            code, cm = line, ""
        # the subcommand links to its entry on the Reference tab (docs/build_help.py)
        code = re.sub(r"\bmethscope (?=[a-z])([a-z0-9_-]+)", r'methscope <a class="sub" href="#ref-\1">\1</a>', code)
        code = re.sub(r"(?<![\w/.-])([\w.-]+\.(%s))\b" % "|".join(TIP_EXT),
                      lambda mm: '<span class="tip" data-tip="%s" tabindex="0">%s</span>' % (mm.group(2), mm.group(1)), code)
        out.append(code + cm)
    return "\n".join(out)

def render(name):
    title, norun, body = load(name)
    attrs = ' data-norun="%s"' % html.escape(norun, quote=True) if norun else ""
    return "<!-- example:%s -->\n<pre%s><code>%s</code></pre>\n<!-- /example -->" % (name, attrs, markup(body))

PAIR = re.compile(r"<!-- example:([0-9A-Za-z_]+) -->.*?<!-- /example -->", re.S)

def build(page):
    return PAIR.sub(lambda m: render(m.group(1)), page)

def strays(page):
    """Runnable-looking blocks the gate would never run: a <pre><code> that is
    not inside a marker pair and carries neither data-norun nor a result class."""
    stripped = PAIR.sub("", page)
    return [m.group(0)[:60] for m in re.finditer(r"<pre([^>]*)><code>", stripped)
            if "data-norun" not in m.group(1) and 'class="' not in m.group(1)]   # res / help blocks carry a class

def main():
    page = open(PAGE, encoding="utf-8").read()
    names = PAIR.findall(page)
    on_disk = sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(EXAMPLES, "*.sh")))
    missing = sorted(set(on_disk) - set(names))
    if missing: sys.exit("build_examples.py: scripts with no marker on the page: " + " ".join(missing))
    new = build(page)
    if "--check" in sys.argv:
        bad = strays(page)
        if bad: sys.exit("build_examples.py: runnable-looking blocks outside docs/examples/: " + "; ".join(bad))
        if new != page: sys.exit("docs/index.html is behind docs/examples/: run `make docs`")
        print("examples are current (%d blocks)" % len(names)); return
    if new != page:
        open(PAGE, "w", encoding="utf-8").write(new); print("rendered %d example blocks into docs/index.html" % len(names))
    else:
        print("example blocks unchanged")

if __name__ == "__main__": main()
