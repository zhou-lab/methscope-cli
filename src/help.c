// SPDX-License-Identifier: AGPL-3.0-or-later
/* Shared help renderer. Subcommand usage text is written as plain strings and
 * printed through ms_help(), which adds ANSI styling only when the destination
 * is a TTY (so redirected/piped `-h` output stays byte-for-byte plain):
 *   - a section header  — a line at column 0 whose leading word(s) end in ':'
 *     (Usage:, Purpose:, Arguments:, Options:, Required:, ...) — is bold;
 *   - an option line    — indented and starting with '-' — has its flag token
 *     (up to the description gap) accent-coloured. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "methscope.h"

void ms_help(FILE *out, const char *text) {
  if (!isatty(fileno(out))) { fputs(text, out); return; }
  const char *B = "\033[1m", *A = "\033[36m", *R = "\033[0m";
  for (const char *p = text; *p; ) {
    const char *nl = strchr(p, '\n');
    size_t len = nl ? (size_t)(nl - p) : strlen(p);
    if (!len) {
      /* blank line */
    } else if (p[0] != ' ' && p[0] != '\t') {
      /* column-0 line: bold a leading "<Words>:" header token */
      size_t c = 0;
      while (c < len && p[c] != ':') ++c;
      int header = c < len && c <= 24; /* max header-label width */
      for (size_t i = 0; header && i < c; ++i) {
        char ch = p[i];
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
              ch == ' ' || ch == '&' || ch == '/' || ch == '-'))
          header = 0;
      }
      if (header)
        fprintf(out, "%s%.*s%s%.*s", B, (int)(c + 1), p, R,
                (int)(len - c - 1), p + c + 1);
      else
        fwrite(p, 1, len, out);
    } else {
      /* indented line: accent the leading flag token of an option */
      size_t s = 0;
      while (s < len && (p[s] == ' ' || p[s] == '\t')) ++s;
      size_t e = s;                   /* flag token ends at the first 2-space gap */
      while (e < len && !(p[e] == ' ' && e + 1 < len && p[e + 1] == ' ')) ++e;
      if (s < len && p[s] == '-' && e < len) {
        /* an option definition (has a description gap) — accent its flag token;
         * a wrapped Usage continuation line has no gap and stays plain */
        fprintf(out, "%.*s%s%.*s%s%.*s", (int)s, p, A, (int)(e - s), p + s, R,
                (int)(len - e), p + e);
      } else {
        fwrite(p, 1, len, out);
      }
    }
    if (!nl) break;
    fputc('\n', out);
    p = nl + 1;
  }
}
