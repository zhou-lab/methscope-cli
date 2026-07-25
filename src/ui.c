// SPDX-License-Identifier: AGPL-3.0-or-later
/* Terminal helpers for `methscope fetch`, following kycg's src/ui.c.
 *
 * TWO TIERS, ON PURPOSE
 *   A raw-mode checkbox list when the terminal can drive one, and a numbered
 *   list read from a single line when it cannot -- a pipe, a dumb TERM, or a
 *   session where raw mode fails. The fallback is not a degraded mode nobody
 *   tests; it is what runs whenever the widget cannot.
 *
 * NEVER BLOCK A SCRIPT
 *   Every entry point here is gated on ms_ui_interactive(), which requires
 *   *both* stdin and stderr to be a terminal. A pipeline, a container build,
 *   or a cron job therefore never reaches a prompt: fetch prints its catalog
 *   and exits instead of waiting for a keystroke nobody will type.
 *
 * STDERR ONLY
 *   The widget draws on stderr. `methscope fetch` puts one absolute path per
 *   file on stdout so $(methscope fetch NAME) composes, and a UI that wrote
 *   there would corrupt that contract. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include "ui.h"

static struct termios g_saved;
static int g_raw = 0;

int ms_ui_interactive(void) { return isatty(0) && isatty(2); }

static int color(void) {
  const char *t = getenv("TERM");
  return isatty(2) && !getenv("NO_COLOR") && t && strcmp(t, "dumb");
}
const char *ms_ui_dim(void)   { return color() ? "\033[2m"  : ""; }
const char *ms_ui_bold(void)  { return color() ? "\033[1m"  : ""; }
const char *ms_ui_green(void) { return color() ? "\033[32m" : ""; }
const char *ms_ui_cyan(void)  { return color() ? "\033[36m" : ""; }
const char *ms_ui_reset(void) { return color() ? "\033[0m"  : ""; }

static int raw_on(void) {
  if (tcgetattr(0, &g_saved)) return 0;
  struct termios t = g_saved;
  t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
  t.c_cc[VMIN] = 1; t.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSANOW, &t)) return 0;
  g_raw = 1;
  return 1;
}
static void raw_off(void) {
  if (g_raw) { tcsetattr(0, TCSANOW, &g_saved); g_raw = 0; }
}

/* Key codes above the ASCII range so arrows and letters share one switch. */
enum { K_UP = 256, K_DOWN, K_OTHER };

static int read_key(void) {
  int c = getchar();
  if (c == EOF) return 'q';
  if (c != 27) return c;
  int a = getchar();
  if (a != '[' && a != 'O') return 27;      /* bare Esc */
  int b = getchar();
  if (b == 'A') return K_UP;
  if (b == 'B') return K_DOWN;
  return K_OTHER;
}

static void draw(const char *title, const char *const *items,
                 const char *const *notes, size_t n, const int *on,
                 size_t cur, int first) {
  if (!first) fprintf(stderr, "\033[%zuA", n + 2);   /* rewind over the list */
  fprintf(stderr, "\033[2K%s%s%s\n", ms_ui_bold(), title, ms_ui_reset());
  for (size_t i = 0; i < n; ++i) {
    fprintf(stderr, "\033[2K %s%s%s %s%s%s  %s%s%s\n",
            i == cur ? ms_ui_cyan() : "", i == cur ? "\xe2\x9d\xaf" : " ",
            i == cur ? ms_ui_reset() : "",
            on[i] ? ms_ui_green() : "", on[i] ? "[x]" : "[ ]",
            on[i] ? ms_ui_reset() : "",
            items[i],
            notes && notes[i] ? ms_ui_dim() : "", notes && notes[i] ? notes[i] : "");
    if (notes && notes[i]) fprintf(stderr, "%s", ms_ui_reset());
  }
  fprintf(stderr, "\033[2K%s  space toggle · a all · n none · enter confirm · q cancel%s\n",
          ms_ui_dim(), ms_ui_reset());
}

static int *widget(const char *title, const char *const *items,
                   const char *const *notes, size_t n, int preselect) {
  if (!raw_on()) return NULL;
  int *on = calloc(n, sizeof(int));
  if (!on) { raw_off(); return NULL; }
  if (preselect) for (size_t i = 0; i < n; ++i) on[i] = 1;

  size_t cur = 0;
  int first = 1, cancelled = 0;
  for (;;) {
    draw(title, items, notes, n, on, cur, first);
    first = 0;
    int k = read_key();
    if (k == K_UP || k == 'k') cur = cur ? cur - 1 : n - 1;
    else if (k == K_DOWN || k == 'j') cur = (cur + 1) % n;
    else if (k == ' ') on[cur] = !on[cur];
    else if (k == 'a') for (size_t i = 0; i < n; ++i) on[i] = 1;
    else if (k == 'n') for (size_t i = 0; i < n; ++i) on[i] = 0;
    else if (k == '\r' || k == '\n') break;
    else if (k == 'q' || k == 27) { cancelled = 1; break; }
  }
  raw_off();
  if (cancelled) { free(on); return NULL; }
  return on;
}

/* "1-3,5" / "all" / "none" -> flags. Returns 0 on success. */
static int parse_selection(const char *s, int *on, size_t n) {
  while (*s == ' ') ++s;
  if (!strcmp(s, "all")) { for (size_t i = 0; i < n; ++i) on[i] = 1; return 0; }
  if (!strcmp(s, "none")) { for (size_t i = 0; i < n; ++i) on[i] = 0; return 0; }
  for (size_t i = 0; i < n; ++i) on[i] = 0;
  while (*s) {
    while (*s == ' ' || *s == ',') ++s;
    if (!*s) break;
    char *e = NULL;
    long a = strtol(s, &e, 10);
    if (e == s || a < 1 || (size_t)a > n) return -1;
    long b = a;
    if (*e == '-') {
      const char *t = e + 1;
      b = strtol(t, &e, 10);
      if (e == t || b < a || (size_t)b > n) return -1;
    }
    for (long i = a; i <= b; ++i) on[i - 1] = 1;
    s = e;
    if (*s && *s != ',' && *s != ' ') return -1;
  }
  return 0;
}

int *ms_ui_multiselect(const char *title, const char *const *items,
                       const char *const *notes, size_t n, int preselect) {
  if (!n || !ms_ui_interactive()) return NULL;
  int *on = widget(title, items, notes, n, preselect);
  if (on) return on;

  /* No usable raw mode: number the list and read one line. */
  on = calloc(n, sizeof(int));
  if (!on) return NULL;
  fprintf(stderr, "\n%s%s%s\n", ms_ui_bold(), title, ms_ui_reset());
  for (size_t i = 0; i < n; ++i)
    fprintf(stderr, "  %2zu. %s%s%s%s\n", i + 1, items[i],
            notes && notes[i] ? ms_ui_dim() : "",
            notes && notes[i] ? notes[i] : "", ms_ui_reset());
  for (;;) {
    fprintf(stderr, "  %sselect: 'all', 'none', or a list like 1-3,5 [none]%s ",
            ms_ui_dim(), ms_ui_reset());
    fflush(stderr);
    char buf[512];
    if (!fgets(buf, sizeof(buf), stdin)) { fputc('\n', stderr); free(on); return NULL; }
    buf[strcspn(buf, "\r\n")] = '\0';
    if (!parse_selection(*buf ? buf : "none", on, n)) return on;
    fprintf(stderr, "  %scould not read that; use 'all', 'none', or numbers 1..%zu%s\n",
            ms_ui_dim(), n, ms_ui_reset());
  }
}

int ms_ui_confirm(const char *question, int default_yes) {
  if (!ms_ui_interactive()) return default_yes;
  for (;;) {
    fprintf(stderr, "%s [%s] ", question, default_yes ? "Y/n" : "y/N");
    fflush(stderr);
    char buf[64];
    if (!fgets(buf, sizeof(buf), stdin)) { fputc('\n', stderr); return 0; }
    buf[strcspn(buf, "\r\n")] = '\0';
    if (!*buf) return default_yes;
    if (buf[0] == 'y' || buf[0] == 'Y') return 1;
    if (buf[0] == 'n' || buf[0] == 'N') return 0;
  }
}
