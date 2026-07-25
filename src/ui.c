// SPDX-License-Identifier: AGPL-3.0-or-later
/* Terminal helpers for `methscope fetch`, following kycg's src/ui.c.
 *
 * TWO TIERS, ON PURPOSE
 *   A full-screen checkbox list when the terminal can drive one, and a
 *   numbered list read from a single line when it cannot -- a pipe, a dumb
 *   TERM, or a session where raw mode fails. The fallback is not a degraded
 *   mode nobody tests; it is what runs whenever the widget cannot.
 *
 * THE ALTERNATE SCREEN
 *   The widget takes the whole terminal (\033[?1049h) and gives it back
 *   untouched on exit. That is what makes a fixed-height viewport reasonable:
 *   a long catalog scrolls inside the frame instead of shoving the user's
 *   scrollback off the top. raw_leave() is wired to atexit and to SIGINT and
 *   SIGTERM, so a ^C during the picker cannot strand a terminal in raw mode
 *   with a hidden cursor.
 *
 * KEYS
 *   The same bindings as `kycg fetch`: arrows or j/k move, space toggles,
 *   a/n select all or none, / filters, f fetches the checked entries, enter
 *   accepts, q or Esc cancels.
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
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/select.h>
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

static void raw_off(void) {
  if (!g_raw) return;
  /* Leave the alternate screen first, so the terminal restores what the user
   * had before we drew over it, then hand back cooked mode and the cursor. */
  fputs("\033[?1049l\033[?25h", stderr);
  tcsetattr(0, TCSAFLUSH, &g_saved);
  fflush(stderr);
  g_raw = 0;
}
static void on_signal(int sig) {
  raw_off();
  signal(sig, SIG_DFL);
  raise(sig);
}
static int raw_on(void) {
  if (tcgetattr(0, &g_saved)) return 0;
  struct termios t = g_saved;
  t.c_lflag &= (tcflag_t)~(ICANON | ECHO);
  t.c_cc[VMIN] = 1; t.c_cc[VTIME] = 0;
  if (tcsetattr(0, TCSAFLUSH, &t)) return 0;
  g_raw = 1;
  static int hooked = 0;
  if (!hooked) {
    atexit(raw_off);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    hooked = 1;
  }
  fputs("\033[?1049h\033[H\033[2J\033[?25l", stderr);
  fflush(stderr);
  return 1;
}

static int term_rows(void) {
  struct winsize ws;
  if (ioctl(2, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 4) return ws.ws_row;
  return 24;
}

/* Key codes above the ASCII range so arrows and letters share one switch. */
enum { K_UP = 256, K_DOWN, K_PGUP, K_PGDN, K_HOME, K_END, K_OTHER };

/* Escape both starts a sequence and is a key in its own right. Poll briefly
 * rather than blocking on a second byte, so pressing Esc exits instead of
 * hanging until the next keystroke arrives. */
static int read_key(void) {
  unsigned char c;
  if (read(0, &c, 1) != 1) return 'q';
  if (c != 27) return c;

  struct timeval tv = {0, 40000};           /* 40 ms, well above key repeat */
  fd_set fds; FD_ZERO(&fds); FD_SET(0, &fds);
  if (select(1, &fds, NULL, NULL, &tv) <= 0) return 27;   /* bare Esc */

  unsigned char a, b;
  if (read(0, &a, 1) != 1) return 27;
  if (a != '[' && a != 'O') return 27;
  if (read(0, &b, 1) != 1) return 27;
  switch (b) {
    case 'A': return K_UP;
    case 'B': return K_DOWN;
    case 'H': return K_HOME;
    case 'F': return K_END;
    case '5': { unsigned char t; read(0, &t, 1); return K_PGUP; }
    case '6': { unsigned char t; read(0, &t, 1); return K_PGDN; }
    default:  return K_OTHER;
  }
}

/* Repaint the whole frame: title, a viewport over the matching rows, and a
 * footer. Home-then-clear each line rather than clearing the screen, so the
 * repaint does not flicker. */
/* Wrap `text` at `width`, indented, into at most `max` lines. */
static int wrap_pane(const char *text, int width, int max) {
  int used = 0, col = 0;
  const char *w = text;
  while (*w && used < max) {
    while (*w == ' ') ++w;
    const char *e = w;
    while (*e && *e != ' ') ++e;
    int len = (int)(e - w);
    if (!len) break;
    if (!col) { fputs("\033[2K   ", stderr); col = 3; }
    else if (col + 1 + len > width) {
      fputc('\n', stderr); ++used;
      if (used >= max) break;
      fputs("\033[2K   ", stderr); col = 3;
    } else { fputc(' ', stderr); ++col; }
    fwrite(w, 1, (size_t)len, stderr);
    col += len; w = e;
  }
  if (col) { fputc('\n', stderr); ++used; }
  return used;
}

static void draw(const char *title, const char *const *items,
                 const char *const *notes, const char *const *details,
                 const size_t *view, size_t nview,
                 const int *on, size_t cur, size_t top, size_t nsel,
                 const char *filter, int filtering) {
  int rows = term_rows();
  int PANE = 4;                               /* the detail pane's budget */
  int avail = rows - 4 - PANE;
  if (avail < 3) avail = 3;
  fputs("\033[H", stderr);
  fprintf(stderr, "\033[2K%s%s%s   %s%zu of %zu selected%s\n",
          ms_ui_bold(), title, ms_ui_reset(),
          ms_ui_dim(), nsel, nview, ms_ui_reset());
  fprintf(stderr, "\033[2K%s%s%s\n",
          ms_ui_dim(), filtering || (filter && *filter) ? "" : " ", ms_ui_reset());
  if (filtering || (filter && *filter)) {
    fprintf(stderr, "\033[2A\033[2K  %s/%s%s%s\n\n",
            ms_ui_cyan(), ms_ui_reset(), filter, filtering ? "_" : "");
  }
  for (int r = 0; r < avail; ++r) {
    size_t i = top + (size_t)r;
    fputs("\033[2K", stderr);
    if (i >= nview) { fputc('\n', stderr); continue; }
    size_t k = view[i];
    fprintf(stderr, " %s%s%s %s%s%s  %s%s%s%s\n",
            i == cur ? ms_ui_cyan() : "", i == cur ? "\xe2\x9d\xaf" : " ",
            i == cur ? ms_ui_reset() : "",
            on[k] ? ms_ui_green() : "", on[k] ? "[x]" : "[ ]",
            on[k] ? ms_ui_reset() : "",
            items[k],
            notes && notes[k] ? ms_ui_dim() : "",
            notes && notes[k] ? notes[k] : "", ms_ui_reset());
  }
  /* Detail pane: what the row under the cursor actually is. */
  fputs("\033[2K\n", stderr);
  int used = 0;
  if (nview && details && details[view[cur]]) {
    struct winsize ws;
    int width = (ioctl(2, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 20) ? ws.ws_col - 2 : 78;
    used = wrap_pane(details[view[cur]], width, PANE - 1);
  }
  for (int r = used; r < PANE - 1; ++r) fputs("\033[2K\n", stderr);
  fprintf(stderr, "\033[2K%s  arrows/jk move · space toggle · a/n all/none · "
          "/ filter · f fetch · enter accept · q or Esc cancel%s\033[J",
          ms_ui_dim(), ms_ui_reset());
  fflush(stderr);
}

/* Case-insensitive substring, so /chr20 finds the reference. */
static int matches(const char *hay, const char *needle) {
  if (!needle || !*needle) return 1;
  size_t n = strlen(needle);
  for (const char *p = hay; *p; ++p) {
    size_t i = 0;
    while (i < n && p[i] &&
           (p[i] | 32) == (needle[i] | 32)) ++i;
    if (i == n) return 1;
  }
  return 0;
}

/* *ran is set when the widget actually drew, so the caller can tell a user's
 * cancel from "this terminal cannot run a widget" -- the first must be
 * obeyed, the second must fall back. */
static int *widget(const char *title, const char *const *items,
                   const char *const *notes, const char *const *details,
                   size_t n, int preselect, int *fetch_now, int *ran) {
  *ran = 0;
  if (!raw_on()) return NULL;
  *ran = 1;
  int *on = calloc(n, sizeof(int));
  size_t *view = calloc(n, sizeof(size_t));
  char filter[64] = {0};
  if (!on || !view) { free(on); free(view); raw_off(); return NULL; }
  if (preselect) for (size_t i = 0; i < n; ++i) on[i] = 1;

  size_t cur = 0, top = 0;
  int filtering = 0, cancelled = 0;
  for (;;) {
    size_t nview = 0;
    for (size_t i = 0; i < n; ++i) {
      const char *note = notes && notes[i] ? notes[i] : "";
      if (matches(items[i], filter) || matches(note, filter)) view[nview++] = i;
    }
    if (cur >= nview) cur = nview ? nview - 1 : 0;
    int avail = term_rows() - 4; if (avail < 3) avail = 3;
    if (cur < top) top = cur;
    if (cur >= top + (size_t)avail) top = cur - (size_t)avail + 1;
    size_t nsel = 0;
    for (size_t i = 0; i < n; ++i) nsel += (size_t)(on[i] != 0);

    draw(title, items, notes, details, view, nview, on, cur, top, nsel, filter, filtering);
    int k = read_key();

    if (filtering) {                       /* typing into the filter box */
      size_t fl = strlen(filter);
      if (k == '\r' || k == '\n' || k == 27) { filtering = 0; }
      else if (k == 127 || k == 8) { if (fl) filter[fl - 1] = '\0'; }
      else if (k >= 32 && k < 127 && fl + 1 < sizeof(filter)) {
        filter[fl] = (char)k; filter[fl + 1] = '\0'; cur = 0; top = 0;
      }
      continue;
    }
    switch (k) {
      case K_UP: case 'k': if (cur) --cur; else cur = nview ? nview - 1 : 0; break;
      case K_DOWN: case 'j': if (nview) cur = (cur + 1) % nview; break;
      case K_PGUP: cur = cur > (size_t)avail ? cur - (size_t)avail : 0; break;
      case K_PGDN: cur += (size_t)avail; if (cur >= nview) cur = nview ? nview - 1 : 0; break;
      case K_HOME: cur = 0; break;
      case K_END:  cur = nview ? nview - 1 : 0; break;
      case ' ': if (nview) { on[view[cur]] = !on[view[cur]]; if (cur + 1 < nview) ++cur; } break;
      case 'a': for (size_t i = 0; i < nview; ++i) on[view[i]] = 1; break;
      case 'n': for (size_t i = 0; i < nview; ++i) on[view[i]] = 0; break;
      case '/': filtering = 1; break;
      case 'f': if (fetch_now) *fetch_now = 1; goto done;   /* fetch the checked */
      case '\r': case '\n': goto done;
      case 'q': case 27: cancelled = 1; goto done;
      default: break;
    }
  }
done:
  raw_off();
  free(view);
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
                       const char *const *notes, const char *const *details,
                       size_t n, int preselect, int *fetch_now) {
  if (fetch_now) *fetch_now = 0;
  if (!n || !ms_ui_interactive()) return NULL;
  int ran = 0;
  int *on = widget(title, items, notes, details, n, preselect, fetch_now, &ran);
  if (on) return on;
  if (ran) return NULL;          /* the user cancelled: do not ask again */

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
