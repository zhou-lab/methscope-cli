// SPDX-License-Identifier: AGPL-3.0-or-later
/* `methscope fetch` -- put the pretrained models on disk.
 *
 * The models are too large for git and live on HuggingFace, which left the
 * catalog stranded in the docs: a reader had to copy a URL out of a web page
 * and pick a directory, and nothing downstream knew where the file went. This
 * command is the tie -- it carries the catalog, so `fetch` with no argument
 * both lists what exists and says which of it is already local.
 *
 *   methscope fetch                 browse the catalog and the local store
 *   methscope fetch hg38_sex        fetch one (comma-separate for several)
 *   methscope fetch all             fetch everything
 *
 * It never prompts, so a container build or a workflow step cannot hang on it:
 * a bare `fetch` only lists, and a named target proceeds. Downloading is the
 * only thing in methscope that touches the network.
 *
 * Integrity is a size check against the catalog, not a cryptographic digest --
 * enough to catch a truncated or redirected-to-HTML transfer, and honest about
 * being no more than that. Files land on a ".part" sibling and are renamed only
 * after the size matches, so an interrupted fetch can never leave a
 * short file that later reads as a valid model. */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef MS_HAVE_CURL
#include <curl/curl.h>
#endif

#include "methscope.h"

#define MS_HF_BASE "https://huggingface.co/zhou-lab/methscope/resolve/main/"

/* Sizes are the published byte counts; they are the integrity check, so they
 * are updated together with a re-upload. */
typedef struct {
  const char *name, *file, *task, *framework, *labels;
  uint64_t bytes;
} ms_model_t;

static const ms_model_t CATALOG[] = {
  {"hg38_celltype", "hg38_celltype.ubjx", "cell-type annotation", "xgboost",
   "62 human cell types", 25661483},
  {"mm10_celltype", "mm10_celltype.ubjx", "cell-type annotation", "xgboost",
   "41 mouse-brain cell types", 22498764},
  {"hg38_sex", "hg38_sex.ubjx", "sex prediction", "logistic",
   "Female, Male (XCI markers)", 30931},
  {"hg38_65celltypes", "hg38_65celltypes.refx", "deconvolution (NNLS)", "refx",
   "65 types = 58 Zhou + 7 Loyfer", 28900711},
  {"hg38_10k1", "hg38_10k1.updecx", "CpG upscaling (one block)", "UPDEC1",
   "block 10k1, 10,000 CpGs", 29075778},
  {"hg38_wg", "hg38_wg.updecx", "CpG upscaling (whole genome)", "UPDEC2",
   "700 units over 29,401,795 CpGs", 2960796438ULL},
};
static const int N_CATALOG = (int)(sizeof(CATALOG) / sizeof(CATALOG[0]));

static void fdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] fetch: %s: %s\n", msg, arg);
  else fprintf(stderr, "[methscope] fetch: %s\n", msg);
  exit(1);
}

/* 25661483 -> "24.5 MB"; the catalog spans 30 KB to 2.8 GB. */
static void human(uint64_t n, char *out, size_t cap) {
  const char *u[] = {"B", "KB", "MB", "GB"};
  double v = (double)n;
  int i = 0;
  while (v >= 1024 && i < 3) { v /= 1024; ++i; }
  snprintf(out, cap, i ? "%.1f %s" : "%.0f %s", v, u[i]);
}

/* $METHSCOPE_HOME, else ~/.methscope/models. */
static const char *store_dir(const char *override) {
  static char buf[4096];
  if (override) return override;
  const char *env = getenv("METHSCOPE_HOME");
  if (env && *env) return env;
  const char *home = getenv("HOME");
  if (!home || !*home) fdie("cannot locate HOME; pass --store DIR", NULL);
  if (snprintf(buf, sizeof(buf), "%s/.methscope/models", home) >= (int)sizeof(buf))
    fdie("store path is too long", home);
  return buf;
}

static void mkdir_p(const char *path) {
  char tmp[4096];
  if (snprintf(tmp, sizeof(tmp), "%s", path) >= (int)sizeof(tmp))
    fdie("store path is too long", path);
  for (char *p = tmp + 1; *p; ++p) {
    if (*p != '/') continue;
    *p = '\0';
    if (mkdir(tmp, 0775) && errno != EEXIST) fdie("cannot create store", tmp);
    *p = '/';
  }
  if (mkdir(tmp, 0775) && errno != EEXIST) fdie("cannot create store", tmp);
}

/* dir + "/" + file into `out`; fatal rather than silently truncating. */
static void join(char *out, size_t cap, const char *dir, const char *file) {
  if ((size_t)snprintf(out, cap, "%s/%s", dir, file) >= cap)
    fdie("store path is too long", dir);
}

static uint64_t file_size(const char *path) {
  struct stat st;
  if (stat(path, &st) || !S_ISREG(st.st_mode)) return 0;
  return (uint64_t)st.st_size;
}

static int usage(void) {
  ms_help(stderr,
    "Usage: methscope fetch [options] [NAME[,NAME...] | all]\n\n"
    "Download pretrained models from HuggingFace (zhou-lab/methscope) into the\n"
    "local store. With no NAME it lists the catalog and what is already there.\n"
    "Never prompts, so it is safe inside a build or workflow step.\n\n"
    "  --store DIR   store location (default $METHSCOPE_HOME,\n"
    "                else ~/.methscope/models)\n"
    "  -f, --force   re-download even if the file is already complete\n"
    "  -n, --dry-run show the plan and exit\n"
    "  -h, --help    show this help\n");
  return 1;
}

static const ms_model_t *find_model(const char *name) {
  for (int i = 0; i < N_CATALOG; ++i)
    if (!strcmp(CATALOG[i].name, name)) return &CATALOG[i];
  /* accept the file name too, so a docs copy-paste works */
  for (int i = 0; i < N_CATALOG; ++i)
    if (!strcmp(CATALOG[i].file, name)) return &CATALOG[i];
  return NULL;
}

static void browse(const char *dir) {
  char sz[32];
  uint64_t have = 0, total = 0;
  printf("store\t%s\n\n", dir);
  printf("%-18s %-10s %-30s %-12s %s\n",
         "name", "size", "task", "framework", "local");
  for (int i = 0; i < N_CATALOG; ++i) {
    const ms_model_t *m = &CATALOG[i];
    char path[4096];
    join(path, sizeof(path), dir, m->file);
    uint64_t on_disk = file_size(path);
    int ok = on_disk == m->bytes;
    human(m->bytes, sz, sizeof(sz));
    printf("%-18s %-10s %-30s %-12s %s\n", m->name, sz, m->task, m->framework,
           ok ? "yes" : on_disk ? "PARTIAL" : "-");
    total += m->bytes;
    if (ok) have += m->bytes;
  }
  human(total, sz, sizeof(sz));
  printf("\n%d models, %s total", N_CATALOG, sz);
  human(have, sz, sizeof(sz));
  printf("; %s present\n", sz);
  printf("\nFetch with:  methscope fetch <name>[,<name>...]   (or 'all')\n");
}

#ifndef MS_HAVE_CURL
/* Built without libcurl: the catalog still browses; only the transfer is gone. */
static void fetch_one(const ms_model_t *m, const char *dir, int force) {
  (void)force;
  fprintf(stderr, "[methscope] fetch: built without libcurl; download manually:\n"
          "  curl -L -o %s/%s \\\n    %s%s\n", dir, m->file, MS_HF_BASE, m->file);
  exit(1);
}
#else
/* Following redirects means libcurl reports progress for the small redirect
 * bodies too; keying on the expected size renders only the real transfer. */
typedef struct { const char *name; uint64_t expect; int last; } ms_progress_t;

static int on_progress(void *ud, curl_off_t dltotal, curl_off_t dlnow,
                       curl_off_t ultotal, curl_off_t ulnow) {
  (void)ultotal; (void)ulnow;
  ms_progress_t *p = (ms_progress_t *)ud;
  if (dltotal <= 0 || (uint64_t)dltotal != p->expect) return 0;
  int pct = (int)(100 * dlnow / dltotal);
  if (pct == p->last) return 0;
  p->last = pct;
  fprintf(stderr, "\r[methscope] fetch: %s %3d%%", p->name, pct);
  fflush(stderr);
  return 0;
}

/* Download one model to <dir>/<file> via a .part sibling. */
static void fetch_one(const ms_model_t *m, const char *dir, int force) {
  char path[4096], part[4200], url[1024];
  join(path, sizeof(path), dir, m->file);
  if ((size_t)snprintf(part, sizeof(part), "%s.part", path) >= sizeof(part) ||
      (size_t)snprintf(url, sizeof(url), "%s%s", MS_HF_BASE, m->file) >= sizeof(url))
    fdie("path is too long", m->file);

  if (!force && file_size(path) == m->bytes) {
    fprintf(stderr, "[methscope] fetch: %s already complete\n", m->name);
    return;
  }
  FILE *fp = fopen(part, "wb");
  if (!fp) fdie("cannot create download file", part);

  CURL *h = curl_easy_init();
  if (!h) fdie("cannot initialize libcurl", NULL);
  curl_easy_setopt(h, CURLOPT_URL, url);
  curl_easy_setopt(h, CURLOPT_WRITEDATA, fp);
  curl_easy_setopt(h, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
  curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(h, CURLOPT_MAXREDIRS, 10L);
  curl_easy_setopt(h, CURLOPT_USERAGENT, "methscope/" METHSCOPE_VERSION);
  curl_easy_setopt(h, CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(h, CURLOPT_XFERINFOFUNCTION, on_progress);
  ms_progress_t prog = {m->name, m->bytes, -1};
  curl_easy_setopt(h, CURLOPT_XFERINFODATA, &prog);
  CURLcode rc = curl_easy_perform(h);
  long code = 0;
  curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
  curl_easy_cleanup(h);
  int closed = fclose(fp);
  fprintf(stderr, "\r");

  if (rc != CURLE_OK || closed) {
    remove(part);
    fprintf(stderr, "[methscope] fetch: %s failed (HTTP %ld): %s\n",
            m->name, code, curl_easy_strerror(rc));
    exit(1);
  }
  uint64_t got = file_size(part);
  if (got != m->bytes) {
    remove(part);
    fprintf(stderr, "[methscope] fetch: %s is %" PRIu64 " bytes, expected %"
            PRIu64 " -- discarded\n", m->name, got, m->bytes);
    exit(1);
  }
  if (rename(part, path)) fdie("cannot install downloaded model", path);
  char sz[32];
  human(got, sz, sizeof(sz));
  fprintf(stderr, "[methscope] fetch: %s -> %s (%s)\n", m->name, path, sz);
}
#endif /* MS_HAVE_CURL */

int main_fetch(int argc, char *argv[]) {
  const char *want = NULL, *override = NULL;
  int force = 0, dry = 0;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
    else if (!strcmp(a, "--store") && i + 1 < argc) override = argv[++i];
    else if (!strcmp(a, "-f") || !strcmp(a, "--force")) force = 1;
    else if (!strcmp(a, "-n") || !strcmp(a, "--dry-run")) dry = 1;
    else if (a[0] == '-') { usage(); fdie("unrecognized or incomplete option", a); }
    else if (!want) want = a;
    else fdie("too many arguments", a);
  }
  const char *dir = store_dir(override);
  if (!want) { browse(dir); return 0; }

  /* Resolve every name before touching the network, so a typo in the third of
   * four names does not leave a half-finished store. */
  const ms_model_t *plan[16];
  int n = 0;
  if (!strcmp(want, "all")) {
    for (int i = 0; i < N_CATALOG; ++i) plan[n++] = &CATALOG[i];
  } else {
    char buf[1024];
    if (snprintf(buf, sizeof(buf), "%s", want) >= (int)sizeof(buf))
      fdie("model list is too long", want);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
      const ms_model_t *m = find_model(tok);
      if (!m) {
        fprintf(stderr, "[methscope] fetch: unknown model '%s'; "
                "run 'methscope fetch' to list the catalog\n", tok);
        return 1;
      }
      if (n == (int)(sizeof(plan) / sizeof(plan[0]))) fdie("too many models", tok);
      plan[n++] = m;
    }
  }

  uint64_t need = 0;
  for (int i = 0; i < n; ++i) {
    char path[4096];
    join(path, sizeof(path), dir, plan[i]->file);
    if (force || file_size(path) != plan[i]->bytes) need += plan[i]->bytes;
  }
  char sz[32];
  human(need, sz, sizeof(sz));
  fprintf(stderr, "[methscope] fetch: %d model(s), %s to download -> %s\n",
          n, sz, dir);
  if (dry) return 0;

  mkdir_p(dir);
#ifdef MS_HAVE_CURL
  curl_global_init(CURL_GLOBAL_DEFAULT);
#endif
  for (int i = 0; i < n; ++i) fetch_one(plan[i], dir, force);
#ifdef MS_HAVE_CURL
  curl_global_cleanup();
#endif
  return 0;
}
