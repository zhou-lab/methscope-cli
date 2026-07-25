// SPDX-License-Identifier: AGPL-3.0-or-later
/* `methscope fetch` -- put the pretrained models on disk.
 *
 * The models are too large for git and live on HuggingFace, which left the
 * catalog stranded in the docs: a reader had to copy a URL out of a web page
 * and pick a directory, and nothing downstream knew where the file went. This
 * command is the tie -- it carries the catalog, so `fetch` with no argument
 * both lists what exists and says which of it is already local.
 *
 *   methscope fetch                 browse and pick, or list when not a terminal
 *   methscope fetch hg38_sex.ubjx   fetch by name (several args, or a group)
 *   methscope fetch all             fetch everything
 *
 * BROWSING AND PICKING ARE THE SAME ACT
 *   Nobody memorises eleven names, so a bare `fetch` on a terminal shows the
 *   catalog as a checkbox list and fetches what you tick -- there is no
 *   separate `list` command to drift out of step with this one. Off a
 *   terminal the identical catalog is printed and nothing is downloaded, so a
 *   container build or workflow step can never hang waiting for a keystroke.
 *   A named target never prompts either way. Downloading is the only thing in
 *   methscope that touches the network.
 *
 * STDOUT IS THE PATHS
 *   Every human-facing line goes to stderr; stdout carries one absolute path
 *   per file the command guarantees is present. Fetching is idempotent, so
 *   `$(methscope fetch NAME)` is a usable argument anywhere -- it downloads on
 *   the first run and just resolves on every one after, which is why the
 *   examples need no path variable at all.
 *
 * TRUST
 *   Every entry carries a SHA-256 compiled into the registry below, so a
 *   download is checked against a digest this binary already held rather than
 *   one fetched alongside the file. Nothing on either host can vouch for
 *   itself. Files land on a ".part" sibling and are renamed only after the
 *   digest matches, so an interrupted or corrupted fetch cannot leave
 *   something in the store that later reads as a valid model. */
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

#include "digest.h"
#include "ui.h"
#include "methscope.h"

#define MS_HF_BASE "https://huggingface.co/zhou-lab/methscope/resolve/main/"
#define MS_GH_BASE \
  "https://raw.githubusercontent.com/zhou-lab/methscope_data/main/test/"

/* Sizes are the published byte counts; they are the integrity check, so they
 * are updated together with a re-upload. `data` entries are the query .cg
 * fixtures every example runs against -- they come from a different host, so
 * each row carries its own base URL. */
typedef struct {
  const char *name, *file, *base;
  const char *genome;      /* hg38 / mm10 / hg38 chr20 */
  const char *kind;        /* what the file is */
  const char *runs;        /* the command that consumes it */
  const char *detail;      /* dimensions and provenance, wrapped when shown */
  const char *sha256;          /* pinned; a download that misses it is discarded */
  uint64_t bytes;
  int is_data;
  /* A YAME .cg is unusable without its .cg.idx, so the sibling rides along
   * rather than being a second name the user has to remember. */
  const char *sibling, *sibling_sha256;
  uint64_t sibling_bytes;
} ms_model_t;

static const ms_model_t CATALOG[] = {
  {"hg38_celltype.ubjx", "hg38_celltype.ubjx", MS_HF_BASE, "hg38",
   "xgboost classifier (.ubjx)", "methscope classify",
   "62 human cell types (Alpha, ASC, AT1, AT2, B Mem, B Naive, ...). Its 10,097 "
   "MRMP feature states ride inside the bundle, so a query .cg runs directly "
   "with no separate annotation file.",
   "e1f738290b05f824873b599894228a33b189ffbda0644ed2aad0e82e7768bc7c", 25661483, 0, NULL, NULL, 0},
  {"mm10_celltype.ubjx", "mm10_celltype.ubjx", MS_HF_BASE, "mm10",
   "xgboost classifier (.ubjx)", "methscope classify",
   "41 mouse-brain cell types. The mouse counterpart of hg38_celltype; same "
   "self-contained bundle layout.",
   "3bbdfeabbc59c2e5810eee06c07d24c8fc9826926dd59ddd167ff46b4cb028de", 22498764, 0, NULL, NULL, 0},
  {"hg38_sex.ubjx", "hg38_sex.ubjx", MS_HF_BASE, "hg38",
   "logistic classifier (.ubjx)", "methscope classify",
   "Female / Male from two X-inactivation features (Xa_lo, Xa_hi) over a "
   "3-state MRMP. At 30 KB it is the smallest bundle in the catalog -- the "
   "cheapest way to check an install end to end.",
   "3d1618f6ca24d9a9d5b0fe1806673540515e5f6e7d7950e7c8873a85bb5c742b", 30931, 0, NULL, NULL, 0},
  {"hg38_65celltypes.refx", "hg38_65celltypes.refx", MS_HF_BASE, "hg38",
   "NNLS signature panel (.refx)", "methscope deconv",
   "65 cell types x 15,300 split-MRMP patterns: 58 Zhou single-cell types plus "
   "7 Loyfer organ/blood types. Built for whole-body and cfDNA deconvolution "
   "and stays usable at very low coverage.",
   "ce16678dd410c83e3df1c6f1c9b912a796f69c77722b28fc83c5690df69a6c03", 28900711, 0, NULL, NULL, 0},
  {"hg38_10k1.updecx", "hg38_10k1.updecx", MS_HF_BASE, "hg38",
   "UPDEC1 block decoder (.updecx)", "methscope upscale",
   "Imputes one 10,000-CpG block (block 10k1) from 101 MRMP inputs, and carries "
   "an output-CpG mask so upscale still emits a whole-genome .cg. About 94% of "
   "binary calls correct from ~0.1% input coverage.",
   "8f7f2d6d42f64daaba5bf9cede5146fa52c0f62b5c76b3d5c6680c5c7554b1b7", 29075778, 0, NULL, NULL, 0},
  {"hg38_wg.updecx", "hg38_wg.updecx", MS_HF_BASE, "hg38",
   "UPDEC2 whole-genome decoder (.updecx)", "methscope upscale",
   "700 processing units covering all 29,401,795 hg38 CpGs from 1,000 MRMP "
   "inputs (beta + missing), trained on the 207-sample Loyfer atlas. At 2.8 GB "
   "it is larger than the rest of the catalog put together.",
   "4aa5968c86aeba228a2001996c6a31501fd98fc40ad383ba956c988101ae0b98", 2960796438ULL, 0, NULL, NULL, 0},

  {"human_hg38_celltypes.cg", "human_hg38_celltypes.cg", MS_GH_BASE, "hg38",
   "query methylomes (.cg, 4 records)", "methscope classify / deconv",
   "4 sorted Loyfer cells -- oligodendrocyte, pancreas beta, NK, monocyte -- "
   "with 23.6M CpGs covered and mean beta 0.847. Their names come from the "
   ".cg.idx that is fetched alongside.",
   "705973c8cdd475dbee6947952bff38fe4dea7fb44dc400c33fa9df4a0bc29a96", 6402003, 1,
   "human_hg38_celltypes.cg.idx", "ca868d49a73c7adf650bd6d58dbe3bb7e37f1472eef9b125fa195fd7ac69bb02", 96},
  {"human_hg38_immune_mixture.cg", "human_hg38_immune_mixture.cg", MS_GH_BASE, "hg38",
   "query methylome (.cg, 1 record)", "methscope deconv",
   "A simulated 70% macrophage / 30% monocyte mixture over 4,194,304 CpGs, mean "
   "beta 0.790. The right answer is known, so it doubles as a deconvolution "
   "check rather than just a demo input.",
   "9806da825b933d2475b3e1c07a5fc399e16279eb9ccd06f5031112304edfdbd4", 3644798, 1, NULL, NULL, 0},
  {"human_hg38_test.cg", "human_hg38_test.cg", MS_GH_BASE, "hg38",
   "sparse query methylome (.cg)", "methscope upscale",
   "23,857 CpGs covered out of 29.4M -- about 0.1% -- at mean beta 0.823. This "
   "is the upscaling input; the sparsity is the point.",
   "276fce3f98a2de653009a2cb489e5f9ea699234a1b72aecd1ea3ba03795554a6", 103698, 1, NULL, NULL, 0},
  {"human_hg38_test.truth.cg", "human_hg38_test.truth.cg", MS_GH_BASE, "hg38",
   "dense truth methylome (.cg)", "scoring upscale output",
   "The same cell sequenced deeply: 22.9M CpGs covered at mean beta 0.821. "
   "Score upscale's binary calls against it.",
   "63dd50e9b86b8abcb4c70c4927fd766026aa29999268f407187c16acfb7f6f6f", 1944547, 1, NULL, NULL, 0},
  {"human_hg38_40_celltypes_chr20.cg", "human_hg38_40_celltypes_chr20.cg", MS_GH_BASE,
   "hg38 chr20", "reference methylomes (.cg, 40 records)", "methscope mrmp-build",
   "40 Loyfer cell types on chr20 only (773,477 CpGs): one sample per type, "
   "spanning neurons, hepatocytes, pancreas, immune, epithelia, endothelium and "
   "muscle. Builds 116,450 patterns in about a second. 40 samples is "
   "mrmp-build's ceiling -- a pattern packs as a base-3 uint64 and "
   "3^40 < 2^64 < 3^41 -- and chr20 is what keeps it small enough to ship.",
   "259b05d9a0708727a800817566a6c8f165a1210f45a9ffae692a8cdcbfe08f2e", 41377893, 1,
   "human_hg38_40_celltypes_chr20.cg.idx", "e17bb044c67853f8954caaf1c6b77bf4ae1a7f7efb05280df796581ab1a0bca7", 2182},
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

/* $METHSCOPE_DATA_DIR, else ~/.cache/methscope -- the same shape as kycg's
 * KYCG_DATA_DIR / ~/.cache/kycg. Models and .cg fixtures share one directory,
 * so an example can name a single path. */
static const char *store_dir(const char *override) {
  static char buf[4096];
  if (override) return override;
  const char *env = getenv("METHSCOPE_DATA_DIR");
  if (env && *env) return env;
  const char *home = getenv("HOME");
  if (!home || !*home) fdie("cannot locate HOME; pass --store DIR", NULL);
  if (snprintf(buf, sizeof(buf), "%s/.cache/methscope", home) >= (int)sizeof(buf))
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
    "Usage: methscope fetch [options] [NAME ... | models | data | all]\n\n"
    "Download pretrained models (HuggingFace zhou-lab/methscope) and the query\n"
    ".cg example fixtures (GitHub methscope_data) into the local store. With no\n"
    "NAME it lists the catalog and what is already there. Never prompts, so it\n"
    "is safe inside a build or workflow step.\n\n"
    "Human-facing lines go to stderr; stdout is one absolute path per file, so\n"
    "$(methscope fetch NAME) resolves to a usable path and re-runs for free.\n\n"
    "  --store DIR   store location (default $METHSCOPE_DATA_DIR,\n"
    "                else ~/.cache/methscope)\n"
    "  -f, --force   re-download and re-verify even if already present\n"
    "  -n, --dry-run show the plan and exit\n"
    "  --verify      re-check present files against their pinned sha256\n"
    "  -h, --help    show this help\n");
  return 1;
}

/* Entries are named by their file, so `fetch X` puts exactly X on disk. The
 * extension-less stem still resolves, since that is what earlier docs used. */
static const ms_model_t *find_model(const char *name) {
  for (int i = 0; i < N_CATALOG; ++i)
    if (!strcmp(CATALOG[i].name, name)) return &CATALOG[i];
  size_t n = strlen(name);
  for (int i = 0; i < N_CATALOG; ++i) {
    const char *f = CATALOG[i].file;
    if (!strncmp(f, name, n) && f[n] == '.') return &CATALOG[i];
  }
  return NULL;
}

/* Print a description at 6-space indent, folded at ~72 columns. */
static void wrap_detail(const char *text) {
  const int width = 72;
  int col = 0;
  const char *w = text;
  while (*w) {
    while (*w == ' ') ++w;
    const char *e = w;
    while (*e && *e != ' ') ++e;
    int len = (int)(e - w);
    if (!len) break;
    if (!col) { printf("      "); col = 6; }
    else if (col + 1 + len > width) { printf("\n      "); col = 6; }
    else { putchar(' '); ++col; }
    fwrite(w, 1, (size_t)len, stdout);
    col += len;
    w = e;
  }
  if (col) putchar('\n');
}

static void browse(const char *dir) {
  char sz[32];
  uint64_t have = 0, total = 0;
  printf("store\t%s\n", dir);
  for (int data = 0; data < 2; ++data) {
    printf("\n%s\n", data ? "example data (GitHub methscope_data)"
                          : "models (HuggingFace zhou-lab/methscope)");
    for (int i = 0; i < N_CATALOG; ++i) {
      const ms_model_t *m = &CATALOG[i];
      if (m->is_data != data) continue;
      char path[4096];
      join(path, sizeof(path), dir, m->file);
      uint64_t on_disk = file_size(path);
      int ok = on_disk == m->bytes;
      human(m->bytes, sz, sizeof(sz));
      printf("  %-34s %-10s %9s   %s\n", m->name, m->genome, sz,
             ok ? "present" : on_disk ? "PARTIAL" : "-");
      printf("      %s \xc2\xb7 run by: %s\n", m->kind, m->runs);
      wrap_detail(m->detail);
      total += m->bytes;
      if (ok) have += m->bytes;
      if (m->sibling) total += m->sibling_bytes;
    }
  }
  human(total, sz, sizeof(sz));
  printf("\n%d entries, %s total", N_CATALOG, sz);
  human(have, sz, sizeof(sz));
  printf("; %s present\n", sz);
  printf("\nFetch with:  methscope fetch <name>[,<name>...]"
         "   (or 'models', 'data', 'all')\n");
}

#ifndef MS_HAVE_CURL
/* Built without libcurl: the catalog still browses; only the transfer is gone. */
static void fetch_one(const ms_model_t *m, const char *dir, int force, int quiet,
                      int verify) {
  (void)force; (void)quiet; (void)verify;
  fprintf(stderr, "[methscope] fetch: built without libcurl; download manually:\n"
          "  curl -L -o %s/%s \\\n    %s%s\n", dir, m->file, m->base, m->file);
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
static void fetch_one(const ms_model_t *m, const char *dir, int force, int quiet,
                      int verify) {
  char path[4096], part[4200], url[1024];
  join(path, sizeof(path), dir, m->file);
  if ((size_t)snprintf(part, sizeof(part), "%s.part", path) >= sizeof(part) ||
      (size_t)snprintf(url, sizeof(url), "%s%s", m->base, m->file) >= sizeof(url))
    fdie("path is too long", m->file);

  /* A present file is accepted on size: re-hashing on every call would make
   * $(methscope fetch NAME) read 2.8 GB to hand back a path. --verify asks for
   * the digest check explicitly, and a fresh download always gets one. */
  if (!force && file_size(path) == m->bytes) {
    if (verify) {
      char digest[65];
      if (ms_sha256_file(path, digest)) fdie("cannot read for verification", path);
      if (strcmp(digest, m->sha256)) {
        fprintf(stderr, "[methscope] fetch: %s is CORRUPT on disk\n"
                "  expected sha256 %s\n  got               %s\n"
                "  re-fetch it with: methscope fetch -f %s\n",
                m->name, m->sha256, digest, m->name);
        exit(1);
      }
      fprintf(stderr, "[methscope] fetch: %s verified\n", m->name);
    } else {
      fprintf(stderr, "[methscope] fetch: %s already present\n", m->name);
    }
    if (!quiet) puts(path);
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
  fprintf(stderr, "\r\033[2K");   /* erase the progress line, not just rewind */

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
  char digest[65];
  if (ms_sha256_file(part, digest)) {
    remove(part);
    fdie("cannot read the download back to verify it", part);
  }
  if (strcmp(digest, m->sha256)) {
    remove(part);
    fprintf(stderr, "[methscope] fetch: %s failed verification -- discarded\n"
            "  expected sha256 %s\n  got               %s\n",
            m->name, m->sha256, digest);
    exit(1);
  }
  if (rename(part, path)) fdie("cannot install downloaded model", path);
  char sz[32];
  human(got, sz, sizeof(sz));
  fprintf(stderr, "[methscope] fetch: %s -> %s (%s)\n", m->name, path, sz);
  if (!quiet) puts(path);   /* one line per requested file: $(...) stays usable */
}
#endif /* MS_HAVE_CURL */

/* Turn the printed catalog into a pick list, then hand the choices to the same
 * download path a named target uses. */
static int browse_and_fetch(const char *dir, int force, int dry) {
  char labels[N_CATALOG][96], notes[N_CATALOG][64], details[N_CATALOG][320];
  const char *items[N_CATALOG], *note_p[N_CATALOG], *detail_p[N_CATALOG];
  int missing[N_CATALOG], n = 0;
  for (int i = 0; i < N_CATALOG; ++i) {
    const ms_model_t *m = &CATALOG[i];
    char path[4096];
    join(path, sizeof(path), dir, m->file);
    int have = file_size(path) == m->bytes;
    char sz[32]; human(m->bytes, sz, sizeof(sz));
    snprintf(labels[n], sizeof(labels[n]), "%-34s %-10s %9s", m->name, m->genome, sz);
    snprintf(notes[n], sizeof(notes[n]), "  %s%s", m->kind, have ? " (present)" : "");
    /* the pane: what it is, what runs it, then the full description */
    snprintf(details[n], sizeof(details[n]), "%s \xc2\xb7 run by: %s%s%s \xe2\x80\x94 %s",
             m->kind, m->runs,
             m->sibling ? " \xc2\xb7 its .cg.idx comes too" : "",
             have ? " \xc2\xb7 already in the store" : "", m->detail);
    items[n] = labels[n]; note_p[n] = notes[n]; detail_p[n] = details[n];
    missing[n] = !have;
    ++n;
  }
  int fetch_now = 0;
  int *pick = ms_ui_multiselect("methscope fetch", items, note_p, detail_p,
                                (size_t)n, 0, &fetch_now);
  if (!pick) { fprintf(stderr, "[methscope] fetch: nothing selected\n"); return 0; }

  const ms_model_t *plan[N_CATALOG];
  int np = 0;
  uint64_t need = 0;
  for (int i = 0; i < n; ++i) {
    if (!pick[i]) continue;
    plan[np++] = &CATALOG[i];
    if (force || missing[i]) {
      need += CATALOG[i].bytes;
      if (CATALOG[i].sibling) need += CATALOG[i].sibling_bytes;
    }
  }
  free(pick);
  if (!np) { fprintf(stderr, "[methscope] fetch: nothing selected\n"); return 0; }

  char sz[32]; human(need, sz, sizeof(sz));
  fprintf(stderr, "%s%d file(s), %s to download -> %s%s\n",
          ms_ui_bold(), np, sz, dir, ms_ui_reset());
  if (dry) return 0;
  if (need && !fetch_now && !ms_ui_confirm("Fetch now?", 1)) {
    fprintf(stderr, "[methscope] fetch: cancelled\n");
    return 0;
  }
  mkdir_p(dir);
#ifdef MS_HAVE_CURL
  curl_global_init(CURL_GLOBAL_DEFAULT);
#endif
  for (int i = 0; i < np; ++i) {
    fetch_one(plan[i], dir, force, 0, 0);
    if (plan[i]->sibling) {
      ms_model_t sib = *plan[i];
      sib.file = plan[i]->sibling;
      sib.sha256 = plan[i]->sibling_sha256;
      sib.bytes = plan[i]->sibling_bytes;
      sib.sibling = NULL;
      fetch_one(&sib, dir, force, 1, 0);
    }
  }
#ifdef MS_HAVE_CURL
  curl_global_cleanup();
#endif
  return 0;
}

int main_fetch(int argc, char *argv[]) {
  const char *want[16] = {NULL}, *override = NULL;
  int n_want = 0, force = 0, dry = 0, verify = 0;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
    else if (!strcmp(a, "--store") && i + 1 < argc) override = argv[++i];
    else if (!strcmp(a, "-f") || !strcmp(a, "--force")) force = 1;
    else if (!strcmp(a, "-n") || !strcmp(a, "--dry-run")) dry = 1;
    else if (!strcmp(a, "--verify")) verify = 1;
    else if (a[0] == '-') { usage(); fdie("unrecognized or incomplete option", a); }
    else if (n_want < (int)(sizeof(want) / sizeof(want[0]))) want[n_want++] = a;
    else fdie("too many arguments", a);
  }
  const char *dir = store_dir(override);
  if (!n_want) {
    /* The picker *is* the catalog on a terminal, so printing the listing first
     * would leave the whole thing behind when the alternate screen closes --
     * and on stdout, which belongs to the paths. Off a terminal there is no
     * picker, so the listing is the whole answer. */
    if (ms_ui_interactive()) return browse_and_fetch(dir, force, dry);
    browse(dir);
    return 0;
  }

  /* Resolve every name before touching the network, so a typo in the third of
   * four names does not leave a half-finished store. */
  const ms_model_t *plan[32];
  int n = 0;
  for (int w = 0; w < n_want; ++w) {
    const char *g = want[w];
    if (!strcmp(g, "all") || !strcmp(g, "models") || !strcmp(g, "data")) {
      int only_data = !strcmp(g, "data"), only_models = !strcmp(g, "models");
      for (int i = 0; i < N_CATALOG; ++i) {
        if (only_data && !CATALOG[i].is_data) continue;
        if (only_models && CATALOG[i].is_data) continue;
        if (n == (int)(sizeof(plan) / sizeof(plan[0]))) fdie("too many names", g);
        plan[n++] = &CATALOG[i];
      }
      continue;
    }
    char buf[1024];
    if (snprintf(buf, sizeof(buf), "%s", g) >= (int)sizeof(buf))
      fdie("name list is too long", g);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
      const ms_model_t *m = find_model(tok);
      if (!m) {
        fprintf(stderr, "[methscope] fetch: unknown name '%s'; "
                "run 'methscope fetch' to list the catalog\n", tok);
        return 1;
      }
      if (n == (int)(sizeof(plan) / sizeof(plan[0]))) fdie("too many names", tok);
      plan[n++] = m;
    }
  }

  uint64_t need = 0;
  for (int i = 0; i < n; ++i) {
    char path[4096];
    join(path, sizeof(path), dir, plan[i]->file);
    if (force || file_size(path) != plan[i]->bytes) need += plan[i]->bytes;
    if (plan[i]->sibling) need += plan[i]->sibling_bytes;
  }
  char sz[32];
  human(need, sz, sizeof(sz));
  fprintf(stderr, "[methscope] fetch: %d file(s), %s to download -> %s\n",
          n, sz, dir);
  if (dry) return 0;

  mkdir_p(dir);
#ifdef MS_HAVE_CURL
  curl_global_init(CURL_GLOBAL_DEFAULT);
#endif
  for (int i = 0; i < n; ++i) {
    fetch_one(plan[i], dir, force, 0, verify);
    if (plan[i]->sibling) {          /* the .cg.idx is not a separate answer */
      ms_model_t sib = *plan[i];
      sib.file = plan[i]->sibling;
      sib.sha256 = plan[i]->sibling_sha256;
      sib.bytes = plan[i]->sibling_bytes;
      sib.sibling = NULL;
      fetch_one(&sib, dir, force, 1, verify);
    }
  }
#ifdef MS_HAVE_CURL
  curl_global_cleanup();
#endif
  return 0;
}
