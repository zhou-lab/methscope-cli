// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * `inspect` — describe a model bundle (.clfx / .updecx) without running
 * inference: the framework mark (kind), the on-disk section layout (each section's
 * offset + size + a short description of its contents), and a breakdown of the
 * `model` section by framework:
 *   xgboost            -> num_feature + embedded labels
 *   threshold/logistic -> method, labels, bias, scale, per-feature weight/mean
 *   upscale (UPDEC1)   -> n_in / n_hidden / n_out from the decoder header
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "methscope.h"
#include "msfm.h"
#include "mrmp.h"
#include "bundle.h"
#include "bmeta.h"
#include "updec2.h"
#include "cfile.h"     /* open_cfile, read_cdata1, decompress, fmt2_get_keys_n */
#include <xgboost/c_api.h>

static void idie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] inspect: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] inspect: %s\n", msg);
  exit(1);
}

static const char *commafmt(unsigned long long v, char *buf);

/* Standalone deconvolution reference. Kept here rather than importing
 * deconv.c's private reader: inspect needs only the fixed 64-byte header. */
#define MSDREF_MAGIC "MSDREF1"
typedef struct __attribute__((packed)) {
  char magic[8];
  uint32_t version, n_class;
  uint64_t n_row, n_keep;
  double qlo, qhi, beta_thr;
  uint32_t mincov, reserved;
} msdref_header_t;

static int inspect_msdref(const char *path) {
  msdref_header_t h;
  struct stat st;
  int fd = open(path, O_RDONLY);
  if (fd < 0) idie("cannot open", path);
  ssize_t got = read(fd, &h, sizeof h);
  if (fstat(fd, &st) || close(fd)) idie("cannot stat", path);
  if (got != (ssize_t)sizeof h || memcmp(h.magic, MSDREF_MAGIC, 7) ||
      (h.version != 1 && h.version != 2) || !h.n_class || !h.n_keep ||
      h.n_keep > h.n_row)
    idie("invalid .msdref header", path);
  char rows[32], kept[32], bytes[32];
  printf("deconvolution reference  MSDREF1/v%u, %s bytes\n\n",
         h.version, commafmt((unsigned long long)st.st_size, bytes));
  printf("  %-12s %u\n", "cell types", h.n_class);
  printf("  %-12s %s\n", "CpGs kept", commafmt(h.n_keep, kept));
  printf("  %-12s %s\n", "CpG row space", commafmt(h.n_row, rows));
  printf("  %-12s %.2f,%.2f\n", "qfilter", h.qlo, h.qhi);
  printf("  %-12s %.2f\n", "beta cut", h.beta_thr);
  printf("  %-12s %u\n", "min coverage", h.mincov);
  return 0;
}

static int inspect_usage(void) {
  ms_help(stderr,
    "\n"
    "Usage:\n"
    "  methscope inspect <FILE>\n"
    "\n"
    "Purpose:\n"
    "  Describe any methscope artifact without running it. The format is detected\n"
    "  from its magic:\n"
    "    .clfx/.updecx        bundle: kind mark, section layout, model breakdown\n"
    "    .mrmp   MRMPIDX1  pattern set: dimensions, binstring parameters, top ranks\n"
    "    .msui   MSUIDX1   processing-unit index: units, memberships, CpG split\n"
    "    .msur   MSURAW2/3 training msur: cells, replicates, embedded truth\n"
    "    .msfm   MSFMAT1   feature matrix: records, patterns, labels, coverage\n"
    "    .msdref MSDREF1   deconvolution reference: cell types and CpG rows\n"
    "\n"
    "Options:\n"
    "  --tree       render an mrmp-tree: its nodes, and which class each one\n"
    "               decides. Works on a chain or a tree bundle; the structure\n"
    "               is DERIVED, so there is no tree file to keep in step.\n"
    "  --patterns   .mrmp only: list the top-ranked patterns\n"
    "  --top K      .mrmp only: how many to list (default 20)\n"
    "  -h           Show this help message.\n"
    "\n");
  return 1;
}

static char *buf_to_tmp(const void *buf, size_t len) {
  char tmpl[] = "/tmp/methscope_insp_XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) idie("cannot create temp file", NULL);
  for (size_t off = 0; off < len; ) {
    ssize_t w = write(fd, (const char *)buf + off, len - off);
    if (w <= 0) idie("temp write error", NULL);
    off += (size_t)w;
  }
  close(fd);
  return strdup(tmpl);
}

/* States in the FIRST record of a .cm, and how many records it holds. The
 * record count matters since the chain-aware resolve: a multi-set .mrmp now
 * resolves to one record per set, and reporting only the first record's states
 * described a fraction of the mask as if it were the whole thing. -1 on
 * failure; *nrec_out is 0 then. */
static long mrmp_state_count(const char *cm_path, long *nrec_out) {
  if (nrec_out) *nrec_out = 0;
  cfile_t cf = open_cfile((char *)cm_path);
  cdata_t c  = read_cdata1(&cf);
  if (c.fmt != '2') { free_cdata(&c); bgzf_close(cf.fh); return -1; }
  cdata_t d = decompress(c);
  free_cdata(&c);
  long n = (long)fmt2_get_keys_n(&d);
  free_cdata(&d);
  long rec = 1;
  for (;;) {                                   /* count the remaining records */
    cdata_t e = read_cdata1(&cf);
    if (!e.n) { free_cdata(&e); break; }
    free_cdata(&e); ++rec;
  }
  bgzf_close(cf.fh);
  if (nrec_out) *nrec_out = rec;
  return n;
}

/* format an unsigned integer with thousands separators into buf (>= 32 bytes) */
static const char *commafmt(unsigned long long v, char *buf) {
  char tmp[24]; int n = snprintf(tmp, sizeof tmp, "%llu", v);
  int commas = (n - 1) / 3, len = n + commas;
  buf[len] = '\0';
  /* fill buf from the right: copy digits back-to-front, inserting a comma
     after every third digit */
  int buf_i = len - 1, out_i = n - 1, cnt = 0;
  while (out_i >= 0) {
    buf[buf_i--] = tmp[out_i--];
    if (++cnt % 3 == 0 && out_i >= 0) buf[buf_i--] = ',';
  }
  return buf;
}

/* print a ", "-joined label list, truncating with "..." past a char budget */
static void print_labels(char *const *labels, int K) {
  const int budget = 58;
  int used = 0, shown = 0;
  for (int i = 0; i < K; ++i) {
    int add = (shown ? 2 : 0) + (int)strlen(labels[i]);
    if (shown && used + add > budget) break;
    printf("%s%s", shown ? ", " : "", labels[i]);
    used += add; shown++;
  }
  if (shown < K) printf(", ... (%d total)", K);
  printf("\n");
}

/* ------------------------------------------------------- inspect --tree ----
 * Render the routing tree of a chain (or a tree bundle) as a reader sees it.
 *
 * Nothing here is stored. The structure derives completely from the chain: a
 * node's parent is its name minus the last dotted component, its classes are
 * the block's own sample names, and a class routes to whichever child covers
 * it -- or is decided here if none does. Verified against every prediction of
 * three cohorts (6,025 cells, 0 disagreements), which is why mrmp-tree no
 * longer writes a tree file: it would be a second copy of something the
 * artifact already answers. */
/* One structure, two edge kinds. A HARD child takes cells: it must partition
 * its parent's classes, and it owns a booster. A SOFT child (a satellite)
 * takes none -- it lends its columns to the parent's booster and has no model
 * of its own -- so soft siblings may OVERLAP, which is the whole point: in the
 * mouse leaf PAL-Inh is a member of 20 of them, impossible under a partition. */
typedef struct { char *name; uint32_t nc, npat; uint64_t cpg; char **cls;
                 int par; int soft; } inode_t;

static void itree_render(const inode_t *nd, uint32_t n, uint32_t k,
                         const char *pre, int last) {
  char b1[32], b2[32];
  /* the node line: what it decides between, and on how much evidence */
  /* The root sits at column 0 with no connector; everything below is indented
   * under its parent, so depth is readable at a glance. */
  int top = (*pre == '\0');
  /* The name field shrinks with depth so the numeric columns stay in one place
   * however deep the tree goes -- otherwise every level shifts them right and
   * the thing you actually compare between nodes stops lining up. */
  int used = (int)strlen(pre) + (top ? 0 : 4);
  int w = 26 - used; if (w < 8) w = 8;
  printf("%s%s%-*s %2u classes  %7s patterns  %9s CpGs\n", pre,
         top ? "" : (last ? "`-- " : "|-- "), w, nd[k].name,
         nd[k].nc, commafmt(nd[k].npat, b1), commafmt(nd[k].cpg, b2));
  char sub[512];
  snprintf(sub, sizeof sub, "%s%s", pre, top ? "  " : (last ? "    " : "|   "));
  uint32_t nhard = 0, nsoft = 0;
  for (uint32_t j = 0; j < n; ++j)
    if (nd[j].par == (int)k) { if (nd[j].soft) ++nsoft; else ++nhard; }
  /* A class is decided HERE unless a HARD child takes it. A soft child takes
   * no cells, so it never removes a class from this node's own list -- its
   * members are still decided here, just on wider evidence. */
  uint32_t shown = 0;
  for (uint32_t c = 0; c < nd[k].nc; ++c) {
    int taken = 0;
    for (uint32_t j = 0; j < n && !taken; ++j) {
      if (nd[j].par != (int)k || nd[j].soft) continue;
      for (uint32_t x = 0; x < nd[j].nc && !taken; ++x)
        taken = !strcmp(nd[j].cls[x], nd[k].cls[c]);
    }
    if (!taken) ++shown;
  }
  uint32_t item = 0, total = shown + nhard + nsoft;
  for (uint32_t c = 0; c < nd[k].nc; ++c) {
    int taken = 0;
    for (uint32_t j = 0; j < n && !taken; ++j) {
      if (nd[j].par != (int)k || nd[j].soft) continue;
      for (uint32_t x = 0; x < nd[j].nc && !taken; ++x)
        taken = !strcmp(nd[j].cls[x], nd[k].cls[c]);
    }
    if (taken) continue;
    /* Just the name. "decided here" on every row was noise -- it is what a
     * leaf line MEANS, and the legend says so once. */
    printf("%s%s%s\n", sub, ++item == total ? "`-- " : "|-- ", nd[k].cls[c]);
  }
  for (uint32_t j = 0; j < n; ++j) {
    if (nd[j].par != (int)k || nd[j].soft) continue;
    itree_render(nd, n, j, sub, ++item == total);
  }
  /* Soft children last: they are leaves by construction (a satellite has no
   * booster and nothing descends from it), so they need no recursion. */
  for (uint32_t j = 0; j < n; ++j) {
    if (nd[j].par != (int)k || !nd[j].soft) continue;
    const char *tag = strchr(nd[j].name, MS_SAT_SEP);
    int uw = (int)strlen(sub) + 4;
    int ww = 26 - uw; if (ww < 8) ww = 8;
    printf("%s%s%-*s %2u classes  %7s patterns  %9s CpGs  soft\n", sub,
           ++item == total ? "`~~ " : "|~~ ", ww, tag ? tag + 1 : nd[j].name,
           nd[j].nc, commafmt(nd[j].npat, b1), commafmt(nd[j].cpg, b2));
  }
}

static int inspect_tree(const char *path) {
  ms_mrmpset_t *ch = ms_mrmpset_open(path);
  const uint32_t n = ch->n_sets;
  /* A flat build is a tree of one level, so it renders the same way -- one
   * node, every class decided there. Refusing it would make the output format
   * depend on how the artifact happened to split. */
  inode_t *nd = calloc(n, sizeof(inode_t));
  mrmp_top_t **top = calloc(n, sizeof(mrmp_top_t *));
  uint32_t *setof = calloc(n ? n : 1, sizeof(uint32_t));
  if (!nd || !top || !setof) idie("out of memory", path);
  /* Hard sets first so a soft child's parent index always already exists. */
  uint32_t nn = 0, nsat = 0;
  for (uint32_t s = 0; s < n; ++s)
    if (!ms_set_is_satellite(ch->name[s])) setof[nn++] = s;
  const uint32_t n_hard = nn;
  for (uint32_t s = 0; s < n; ++s)
    if (ms_set_is_satellite(ch->name[s])) { setof[nn++] = s; ++nsat; }
  if (!n_hard) idie("a tree needs at least one node", path);
  uint64_t tot_cpg = 0;
  for (uint32_t k = 0; k < nn; ++k) {
    top[k] = ms_mrmp_top_read_at(path, ch->block_off[setof[k]], UINT32_MAX);
    nd[k].name = ch->name[setof[k]];
    nd[k].nc   = top[k]->n_samples;
    nd[k].cls  = top[k]->labels;
    nd[k].npat = top[k]->n_patterns;
    nd[k].cpg  = 0;
    for (uint32_t p = 0; p < top[k]->n_patterns; ++p) nd[k].cpg += top[k]->count[p];
    tot_cpg += nd[k].cpg;
  }
  for (uint32_t k = 0; k < nn; ++k) {
    char ob[512];
    nd[k].par = -1;
    if (ms_set_owner(nd[k].name, ob, sizeof ob)) {     /* a SOFT child */
      nd[k].soft = 1;
      for (uint32_t j = 0; j < n_hard; ++j)
        if (!strcmp(nd[j].name, ob)) { nd[k].par = (int)j; break; }
      if (nd[k].par < 0)
        idie("a satellite names a node not in the chain", nd[k].name);
    } else {                                            /* a HARD child */
      const char *dot = strrchr(nd[k].name, '.');
      if (!dot) continue;
      size_t plen = (size_t)(dot - nd[k].name);
      for (uint32_t j = 0; j < n_hard; ++j)
        if (strlen(nd[j].name) == plen && !strncmp(nd[j].name, nd[k].name, plen))
          { nd[k].par = (int)j; break; }
      if (nd[k].par < 0)
        idie("a node's parent is missing from the chain", nd[k].name);
    }
    /* Both kinds must be a SUBSET of the parent; only hard children must also
     * be disjoint from their siblings, and that is checked where routing is
     * built, not here. A soft child overlapping its siblings is correct. */
    for (uint32_t c = 0; c < nd[k].nc; ++c) {
      uint32_t p = (uint32_t)nd[k].par, seen = 0;
      for (uint32_t d = 0; d < nd[p].nc && !seen; ++d)
        seen = !strcmp(nd[p].cls[d], nd[k].cls[c]);
      if (!seen) idie("a child has a class its parent does not", nd[k].cls[c]);
    }
  }
  uint32_t root = 0, nroot = 0;
  for (uint32_t k = 0; k < nn; ++k) if (nd[k].par < 0) { root = k; ++nroot; }
  if (nroot != 1) idie("a tree needs exactly one root", path);

  char b1[32], b2[32];
  uint64_t tot_pat = 0;
  for (uint32_t k = 0; k < nn; ++k) tot_pat += nd[k].npat;
  printf("\nTREE  %s\n", path);
  printf("  %u node%s", n_hard, n_hard == 1 ? "" : "s");
  if (nsat) printf(" + %u soft", nsat);
  printf(", %u classes, %s patterns over %s CpGs\n\n",
         nd[root].nc, commafmt(tot_pat, b1), commafmt(tot_cpg, b2));
  itree_render(nd, nn, root, "", 1);

  printf("\n  A class listed under a node is DECIDED there. A HARD child (|--) takes\n"
         "  those cells and re-decides them on its own evidence. A SOFT child\n"
         "  (|~~) takes no cells: it lends its columns to this node's booster, so\n"
         "  soft siblings may OVERLAP where hard ones may not. Nothing here is\n"
         "  stored -- a hard parent is the name minus its last dotted component,\n"
         "  a soft one is the name before '@'.\n");
  putchar('\n');
  for (uint32_t k = 0; k < nn; ++k) ms_mrmp_top_free(top[k]);
  free(nd); free(top); free(setof); ms_mrmpset_free(ch);
  return 0;
}

int main_inspect(int argc, char *argv[]) {
  if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
    inspect_usage(); return 0;
  }
  /* The first bare argument is the file; --top takes a value, so skip its. */
  const char *path = NULL;
  int want_tree = 0, show_patterns = 0;
  uint32_t top_k = 20;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--tree")) want_tree = 1;
    else if (!strcmp(argv[i], "--patterns")) show_patterns = 1;
    else if (!strcmp(argv[i], "--top") && i + 1 < argc)
      top_k = (uint32_t)strtoul(argv[++i], NULL, 10);
  }
  for (int i = 1; i < argc && !path; ++i)
    if (argv[i][0] != '-' && (i == 1 || strcmp(argv[i - 1], "--top")))
      path = argv[i];
  if (!path) return inspect_usage();
  if (want_tree) return inspect_tree(path);

  /* Detect the artifact from its magic and hand off to the owning reporter. */
  unsigned char magic[8] = {0};
  int mfd = open(path, O_RDONLY);
  if (mfd < 0) idie("cannot open", path);
  ssize_t got = pread(mfd, magic, sizeof(magic), 0);
  close(mfd);
  if (got < 8) idie("file is too short to identify", path);
  /* Bundle first: a bundle now leads with the .mrmp bytes, so its first magic
   * IS MRMPIDX1 and the chain branch below would claim it. ms_bundle_is() looks
   * at the footer, which only a bundle has. */
  if (ms_bundle_is(path)) goto bundle_report;

  /* One magic for both arities: a .mrmp is a chain of MRMPIDX1 blocks, so the
   * SET COUNT picks the report rather than a second magic. */
  if (!memcmp(magic, "MRMPIDX1", 8)) {
    ms_mrmpset_t *s = ms_mrmpset_open(path);
    uint32_t n = s->n_sets;
    ms_mrmpset_free(s);
    return n > 1 ? main_mrmpset_inspect(path, show_patterns, top_k)
                  : main_mrmp_inspect(argc, argv);
  }
  if (!memcmp(magic, "MSUIDX1", 7)) { ms_msui_report(path); return 0; }
  if (!memcmp(magic, "MSURAW2", 7) || !memcmp(magic, "MSURAW3", 7)) { ms_msur_report(path); return 0; }
  if (!memcmp(magic, MSDREF_MAGIC, 7)) return inspect_msdref(path);
  if (!memcmp(magic, "MSFMAT1", 7)) { ms_msfm_report(path); return 0; }

  if (argc != 2) return inspect_usage();
  if (!ms_bundle_is(path))
    idie("not a methscope bundle, .mrmp, .msui, .msur, or .msdref", path);
bundle_report:;

  char  *kind = ms_bundle_kind(path);               /* NULL if unmarked */
  /* A TREE bundle has no single "model": it holds one booster section per node,
   * named for the node, over a chain prefix. Report the tree instead of dying
   * on the missing section. */
  if (kind && !strcmp(kind, "tree")) {
    int nsec = 0;
    ms_bundle_entry_t *secs = ms_bundle_list(path, &nsec);
    ms_mrmpset_t *ch = ms_mrmpset_open(path);
    printf("\ncontainer  MSBNDL1 (MethScope BuNDLe v1) - %d sections\n\n", nsec);
    printf("  tree     %s\n", path);
    printf("  %-14s %u node(s) over a %u-set chain\n\n", "format", nsec - 2,
           ch->n_sets);
    printf("  %3s  %-15s  %11s  %11s  %s\n", "#", "section", "offset",
           "size", "content");
    printf("  ---  ---------------  -----------  -----------  "
           "--------------------------------\n");
    uint64_t booster_bytes = 0;
    for (int i = 2; i < nsec; ++i) booster_bytes += secs[i].length;
    char ob[32], sb[32];
    printf("  %3d  %-15s  %11s  %11s  %s\n", 0, "mrmp",
           commafmt(secs[0].offset, ob), commafmt(secs[0].length, sb),
           "MRMP chain (MRMPIDX1)");
    printf("  %3d  %-15s  %11s  %11s  %s\n", 1, "kind",
           commafmt(secs[1].offset, ob), commafmt(secs[1].length, sb),
           "\\\"tree\\\" framework mark");
    printf("  %3s  %-15s  %11s  %11s  %s\n", "2-", "booster chain",
           commafmt(secs[2].offset, ob), commafmt(booster_bytes, sb),
           "xgboost boosters for tree nodes");
    printf("\n  Node details\n");
    /* Soft children (satellites) are sets of this chain but carry no booster
     * -- listing them here with a blank model and a parent derived by the
     * DOTTED rule printed "root.0" for a child of root.0.0, which is simply
     * wrong. They are summarised per parent instead. */
    uint32_t nsoft = 0;
    for (uint32_t k = 0; k < ch->n_sets; ++k)
      if (ms_set_is_satellite(ch->name[k])) ++nsoft;
    printf("\n  %-12s %8s %9s %10s  %s\n", "node", "classes", "booster",
           "patterns", "parent");
    for (uint32_t k = 0; k < ch->n_sets; ++k) {
      if (ms_set_is_satellite(ch->name[k])) continue;
      ms_bundle_entry_t e; uint64_t blen = 0;
      if (ms_bundle_find(path, ch->name[k], &e)) blen = e.length;
      mrmp_top_t *t = ms_mrmp_top_read_at(path, ch->block_off[k], UINT32_MAX);
      const char *dot = strrchr(ch->name[k], '.');
      char par[64]; snprintf(par, sizeof par, "%.*s",
                             dot ? (int)(dot - ch->name[k]) : 2,
                             dot ? ch->name[k] : "--");
      /* how much extra evidence its soft children lend it */
      uint32_t ns = 0; uint64_t scpg = 0;
      for (uint32_t j = 0; j < ch->n_sets; ++j) {
        char ob[512];
        if (!ms_set_owner(ch->name[j], ob, sizeof ob)) continue;
        if (strcmp(ob, ch->name[k])) continue;
        mrmp_top_t *st = ms_mrmp_top_read_at(path, ch->block_off[j], UINT32_MAX);
        for (uint32_t q = 0; q < st->n_patterns; ++q) scpg += st->count[q];
        ms_mrmp_top_free(st); ++ns;
      }
      char b1[32], b2[32], b3[32];
      printf("  %-12s %8u %9s %10s  %s", ch->name[k], t->n_samples,
             commafmt(blen, b1), commafmt(t->n_patterns, b2), par);
      if (ns) printf("   + %u soft, %s CpGs", ns, commafmt(scpg, b3));
      putchar('\n');
      ms_mrmp_top_free(t);
    }
    if (nsoft)
      printf("\n  %u soft child(ren) lend columns to a node's booster and carry\n"
             "  no model of their own; `inspect --tree` shows them in place.\n",
             nsoft);
    printf("\n  Score with: methscope classify %s query.cg\n\n", path);
    ms_mrmpset_free(ch); free(secs); free(kind);
    return 0;
  }
  ms_bundle_entry_t me;
  if (!ms_bundle_find(path, "model", &me)) idie("model section not found", path);
  unsigned char prefix[sizeof(ms_updec2_header_t)] = {0};
  int pfd = open(path, O_RDONLY);
  if (pfd < 0 || pread(pfd, prefix, sizeof(prefix), (off_t)me.offset) < 8)
    idie("cannot read model header", path);
  int is_updec2 = me.length >= sizeof(ms_updec2_header_t) &&
                  memcmp(prefix, MS_UPDEC2_MAGIC, 8) == 0;
  ms_updec2_header_t u2 = {0};
  if (is_updec2) memcpy(&u2, prefix, sizeof(u2));
  uint32_t u2_pure = 0, u2_mixed = 0, u2_pna = 0, u2_factor = 0, u2_direct = 0;
  uint32_t u2_min_rank = UINT32_MAX, u2_max_rank = 0;
  if (is_updec2) {
    if (!u2.n_units || u2.unit_offset > me.length ||
        (uint64_t)u2.n_units * sizeof(ms_updec2_unit_t) >
          me.length - u2.unit_offset)
      idie("invalid UPDEC2 unit directory", path);
    ms_updec2_unit_t *uu = malloc((size_t)u2.n_units * sizeof(*uu));
    if (!uu || pread(pfd, uu, (size_t)u2.n_units * sizeof(*uu),
                     (off_t)(me.offset + u2.unit_offset)) !=
               (ssize_t)((size_t)u2.n_units * sizeof(*uu)))
      idie("cannot read UPDEC2 unit directory", path);
    for (uint32_t j = 0; j < u2.n_units; ++j) {
      /* unit flag bits (see MSUI_UNIT_* in upunit_index.c): 2=PNA, 1=pure */
      if (uu[j].flags & 2) ++u2_pna;
      else if (uu[j].flags & 1) ++u2_pure;
      else ++u2_mixed;
      if (uu[j].mode == MS_UPDEC2_DIRECT) ++u2_direct;
      else {
        ++u2_factor;
        if (uu[j].bottleneck_dim < u2_min_rank) u2_min_rank = uu[j].bottleneck_dim;
        if (uu[j].bottleneck_dim > u2_max_rank) u2_max_rank = uu[j].bottleneck_dim;
      }
    }
    free(uu);
  }
  close(pfd);
  size_t mlen = 0; void *mbuf = NULL;
  if (!is_updec2) mbuf = ms_bundle_section(path, "model", &mlen);
  int is_updec  = (mlen >= 6  && memcmp(mbuf, "UPDEC1", 6) == 0);
  int is_linear = (mlen >= 16 && memcmp(mbuf, "methscope-linear", 16) == 0);
  int is_vio    = (mlen >= 19 && memcmp(mbuf, "methscope-violation", 19) == 0);

  /* bundled-MRMP state count (used in both the layout and the model summary) */
  size_t rlen = 0;
  void *rbuf = ms_bundle_section_opt(path, "mrmp", &rlen);
  long ns = -1;
  long nrec = 0;
  int  ref_is_mrmp = (rbuf && rlen >= 8 && !memcmp(rbuf, "MRMPIDX1", 8));
  if (rbuf && !ref_is_mrmp) {
    char *t = buf_to_tmp(rbuf, rlen);
    ns = mrmp_state_count(t, &nrec);
    unlink(t); free(t);
  }
  free(rbuf);

  /* framework-specific dims parsed once (reused by the layout + the model block) */
  int32_t ud[3] = {0, 0, 0};
  if (is_updec && mlen >= 20) memcpy(ud, (char *)mbuf + 8, 12);  /* magic(8) then n_in,n_hidden,n_out */
  /* violation dims straight off the spec: labels from the tab count on the
   * 'labels' line, features from the number of 'pattern' rows. */
  int vio_lab = 0, vio_pat = 0;
  if (is_vio) {
    const char *s = (const char *)mbuf, *end = s + mlen;
    for (const char *p = s; p < end; ) {
      const char *nl = memchr(p, '\n', (size_t)(end - p));
      size_t n = nl ? (size_t)(nl - p) : (size_t)(end - p);
      if (n > 7 && memcmp(p, "labels\t", 7) == 0)
        for (size_t k = 0; k < n; ++k) { if (p[k] == '\t') vio_lab++; }
      else if (n > 8 && memcmp(p, "pattern\t", 8) == 0) vio_pat++;
      if (!nl) break;
      p = nl + 1;
    }
  }

  /* short "content" for the section table = the model section's inner type */
  char mdesc[160];
  if      (is_updec2) snprintf(mdesc, sizeof mdesc, "UPDEC2 whole-genome unit decoder (%u units)", u2.n_units);
  else if (is_updec)  snprintf(mdesc, sizeof mdesc, "UPDEC1 MLP decoder (%d->%d->%d)", ud[0], ud[1], ud[2]);
  else if (is_linear) snprintf(mdesc, sizeof mdesc, "methscope-linear text spec");
  else if (is_vio)    snprintf(mdesc, sizeof mdesc,
                        "methscope-violation text spec (%d classes x %d patterns)",
                        vio_lab, vio_pat);
  else                snprintf(mdesc, sizeof mdesc, "xgboost booster (UBJ binary)");

  /* role = how the whole bundle is used (its framework mark applies bundle-wide) */
  char role[160];
  if      (is_updec2) snprintf(role, sizeof role, "whole-genome upscale decoder - run via `upscale`");
  else if (is_updec)  snprintf(role, sizeof role, "upscale decoder - run via `upscale`");
  else if (is_linear) snprintf(role, sizeof role, "%s linear classifier - run via `classify`", kind ? kind : "linear");
  else if (is_vio)    snprintf(role, sizeof role,
                        "violation rule (unfitted) - run via `classify`");
  else                snprintf(role, sizeof role, "xgboost classifier - run via `classify`");

  /* ---- container ---- */
  int nsec = 0;
  ms_bundle_entry_t *secs = ms_bundle_list(path, &nsec);
  unsigned long long maxend = 0;                   /* end of the last section blob */
  for (int i = 0; i < nsec; ++i)
    if (secs[i].offset + secs[i].length > maxend) maxend = secs[i].offset + secs[i].length;
  unsigned long long total = maxend + 8;           /* + 8-byte MSBNDL1 offset footer */
  char cb[32];
  printf("container  MSBNDL1 (MethScope BuNDLe v1) - %d sections, %s bytes\n\n",
         nsec, commafmt(total, cb));
  printf("  bundle     %s\n\n", path);

  /* ---- section list ---- */
  printf("  %3s  %-15s  %11s  %11s  %s\n", "#", "section", "offset",
         "size", "content");
  printf("  ---  ---------------  -----------  -----------  "
         "--------------------------------\n");
  for (int i = 0; i < nsec; ++i) {
    char cbuf[176]; const char *c = "";
    if      (strcmp(secs[i].name, "model")     == 0) c = mdesc;
    else if (strcmp(secs[i].name, "mrmp")   == 0) {
      if (ref_is_mrmp) { snprintf(cbuf, sizeof cbuf, "MRMPIDX1 artifact"); c = cbuf; }
      else if (ns >= 0 && nrec > 1)
        { snprintf(cbuf, sizeof cbuf, "YAME .cm mask, %ld records, %ld states in the first",
                   nrec, ns); c = cbuf; }
      else if (ns >= 0)
        { snprintf(cbuf, sizeof cbuf, "YAME .cm mask, %ld states", ns); c = cbuf; }
      else c = "MRMP definition (YAME .cm)";
    }
    else if (strcmp(secs[i].name, "outcpg") == 0) c = "genome-wide output-CpG mask";
    else if (strcmp(secs[i].name, "kind")      == 0) {
      snprintf(cbuf, sizeof cbuf, "\"%s\" framework mark", kind ? kind : ""); c = cbuf;
    }
    char ob[32], sb[32];
    printf("  %3d  %-15s  %11s  %11s  %s\n", i, secs[i].name,
           commafmt(secs[i].offset, ob), commafmt(secs[i].length, sb), c);
  }

  /* ---- per-section detail (in file order) ---- */
  for (int i = 0; i < nsec; ++i) {
    const char *nm = secs[i].name;
    const char *hdr =
      strcmp(nm, "model")     == 0 ? role :
      strcmp(nm, "mrmp")   == 0 ? (ref_is_mrmp ? "MRMP feature definition (MRMPIDX1)"
                                              : "MRMP feature definition (YAME .cm)") :
      strcmp(nm, "outcpg") == 0 ? "genome-wide output-CpG mask" :
      strcmp(nm, "kind")      == 0 ? "bundle framework mark" : "";
    printf("\n  [%d] %-9s  %s\n", i, nm, hdr);

    /* detail fields: "      <name padded to 10> <value>" so every value (and the
       role in the header above) lines up in the same column across all sections */
    if (strcmp(nm, "model") == 0) {
      if (is_updec2) {
        char nc[32], fb[32];
        printf("      %-14s UPDEC2/v%u\n", "format", u2.version);
        printf("      %-14s %u\n", "patterns", u2.patterns);
        printf("      %-14s %u\n", "input_dim", u2.input_dim);
        printf("      %-14s %s\n", "features",
               u2.version >= 3 && (u2.flags & MS_UPDEC2_FLAG_BETA_ONLY)
                 ? "standardized beta only"
                 : u2.version >= 3 && (u2.flags & MS_UPDEC2_FLAG_COUNT)
                   ? "standardized beta + log1p count"
                   : "standardized beta + missing indicator");
        printf("      %-14s %s\n", "missing",
               u2.version >= 3 && (u2.flags & MS_UPDEC2_FLAG_BETA_ONLY)
                 ? "beta mean-imputed"
                 : u2.version >= 3 && (u2.flags & MS_UPDEC2_FLAG_COUNT)
                   ? "count 0; beta mean-imputed" : "NaN -> beta 0, indicator 1");
        printf("      %-14s %s\n", "shared trunk",
               u2.version >= 3 && (u2.flags & MS_UPDEC2_FLAG_TRUNK)
                 ? "two-layer residual LeakyReLU" : "none");
        if (u2.version >= 3 && (u2.flags & MS_UPDEC2_FLAG_TRUNK))
          printf("      %-14s %u\n", "trunk_dim", (uint32_t)u2.trunk_dim);
        printf("      %-14s %s\n", "activation",
               u2.activation == MS_UPDEC2_LEAKY_RELU ? "leaky_relu_0.01" : "linear");
        printf("      %-14s %u\n", "units", u2.n_units);
        printf("      %-14s %u / %u / %u\n", "pure/mixed/PNA",
               u2_pure, u2_mixed, u2_pna);
        printf("      %-14s %u / %u\n", "factor/direct",
               u2_factor, u2_direct);
        if (u2_factor)
          printf("      %-14s %u..%u\n", "bottleneck_dim",
                 u2_min_rank, u2_max_rank);
        printf("      %-14s %u\n", "memberships", u2.n_memberships);
        printf("      %-14s %s\n", "CpGs", commafmt(u2.n_cpg, nc));
        printf("      %-14s %u\n", "target/unit", u2.target_unit_cpgs);
        printf("      %-14s %s\n", "model bytes", commafmt(u2.file_bytes, fb));
        printf("      %-14s %016llx\n", "index checksum",
               (unsigned long long)u2.index_checksum);
        printf("      %-14s %016llx\n", "parameter sum",
               (unsigned long long)u2.parameter_checksum);
      } else if (is_updec) {
        printf("      %-10s %d\n", "n_in",     ud[0]);
        printf("      %-10s %d\n", "n_hidden", ud[1]);
        printf("      %-10s %d\n", "n_out",    ud[2]);
      } else if (is_linear) {
        linmodel_t *lm = ms_linmodel_parse(mbuf, mlen);
        printf("      %-10s %s\n",     "method",   lm->method);
        printf("      %-10s %s, %s\n", "labels",   lm->label0, lm->label1);
        printf("      %-10s %.6g\n",   "bias",     lm->bias);
        printf("      %-10s %.6g\n",   "scale",    lm->scale);
        printf("      %-10s %d\n",     "features", lm->n_feat);
        for (int j = 0; j < lm->n_feat; ++j)
          printf("        %-12s weight=%-12.6g mean=%.6g\n", lm->names[j], lm->w[j], lm->mean[j]);
        ms_linmodel_free(lm);
      } else {
        BoosterHandle b;
        if (XGBoosterCreate(NULL, 0, &b) == 0) {
          if (XGBoosterLoadModelFromBuffer(b, mbuf, mlen) == 0) {
            bst_ulong nf = 0; XGBoosterGetNumFeature(b, &nf);
            printf("      %-10s %lu\n", "features", (unsigned long)nf);
            int K = 0; char **labels = ms_booster_get_labels(b, &K);
            if (labels) {
              printf("      %-10s ", "labels"); print_labels(labels, K);
              for (int c = 0; c < K; ++c) free(labels[c]);
              free(labels);
            } else printf("      %-10s (none embedded)\n", "labels");
          }
          XGBoosterFree(b);   /* free even when the model failed to load */
        }
      }
    } else if (strcmp(nm, "mrmp") == 0) {
      if (ns >= 0) printf("      %-10s %ld\n", "states", ns);
      if (nrec > 1) printf("      %-10s %ld  (one per set)\n", "records", nrec);
      if (ref_is_mrmp) {
        /* The chain sits at offset 0 of the bundle and the walker stops at the
         * MSBNDL1 trailer, so it reads straight off the bundle path -- the same
         * prefix trick a bundled .cm relied on. */
        ms_mrmpset_t *ch = ms_mrmpset_open(path);
        uint64_t pats = 0;
        for (uint32_t k = 0; k < ch->n_sets; ++k) {
          mrmp_top_t *t = ms_mrmp_top_read_at(path, ch->block_off[k], UINT32_MAX);
          pats += t->n_patterns;
          ms_mrmp_top_free(t);
        }
        printf("      %-10s %u\n", "sets", ch->n_sets);
        printf("      %-10s %llu\n", "patterns", (unsigned long long)pats);
        ms_mrmpset_free(ch);
      }
    } else if (strcmp(nm, "outcpg") == 0) {
      printf("      imputed-CpG locations; makes `upscale` emit a whole-genome .cg\n");
    } else if (strcmp(nm, "kind") == 0) {
      printf("      %-10s %s\n", "value", kind ? kind : "");
    }
  }

  if (!kind && !is_updec && !is_updec2)
    printf("\n  note  no `kind` section -> `predict` will reject this bundle; "
           "stamp one with `bundle -k`\n");

  free(secs);
  free(mbuf); free(kind);
  return 0;
}
