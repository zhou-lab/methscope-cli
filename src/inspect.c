// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * `inspect` — describe a model bundle (.ubjx / .updecx / .refx) without running
 * inference: the framework mark (kind), the on-disk section layout (each section's
 * offset + size + a short description of its contents), and a breakdown of the
 * `model` section by framework:
 *   xgboost            -> num_feature + embedded labels
 *   threshold/logistic -> method, labels, bias, scale, per-feature weight/mean
 *   refx               -> cell types x patterns + the cell-type labels
 *   upscale (UPDEC1)   -> n_in / n_hidden / n_out from the decoder header
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include "methscope.h"
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

static int inspect_usage(void) {
  fprintf(stderr,
    "\n"
    "Usage:\n"
    "  methscope inspect <model.ubjx|.updecx|.refx>\n"
    "\n"
    "Purpose:\n"
    "  Describe a bundle without running it: its framework mark (kind), the on-disk\n"
    "  section layout (offset/size/contents), and a breakdown of the model section\n"
    "  (labels, feature/pattern counts, linear-model weights).\n"
    "\n"
    "Options:\n"
    "  -h   Show this help message.\n"
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

/* number of distinct states in a fmt2 .cm, or -1 on failure */
static long mrmp_state_count(const char *cm_path) {
  cfile_t cf = open_cfile((char *)cm_path);
  cdata_t c  = read_cdata1(&cf);
  bgzf_close(cf.fh);
  if (c.fmt != '2') { free_cdata(&c); return -1; }
  cdata_t d = decompress(c);
  free_cdata(&c);
  long n = (long)fmt2_get_keys_n(&d);
  free_cdata(&d);
  return n;
}

/* format an unsigned integer with thousands separators into buf (>= 32 bytes) */
static const char *commafmt(unsigned long long v, char *buf) {
  char tmp[24]; int n = snprintf(tmp, sizeof tmp, "%llu", v);
  int commas = (n - 1) / 3, len = n + commas;
  buf[len] = '\0';
  int bi = len - 1, ti = n - 1, cnt = 0;
  while (ti >= 0) {
    buf[bi--] = tmp[ti--];
    if (++cnt % 3 == 0 && ti >= 0) buf[bi--] = ',';
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

int main_inspect(int argc, char *argv[]) {
  if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
    inspect_usage(); return 0;
  }
  if (argc != 2) return inspect_usage();
  const char *path = argv[1];
  if (!ms_bundle_is(path))
    idie("not a methscope bundle (.ubjx/.updecx/.refx)", path);

  char  *kind = ms_bundle_kind(path);               /* NULL if unmarked */
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
  int is_refx   = (kind && strcmp(kind, "refx") == 0) ||
                  (mlen >= 5 && memcmp(mbuf, "cell\t", 5) == 0);

  /* bundled-MRMP state count (used in both the layout and the model summary) */
  size_t rlen = 0;
  void *rbuf = ms_bundle_section_opt(path, "mrmp", &rlen);
  long ns = -1;
  if (rbuf) { char *t = buf_to_tmp(rbuf, rlen); ns = mrmp_state_count(t); unlink(t); free(t); }
  free(rbuf);

  /* framework-specific dims parsed once (reused by the layout + the model block) */
  int refx_cells = 0, refx_pat = 0;
  if (is_refx) {
    for (size_t k = 0; k < mlen && ((char *)mbuf)[k] != '\n'; ++k)
      if (((char *)mbuf)[k] == '\t') refx_pat++;                 /* header tabs = #patterns */
    int lines = 0; for (size_t k = 0; k < mlen; ++k) if (((char *)mbuf)[k] == '\n') lines++;
    refx_cells = lines > 0 ? lines - 1 : 0;                      /* rows minus header */
  }
  int32_t ud[3] = {0, 0, 0};
  if (is_updec && mlen >= 20) memcpy(ud, (char *)mbuf + 8, 12);  /* magic(8) then n_in,n_hidden,n_out */

  /* short "content" for the section table = the model section's inner type */
  char mdesc[160];
  if      (is_updec2) snprintf(mdesc, sizeof mdesc, "UPDEC2 whole-genome unit decoder (%u units)", u2.n_units);
  else if (is_updec)  snprintf(mdesc, sizeof mdesc, "UPDEC1 MLP decoder (%d->%d->%d)", ud[0], ud[1], ud[2]);
  else if (is_refx)   snprintf(mdesc, sizeof mdesc, "refx signature TSV (%d cell types x %d patterns)", refx_cells, refx_pat);
  else if (is_linear) snprintf(mdesc, sizeof mdesc, "methscope-linear text spec");
  else                snprintf(mdesc, sizeof mdesc, "xgboost booster (UBJ binary)");

  /* role = how the whole bundle is used (its framework mark applies bundle-wide) */
  char role[160];
  if      (is_updec2) snprintf(role, sizeof role, "whole-genome upscale decoder - run via `upscale`");
  else if (is_updec)  snprintf(role, sizeof role, "upscale decoder - run via `upscale`");
  else if (is_refx)   snprintf(role, sizeof role, "deconvolution reference - run via `deconv`");
  else if (is_linear) snprintf(role, sizeof role, "%s linear classifier - run via `predict`", kind ? kind : "linear");
  else                snprintf(role, sizeof role, "xgboost classifier - run via `predict`");

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

  /* ---- section list ---- */
  printf("  %3s  %-9s  %11s  %11s  %s\n", "#", "section", "offset", "size", "content");
  printf("  ---  ---------  -----------  -----------  "
         "------------------------------------------------\n");
  for (int i = 0; i < nsec; ++i) {
    char cbuf[176]; const char *c = "";
    if      (strcmp(secs[i].name, "model")     == 0) c = mdesc;
    else if (strcmp(secs[i].name, "mrmp")   == 0) {
      if (ns >= 0) { snprintf(cbuf, sizeof cbuf, "fmt2 MRMP definition, %ld states", ns); c = cbuf; }
      else c = "MRMP definition (YAME .cm)";
    }
    else if (strcmp(secs[i].name, "outcpg") == 0) c = "genome-wide output-CpG mask";
    else if (strcmp(secs[i].name, "kind")      == 0) {
      snprintf(cbuf, sizeof cbuf, "\"%s\" framework mark", kind ? kind : ""); c = cbuf;
    }
    char ob[32], sb[32];
    printf("  %3d  %-9s  %11s  %11s  %s\n", i, secs[i].name,
           commafmt(secs[i].offset, ob), commafmt(secs[i].length, sb), c);
  }

  /* ---- per-section detail (in file order) ---- */
  for (int i = 0; i < nsec; ++i) {
    const char *nm = secs[i].name;
    const char *hdr =
      strcmp(nm, "model")     == 0 ? role :
      strcmp(nm, "mrmp")   == 0 ? "MRMP feature definition (fmt2 YAME .cm)" :
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
          printf("      %-14s %u\n", "trunk_dim", (uint32_t)u2.reserved0);
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
      } else if (is_refx) {
        printf("      %-10s %d\n", "cell types", refx_cells);
        printf("      %-10s %d\n", "patterns",   refx_pat);
        char **cts = malloc((refx_cells ? refx_cells : 1) * sizeof *cts);
        int nct = 0; size_t p = 0; int line = 0;
        while (p < mlen && nct < refx_cells) {         /* first field of each data row */
          size_t e = p; while (e < mlen && ((char *)mbuf)[e] != '\n') e++;
          if (line > 0 && e > p) {
            size_t f = p; while (f < e && ((char *)mbuf)[f] != '\t') f++;
            char *s = malloc(f - p + 1); memcpy(s, (char *)mbuf + p, f - p); s[f - p] = '\0';
            cts[nct++] = s;
          }
          p = e + 1; line++;
        }
        printf("      %-10s ", "labels"); print_labels(cts, nct);
        for (int j = 0; j < nct; ++j) free(cts[j]);
        free(cts);
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
        if (XGBoosterCreate(NULL, 0, &b) == 0 && XGBoosterLoadModelFromBuffer(b, mbuf, mlen) == 0) {
          bst_ulong nf = 0; XGBoosterGetNumFeature(b, &nf);
          printf("      %-10s %lu\n", "features", (unsigned long)nf);
          int K = 0; char **labels = ms_booster_get_labels(b, &K);
          if (labels) {
            printf("      %-10s ", "labels"); print_labels(labels, K);
            for (int c = 0; c < K; ++c) free(labels[c]);
            free(labels);
          } else printf("      %-10s (none embedded)\n", "labels");
          XGBoosterFree(b);
        }
      }
    } else if (strcmp(nm, "mrmp") == 0) {
      if (ns >= 0) printf("      %-10s %ld\n", "states", ns);
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
