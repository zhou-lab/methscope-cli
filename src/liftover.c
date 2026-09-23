// SPDX-License-Identifier: LicenseRef-CHOP-Academic-BSD-2-Clause
// Use of this software is available to academic and non-profit institutions
// for research purposes under the 2-Clause BSD License; for use or transfers
// to commercial entities, inquire with Dr. Wanding Zhou at zhouw3@chop.edu.
// See the LICENSE file at the root of the repository for the full terms.
/*
 * mliftover: re-index a MODEL onto an array platform's row space.
 *
 * Every methscope artifact is positional over one CpG row space -- the
 * reference genome's cpg_nocontig.cr -- and refuses a query built on another.
 * An array is another row space: one row per probe, in the platform's
 * ordering. This command carries a model ACROSS, so array data can be scored
 * in its own space, untouched, by a model that was built genome-wide.
 *
 * The alternative is to lift the DATA (sesame mliftover --to hg38, which
 * writes a genome-space .cg from array betas) and run the original model.
 * Both yield the same feature values -- a pattern's feature is the mean over
 * its observed CpGs, and the probes ARE the observed CpGs either way -- so the
 * choice is about cost: lifting the model once keeps array files small and
 * scoring fast; lifting the data needs no per-platform artifact.
 *
 * What moves and what does not. Only the row index is rewritten: a target row
 * takes what its CpG had (membership rank, kept-row M/U), rows under no CpG
 * take the format's "absent" (PNA / not kept). Pattern keys, binstrings,
 * midpoints, boosters, linear weights, the deconvolution trailer -- the MODEL
 * -- are copied byte for byte. Per-pattern CpG counts are recomputed, since
 * they drive `--top` ranking and the violation rule's weights, and the
 * artifact's reference name becomes the platform so `inspect` says which row
 * space it lives in and the row-count guards fire against the wrong query.
 *
 * What it will not lift: an upscaler. Its OUTPUT is the genome, so "on an
 * array" is a different model (predict only the probes), not a lift. The
 * route for array input there is the data direction, and the refusal says
 * so. Array-to-array imputation is its own future model.
 *
 * The map is the same join sesame uses: <platform>.<genome>.coord.tsv.gz gives
 * each probe's CpG_chrm / CpG_beg (0-based), and the genome's cpg_nocontig.cr
 * row is found at beg + 1 through YAME's row finder. A probe with no
 * coordinate, on a chromosome the track does not carry (alt contigs), or
 * whose position is not a CpG row, is unmapped. Two probes on one CpG both
 * take it (the platform's row space has two rows there; a data lift the other
 * way averages them into one, which is why the two routes agree to the
 * fourth decimal and not the sixth).
 *
 * Only cg probes are joined, decided by Probe_ID from the platform's
 * ordering, never by where the probe maps (Wanding, 2026-09-23). An rs or nv
 * probe whose sequence maps onto a CpG site has a TRUE coordinate -- the
 * mapper found it -- but its value is an allele or variant fraction, not
 * methylation, and only its identity says so. MSA carries 329 rs probes on
 * CpG rows, all of which would otherwise take a pattern state and feed a
 * genotype into that pattern's mean. Counted on their own line, distinct
 * from cg probes with no CpG row. The rule sits in the map so it holds in
 * both directions, exactly as sesame applies it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <unistd.h>
#include <zlib.h>
#include "methscope.h"
#include "mrmp.h"
#include "bundle.h"
#include "cfile.h"   /* open_cfile, read_cdata1, cdata_write */
#include "cdata.h"   /* cdata_t, fmt2, fmt7, row_finder */

static const char *g_cmd = "mliftover";
static void ldie(const char *msg, const char *arg) {
  fprintf(stderr, "[methscope] %s: %s%s%s\n", g_cmd, msg,
          arg ? ": " : "", arg ? arg : "");
  exit(1);
}
static void *lalloc(size_t n, size_t sz, const char *what) {
  void *p = calloc(n ? n : 1, sz ? sz : 1);
  if (!p) ldie("out of memory", what);
  return p;
}

/* ---------------------------------------------------------------- map --- */

/* cpg_of_row[t] for every target row t (one per coord-table row), or -1.
 * Mirrors sesame_rowmap_coords() exactly -- same table, same +1, same
 * chromosome-membership check before the search, which exits the process on
 * a chromosome the track lacks. */
/* is_cg[i] for every ordering row: the Probe_ID starts with "cg". The
 * ordering is positional to the coord table, which lo_load_coords asserts by
 * row count, so this is one parallel column and no join. */
static uint8_t *load_probe_types(const char *ordering_gz, uint64_t *n_out) {
  gzFile f = gzopen(ordering_gz, "rb");
  if (!f) ldie("cannot open ordering", ordering_gz);
  size_t cap = 1 << 16; char *line = lalloc(cap, 1, "line");
  uint64_t n = 0, ncap = 1 << 16;
  uint8_t *is_cg = lalloc(ncap, 1, "probe types");
  if (!gzgets(f, line, (int)cap)) ldie("empty ordering", ordering_gz);
  while (gzgets(f, line, (int)cap)) {
    if (n == ncap) { ncap *= 2; is_cg = realloc(is_cg, ncap); if (!is_cg) ldie("out of memory", "probe types"); }
    is_cg[n++] = (line[0] == 'c' && line[1] == 'g');
  }
  gzclose(f); free(line);
  *n_out = n;
  return is_cg;
}

static int64_t *map_from_coords(const char *coord_gz, const char *ordering_gz,
                                const char *cr_path,
                                uint64_t *n_rows_out, uint64_t *n_mapped_out,
                                uint64_t *n_noncg_out, uint64_t *n_cpg_out) {
  gzFile f = gzopen(coord_gz, "rb");
  if (!f) ldie("cannot open coordinate table", coord_gz);
  size_t cap = 1 << 16; char *line = lalloc(cap, 1, "line");
  uint64_t n = 0, ncap = 1 << 16;
  char **chrm = lalloc(ncap, sizeof(char *), "chromosomes");
  long *beg = lalloc(ncap, sizeof(long), "positions");
  if (!gzgets(f, line, (int)cap)) ldie("empty coordinate table", coord_gz);
  while (gzgets(f, line, (int)cap)) {
    if (n == ncap) {
      ncap *= 2;
      chrm = realloc(chrm, ncap * sizeof(char *)); beg = realloc(beg, ncap * sizeof(long));
      if (!chrm || !beg) ldie("out of memory", "coordinate table");
    }
    line[strcspn(line, "\r\n")] = '\0';
    char *tab = strchr(line, '\t'); if (tab) *tab = '\0';
    if (!line[0] || !strcmp(line, "*") || !strcmp(line, "NA")) {
      chrm[n] = NULL; beg[n] = -1;
    } else {
      chrm[n] = strdup(line);
      char *tab2 = tab ? strchr(tab + 1, '\t') : NULL;
      if (tab2) *tab2 = '\0';
      beg[n] = tab ? strtol(tab + 1, NULL, 10) : -1;
    }
    ++n;
  }
  gzclose(f); free(line);
  if (!n) ldie("coordinate table has no rows", coord_gz);
  uint64_t n_ord = 0;
  uint8_t *is_cg = load_probe_types(ordering_gz, &n_ord);
  if (n_ord != n) {
    char m[256];
    snprintf(m, sizeof m, "ordering has %" PRIu64 " probes, coordinate table %" PRIu64
             " -- not the same platform lineage", n_ord, n);
    ldie(m, ordering_gz);
  }

  cfile_t cf = open_cfile((char *)cr_path);
  if (!cf.fh) ldie("cannot open coordinate track", cr_path);
  cdata_t cr = read_cdata1(&cf);
  bgzf_close(cf.fh);
  if (!cr.n || cr.fmt != '7') ldie("not a format-7 row-coordinate track", cr_path);
  /* cr.n on a format-7 record is the payload length, not the row count */
  const uint64_t n_cpg = fmt7_data_length(&cr);
  row_finder_t fdr = init_finder(&cr);
  int64_t *map = lalloc(n, sizeof(int64_t), "row map");
  uint64_t mapped = 0, noncg = 0;
  for (uint64_t i = 0; i < n; ++i) {
    map[i] = -1;
    if (!is_cg[i]) { ++noncg; free(chrm[i]); continue; }
    if (chrm[i] && beg[i] >= 0 &&
        kh_get(str2int, fdr.h, chrm[i]) != kh_end(fdr.h)) {
      uint64_t r = row_finder_search(chrm[i], (uint64_t)beg[i] + 1, &fdr, &cr);
      if (r) { map[i] = (int64_t)r - 1; ++mapped; }
    }
    free(chrm[i]);
  }
  free_row_finder(&fdr); free_cdata(&cr); free(chrm); free(beg); free(is_cg);
  *n_rows_out = n; *n_mapped_out = mapped; *n_noncg_out = noncg; *n_cpg_out = n_cpg;
  return map;
}

/* ------------------------------------------------------------- msdref --- */

static uint32_t get_u32(FILE *f, const char *p) { uint32_t v; if (fread(&v, 4, 1, f) != 1) ldie("truncated", p); return v; }
static uint64_t get_u64(FILE *f, const char *p) { uint64_t v; if (fread(&v, 8, 1, f) != 1) ldie("truncated", p); return v; }
static void put_u32(FILE *f, uint32_t v, const char *p) { if (fwrite(&v, 4, 1, f) != 1) ldie("write error", p); }
static void put_u64(FILE *f, uint64_t v, const char *p) { if (fwrite(&v, 8, 1, f) != 1) ldie("write error", p); }

/* The layout is deconv.c's MSDREF1: a 64-byte header, NUL-joined class names,
 * n_kept ascending row indices, class-major M/U words, then (version 3) the
 * confusion trailer. Rows are re-indexed and gathered; every other byte,
 * trailer included, is copied. */
static void lift_msdref(const char *in, const char *out, const int64_t *map,
                        uint64_t n_rows, uint64_t n_cpg) {
  FILE *f = fopen(in, "rb");
  if (!f) ldie("cannot open reference", in);
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "MSDREF1", 8)) ldie("not a .msdref", in);
  uint32_t ver = get_u32(f, in), n_class = get_u32(f, in);
  uint64_t n_row = get_u64(f, in), n_keep = get_u64(f, in);
  double qlo, qhi, bthr;
  if (fread(&qlo, 8, 1, f) != 1 || fread(&qhi, 8, 1, f) != 1 || fread(&bthr, 8, 1, f) != 1)
    ldie("truncated header", in);
  uint32_t mincov = get_u32(f, in), rsv = get_u32(f, in); (void)rsv;
  if (ver < 1 || ver > 3) ldie("unsupported .msdref version", in);
  if (n_row != n_cpg) {
    char m[256];
    snprintf(m, sizeof m, "reference spans %" PRIu64 " rows but the coordinate track "
             "has %" PRIu64 " -- built on a different genome", n_row, n_cpg);
    ldie(m, in);
  }
  /* names: keep the bytes as they are */
  size_t nb_cap = 4096, nb = 0; char *names = lalloc(nb_cap, 1, "names");
  for (uint32_t k = 0; k < n_class; ++k) {
    int c;
    while ((c = fgetc(f)) > 0) { if (nb + 2 > nb_cap) { nb_cap *= 2; names = realloc(names, nb_cap); if (!names) ldie("out of memory", "names"); } names[nb++] = (char)c; }
    if (c < 0) ldie("truncated class names", in);
    if (nb + 1 > nb_cap) { nb_cap *= 2; names = realloc(names, nb_cap); if (!names) ldie("out of memory", "names"); }
    names[nb++] = '\0';
  }
  uint32_t *rows = lalloc(n_keep, sizeof(uint32_t), "row index");
  if (fread(rows, sizeof(uint32_t), n_keep, f) != n_keep) ldie("truncated row index", in);
  uint16_t *mu = lalloc((size_t)n_class * n_keep, sizeof(uint16_t), "M/U");
  if (fread(mu, sizeof(uint16_t), (size_t)n_class * n_keep, f) != (size_t)n_class * n_keep)
    ldie("truncated M/U block", in);
  /* the trailer, if any: everything to EOF */
  long tail_at = ftell(f);
  if (fseek(f, 0, SEEK_END)) ldie("cannot seek", in);
  long end = ftell(f);
  size_t tail_n = (size_t)(end - tail_at);
  char *tail = lalloc(tail_n, 1, "trailer");
  if (fseek(f, tail_at, SEEK_SET) || (tail_n && fread(tail, 1, tail_n, f) != tail_n))
    ldie("truncated trailer", in);
  fclose(f);

  /* how many target rows land on a kept source row */
  uint64_t n_out = 0;
  for (uint64_t t = 0; t < n_rows; ++t) {
    if (map[t] < 0) continue;
    uint32_t want = (uint32_t)map[t];
    uint64_t lo = 0, hi = n_keep;
    while (lo < hi) { uint64_t mid = (lo + hi) / 2; if (rows[mid] < want) lo = mid + 1; else hi = mid; }
    if (lo < n_keep && rows[lo] == want) ++n_out;
  }
  /* then the target rows and the kept slots they gather from, in target order */
  uint32_t *trow = lalloc(n_out ? n_out : 1, sizeof(uint32_t), "target rows");
  uint32_t *tslot = lalloc(n_out ? n_out : 1, sizeof(uint32_t), "target slots");
  uint64_t j = 0;
  for (uint64_t t = 0; t < n_rows; ++t) {
    if (map[t] < 0) continue;
    uint32_t want = (uint32_t)map[t];
    uint64_t lo = 0, hi = n_keep;
    while (lo < hi) { uint64_t mid = (lo + hi) / 2; if (rows[mid] < want) lo = mid + 1; else hi = mid; }
    if (lo < n_keep && rows[lo] == want) { trow[j] = (uint32_t)t; tslot[j] = (uint32_t)lo; ++j; }
  }
  if (n_rows > 0xFFFFFFFFull) ldie("target row space exceeds uint32", out);

  FILE *o = fopen(out, "wb");
  if (!o) ldie("cannot open output", out);
  if (fwrite("MSDREF1\0", 1, 8, o) != 8) ldie("write error", out);
  put_u32(o, ver, out); put_u32(o, n_class, out);
  put_u64(o, n_rows, out); put_u64(o, n_out, out);
  if (fwrite(&qlo, 8, 1, o) != 1 || fwrite(&qhi, 8, 1, o) != 1 || fwrite(&bthr, 8, 1, o) != 1)
    ldie("write error", out);
  put_u32(o, mincov, out); put_u32(o, 0, out);
  if (fwrite(names, 1, nb, o) != nb) ldie("write error", out);
  if (fwrite(trow, sizeof(uint32_t), n_out, o) != n_out) ldie("write error", out);
  uint16_t *buf = lalloc(n_out ? n_out : 1, sizeof(uint16_t), "row buffer");
  for (uint32_t k = 0; k < n_class; ++k) {
    const uint16_t *src = mu + (size_t)k * n_keep;
    for (uint64_t i = 0; i < n_out; ++i) buf[i] = src[tslot[i]];
    if (fwrite(buf, sizeof(uint16_t), n_out, o) != n_out) ldie("write error", out);
  }
  if (tail_n && fwrite(tail, 1, tail_n, o) != tail_n) ldie("write error", out);
  if (fclose(o)) ldie("write error", out);
  fprintf(stderr, "[methscope] mliftover: %u classes: %" PRIu64 " of %" PRIu64
          " kept rows land on %" PRIu64 " of %" PRIu64 " target rows%s\n",
          n_class, n_out, n_keep, n_out, n_rows,
          tail_n ? " (confusion trailer carried)" : "");
  free(buf); free(trow); free(tslot); free(tail); free(mu); free(rows); free(names);
}

/* ------------------------------------------------------------ fmt2 .cm --- */

/* A runtime mask: one state code per row with a key table. The bundled MRMP
 * of a model trained before artifacts were bundled whole (hg38_sex.clfx) is
 * this, as is anything mrmp-export wrote. Rows under no CpG take the PNA key
 * when the table has one, else code 0 -- the same rule the data lift uses. */
static void lift_cm(const char *in, const char *out, const int64_t *map,
                    uint64_t n_rows, uint64_t n_cpg, const char *pna_label,
                    int64_t limit) {
  cfile_t cf = open_cfile((char *)in);
  if (!cf.fh) ldie("cannot open mask", in);
  (void)limit;
  cdata_t c = read_cdata1(&cf);
  bgzf_close(cf.fh);
  if (!c.n) ldie("mask has no records", in);
  if (c.fmt != '2') ldie("the bundled MRMP is not a format-2 mask", in);
  decompress_in_situ(&c);
  fmt2_set_aux(&c);
  if (c.n != n_cpg) {
    char m[256];
    snprintf(m, sizeof m, "mask spans %" PRIu64 " rows but the coordinate track has "
             "%" PRIu64 " -- built on a different genome", (uint64_t)c.n, n_cpg);
    ldie(m, in);
  }
  f2_aux_t *aux = (f2_aux_t *)c.aux;
  const uint64_t nk0 = aux->nk;
  /* Where do rows under no CpG go? To the background key when the table has
   * one. A mask that covered every CpG never wrote one (mrmp-export names
   * only the keys it uses), so it is APPENDED here rather than borrowing key
   * 0, which would fold every unmapped probe into the first pattern. */
  uint64_t pna = nk0, nk = nk0;
  for (uint64_t k = 0; k < nk0; ++k) if (!strcmp(aux->keys[k], pna_label)) { pna = k; break; }
  int unmapped = 0;
  for (uint64_t t = 0; t < n_rows && !unmapped; ++t) if (map[t] < 0) unmapped = 1;
  const int append_pna = (pna == nk0) && unmapped;
  if (append_pna) nk = nk0 + 1;
  size_t keys_bytes = 0;
  for (uint64_t k = 0; k < nk0; ++k) keys_bytes += strlen(aux->keys[k]) + 1;
  if (append_pna) keys_bytes += strlen(pna_label) + 1;
  cdata_t d; memset(&d, 0, sizeof d);
  d.fmt = '2'; d.compressed = 0; d.unit = 8; d.n = n_rows;
  d.aux = calloc(1, sizeof(f2_aux_t));
  d.s = lalloc(keys_bytes + 1 + (size_t)n_rows * 8, 1, "lifted mask");
  f2_aux_t *da = (f2_aux_t *)d.aux;
  da->nk = nk; da->keys = lalloc(nk, sizeof(char *), "keys");
  size_t pos = 0;
  for (uint64_t k = 0; k < nk; ++k) {
    const char *key = k < nk0 ? aux->keys[k] : pna_label;
    size_t m = strlen(key);
    memcpy(d.s + pos, key, m); da->keys[k] = (char *)(d.s + pos);
    pos += m; d.s[pos++] = '\0';
  }
  d.s[pos++] = '\0';
  da->data = d.s + pos;
  uint64_t *per_key = lalloc(nk, sizeof(uint64_t), "per-key counts");
  for (uint64_t t = 0; t < n_rows; ++t) {
    uint64_t v = map[t] < 0 ? pna : f2_get_uint64(&c, (uint64_t)map[t]);
    if (v >= nk) v = pna;
    ++per_key[v];
    uint8_t *w = d.s + pos + t * 8;
    for (int b = 0; b < 8; ++b) w[b] = (uint8_t)(v >> (8 * b));
  }
  /* Report before compressing: cdata_compress() replaces d.s, and the key
   * pointers in the aux table point into the buffer it frees. */
  fprintf(stderr, "[methscope] mliftover: mask: %" PRIu64 " states over %" PRIu64
          " target rows;", nk, n_rows);
  for (uint64_t k = 0; k < nk; ++k)
    fprintf(stderr, " %s %" PRIu64, da->keys[k], per_key[k]);
  fputc('\n', stderr);
  cdata_compress(&d);
  cdata_write((char *)out, &d, "w", 0);
  free(per_key); free_cdata(&d); free_cdata(&c);
}

/* ------------------------------------------------------------- driver --- */

static int usage(FILE *out) {
  ms_help(out,
    "Usage:\n"
    "  methscope mliftover --to <platform> [options] <model> -o <out>\n\n"
    "Purpose:\n"
    "  Re-index a model onto an array platform's row space, so array data is\n"
    "  scored in its own space by a model built genome-wide: \"lift the model,\n"
    "  use the data as it is\". The model itself -- boosters, weights, the\n"
    "  deconvolution trailer -- is copied untouched; only the CpG index moves.\n"
    "  Reads any .clfx (the bundled .mrmp chain or .cm mask), a loose .mrmp or\n"
    "  .cm, or a .msdref deconvolution reference.\n\n"
    "  It refuses an upscaler (.updecx): its output is the genome, so it does\n"
    "  not lift. Carry the DATA the other way instead --\n"
    "  `sesame mliftover --to hg38 --simulated-depth 100 beta.cg beta.hg38.cg`\n"
    "  -- and run the original model on that.\n\n"
    "Options:\n"
    "  --to <platform>    Target platform (MSA, EPIC, HM450, ...). Its ordering\n"
    "                     (<platform>.ordering.tsv.gz: the probe IDs, in row\n"
    "                     order), its coordinate table (<platform>.<genome>\n"
    "                     .coord.tsv.gz) and the genome's cpg_nocontig.cr come\n"
    "                     from the store; `methscope fetch` names all three.\n"
    "  --genome <g>       The model's genome (default hg38).\n"
    "  --coord <file>     Coordinate table to use instead of the store's.\n"
    "  --ordering <file>  Ordering to use instead of the store's.\n"
    "  --cr <file>        Row-coordinate track to use instead of the store's.\n"
    "  -o <out>           Output path (required).\n"
    "  --pna-label <s>    The background key of a .cm mask (default Pna).\n"
    "  --min-retained N   Refuse if any pattern keeps fewer than N CpGs on the\n"
    "                     platform. Default 0: report and continue -- an emptied\n"
    "                     pattern stays a column and reads as missing.\n"
    "  --force            Overwrite an existing output.\n"
    "  -h                 Show this help message.\n\n"
    "Output:\n"
    "  The same kind of artifact, in the platform's row space (one row per\n"
    "  probe, ordering order), with its reference named after the platform.\n"
    "  Only cg probes are joined, decided by Probe_ID: an rs, nv or ch probe\n"
    "  that maps onto a CpG site carries a genotype or a non-CpG call, not\n"
    "  methylation, so it takes the background state whatever its coordinate\n"
    "  says. Those are counted separately from cg probes with no CpG row.\n"
    "  Per-set retained-CpG counts go to stderr: a feature that keeps 4% of\n"
    "  its CpGs is still an estimate of the same mean, but the reader should\n"
    "  see that number before trusting a call made on it.\n");
  return out == stdout ? 0 : 1;
}

int main_mliftover(int argc, char *argv[]) {
  const char *to = NULL, *genome = "hg38", *coord = NULL, *cr = NULL, *ordering = NULL;
  const char *out = NULL, *in = NULL, *pna_label = "Pna";
  uint64_t min_retained = 0; int force = 0;
  if (argc == 1) return usage(stderr);
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--to") && i + 1 < argc) to = argv[++i];
    else if (!strcmp(argv[i], "--genome") && i + 1 < argc) genome = argv[++i];
    else if (!strcmp(argv[i], "--coord") && i + 1 < argc) coord = argv[++i];
    else if (!strcmp(argv[i], "--ordering") && i + 1 < argc) ordering = argv[++i];
    else if (!strcmp(argv[i], "--cr") && i + 1 < argc) cr = argv[++i];
    else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--pna-label") && i + 1 < argc) pna_label = argv[++i];
    else if (!strcmp(argv[i], "--min-retained") && i + 1 < argc)
      min_retained = strtoull(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--force")) force = 1;
    else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) return usage(stdout);
    else if (argv[i][0] == '-' && argv[i][1]) ldie("unrecognized option", argv[i]);
    else if (!in) in = argv[i];
    else ldie("too many arguments", argv[i]);
  }
  if (!in || !out) return usage(stderr);
  if (!to && !(coord && cr && ordering))
    ldie("--to <platform> is required (or --coord, --ordering and --cr)", NULL);
  if (!force && access(out, F_OK) == 0) ldie("output exists (use --force)", out);
  const char *refname = to ? to : "lifted";

  /* the two tables, from the store unless given */
  char spec[1024]; char *own_coord = NULL, *own_cr = NULL, *own_ord = NULL;
  if (!coord) {
    snprintf(spec, sizeof spec, "%s/%s.%s.coord.tsv.gz", to, to, genome);
    coord = ms_model_resolve(spec, &own_coord);
    if (!coord) ldie("no coordinate table for this platform", spec);
  }
  if (!ordering) {
    snprintf(spec, sizeof spec, "%s/%s.ordering.tsv.gz", to, to);
    ordering = ms_model_resolve(spec, &own_ord);
    if (!ordering) ldie("no ordering for this platform", spec);
  }
  if (!cr) {
    snprintf(spec, sizeof spec, "%s/cpg_nocontig.cr", genome);
    cr = ms_model_resolve(spec, &own_cr);
    if (!cr) ldie("no row-coordinate track for this genome", spec);
  }
  uint64_t n_rows = 0, n_mapped = 0, n_noncg = 0, n_cpg = 0;
  int64_t *map = map_from_coords(coord, ordering, cr, &n_rows, &n_mapped, &n_noncg, &n_cpg);
  fprintf(stderr, "[methscope] mliftover: %s: %" PRIu64 " of %" PRIu64 " probes sit on "
          "a CpG row of %s (%" PRIu64 " rows); %" PRIu64 " not lifted by type (not cg), "
          "%" PRIu64 " cg probes with no CpG row\n", refname, n_mapped, n_rows, genome,
          n_cpg, n_noncg, n_rows - n_noncg - n_mapped);

  /* what is it */
  FILE *f = fopen(in, "rb");
  if (!f) ldie("cannot open input", in);
  unsigned char mg[8] = {0}; size_t got = fread(mg, 1, 8, f); fclose(f);
  if (got < 8) ldie("input too short", in);
  const int is_bundle = ms_bundle_is(in);
  const int is_mrmp = !memcmp(mg, "MRMPIDX1", 8);
  const int is_msdref = !memcmp(mg, "MSDREF1\0", 8);
  const int is_bgzf = mg[0] == 0x1f && mg[1] == 0x8b;
  const char *ext = strrchr(in, '.');

  if (is_bundle) {
    char *kind = ms_bundle_kind(in);
    if ((ext && !strcmp(ext, ".updecx")) || (kind && !strcmp(kind, "upscale")) ||
        ms_bundle_find(in, "outcpg", NULL)) {
      free(kind);
      ldie("an upscaler does not lift: its output is the genome. Carry the data "
           "instead -- `sesame mliftover --to hg38 --simulated-depth 100 beta.cg "
           "beta.hg38.cg` -- and run the original model on it", in);
    }
    free(kind);
    char tmp[4096];
    const char *td = getenv("TMPDIR");
    snprintf(tmp, sizeof tmp, "%s/methscope_lift_XXXXXX", td && *td ? td : "/tmp");
    int fd = mkstemp(tmp); if (fd < 0) ldie("cannot create a temp file", tmp); close(fd);
    if (is_mrmp) {
      /* the chain is the prefix; the walker stops at MSBNDL1 on its own */
      uint64_t below = ms_mrmp_lift(in, tmp, map, n_rows, refname, min_retained);
      if (below) { unlink(tmp); char m[128]; snprintf(m, sizeof m, "%" PRIu64 " pattern(s) keep fewer than --min-retained CpGs", below); ldie(m, in); }
    } else if (is_bgzf) {
      lift_cm(in, tmp, map, n_rows, n_cpg, pna_label, ms_bundle_cx_limit(in));
    } else ldie("bundle prefix is neither an MRMPIDX1 chain nor a .cm mask", in);
    ms_bundle_repack(out, tmp, in);
    unlink(tmp);
  } else if (is_mrmp) {
    uint64_t below = ms_mrmp_lift(in, out, map, n_rows, refname, min_retained);
    if (below) { unlink(out); char m[128]; snprintf(m, sizeof m, "%" PRIu64 " pattern(s) keep fewer than --min-retained CpGs", below); ldie(m, in); }
  } else if (is_msdref) {
    lift_msdref(in, out, map, n_rows, n_cpg);
  } else if (is_bgzf) {
    lift_cm(in, out, map, n_rows, n_cpg, pna_label, -1);
  } else ldie("not a bundle, .mrmp, .msdref or .cm", in);

  fprintf(stderr, "[methscope] mliftover: %s -> %s (%s row space, %" PRIu64 " rows)\n",
          in, out, refname, n_rows);
  free(map); free(own_coord); free(own_cr); free(own_ord);
  return 0;
}
