// SPDX-License-Identifier: AGPL-3.0-or-later
/* MSFMAT1 -- a featurized record x pattern matrix, shared by the writer
 * (msfm.c) and its consumers (train.c, predict.c) so the two cannot drift.
 *
 * This is a `.msur` stripped to what a LABEL model needs. The upscale msur is
 * dominated by things only a CpG-level decoder uses -- the embedded truth
 * matrix (n_cells x n_cpg x 2 B: 4.97 TB for 84,601 cells), the CpG->pattern
 * groups map, and a per-record observed set. A classifier needs none of them:
 * its target is a class id, not 29.4M CpGs. What is left is the summary the
 * featurization already computes.
 *
 * It is also structurally simpler than msur. Every row is the same width, so
 * there is no replicate table, no variable-length addressing and no
 * list/bitmap encoding choice -- a sparsity-ladder replicate is just another
 * row. Addressing is record * n_patterns, and the whole file mmaps.
 *
 * Layout (offsets are absolute, sections in this order):
 *   [0]              header, 88 bytes
 *   [names_offset]   n_patterns NUL-terminated pattern names
 *   [rows_offset]    n_records  NUL-terminated record names
 *   [labels_offset]  u16 class id per record, then n_classes NUL-terminated
 *                    class names (sorted, so the id IS the model's class index)
 *   [levels_offset]  u32 covered-CpG total per record
 *   [beta_offset]    u16 per (record, pattern), row-major
 * count_offset is a RESERVED header slot, always 0. A per-pattern count block
 * was specified but never populated: the sampled builder is the only featurize
 * path and it tallied the counts to average the betas, then dropped them. Rather
 * than plumb it through, `classify-featurize --counts N` now spends that
 * information where it matters -- a beta backed by fewer than N measured CpGs is
 * recorded MISSING instead of kept -- which needs no format change, since every
 * reader already handles MSFM_NA. The slot stays so the 88-byte header layout,
 * and therefore every existing .msfm, is unchanged.
 */
#ifndef MSFM_H
#define MSFM_H

#include <stdint.h>
#include <string.h>

#define MSFM_MAGIC "MSFMAT1"

/* Beta is u16 fixed point, exactly the convention msur uses for its truth
 * matrix: code/65534 in [0,1], and MSFM_NA for missing. Half the size of f32,
 * and 1/65534 is far finer than a beta backed by a few hundred CpGs warrants. */
#define MSFM_NA UINT16_MAX

static inline uint16_t msfm_encode(double beta) {
  if (!(beta >= 0.0) && !(beta <= 0.0)) return MSFM_NA;   /* NaN */
  if (beta < 0.0) beta = 0.0;
  if (beta > 1.0) beta = 1.0;
  return (uint16_t)(beta * 65534.0 + 0.5);
}
static inline double msfm_decode(uint16_t v) {
  return v == MSFM_NA ? (0.0 / 0.0) : (double)v / 65534.0;
}

#pragma pack(push, 1)
typedef struct {
  char     magic[8];        /* "MSFMAT1\0" */
  uint32_t version;         /* 1 */
  uint32_t n_records;
  uint32_t n_patterns;
  uint32_t n_classes;
  uint32_t flags;
  uint32_t reserved;
  uint64_t names_offset;
  uint64_t rows_offset;
  uint64_t labels_offset;
  uint64_t levels_offset;
  uint64_t beta_offset;
  uint64_t count_offset;    /* RESERVED, always 0 -- see the header comment */
  uint64_t file_bytes;
} msfm_header_t;
#pragma pack(pop)

static inline int msfm_is(const msfm_header_t *h) {
  return !memcmp(h->magic, MSFM_MAGIC, 7) && h->version == 1;
}

/* An opened (mmapped) artifact. */
typedef struct {
  int              fd;
  void            *map;
  uint64_t         length;
  const msfm_header_t *header;
  const uint16_t  *beta;        /* n_records * n_patterns */
  const uint32_t  *levels;      /* n_records */
  const uint16_t  *class_id;    /* n_records */
  char           **pattern_names;
  char           **record_names;
  char           **class_names;
} ms_msfm_t;

/* Open/close. Returns 0 and fills err on a bad or truncated file. */
int  ms_msfm_open(ms_msfm_t *f, const char *path, char *err, size_t errn);
void ms_msfm_close(ms_msfm_t *f);

/* Materialize an artifact as the ms_matrix_t every label model already expects,
 * so `--data` changes only where the features come from, never what is done
 * with them. *labels_out gets one malloc'd label per record (NULL when the
 * artifact is unlabeled) and *levels_out a malloc'd per-record covered-CpG
 * total; both are caller-freed. Either out-pointer may be NULL. */
ms_matrix_t *ms_msfm_to_matrix(const char *path, char ***labels_out,
                               uint32_t **levels_out);

/* Featurize a query .cg into the ms_matrix_t the label models expect, using the
 * threaded scatter-add builder instead of the genome-scanning summarize1 path.
 * Semantically identical -- verified bit-for-bit over 4.3M u16 codes -- but
 * O(covered CpGs) per record rather than O(genome), and it uses `threads`
 * workers. Needs the query's .cg index, since workers seek per record.
 *
 * *levels_out (optional) receives the per-record covered-CpG total, which is
 * what a --scalar-coverage model needs; caller frees. Betas round-trip through
 * u16 (step 1.5e-5), so a confidence can differ in its last decimals from the
 * double-precision path -- far below the sampling noise on any real beta, and
 * predicted labels were unaffected across every comparison run so far. */
ms_matrix_t *ms_matrix_build_threaded(const char *query, const char *mrmp,
                                      uint32_t patterns, unsigned threads,
                                      uint32_t **levels_out);

/* `methscope inspect` reporter. */
void ms_msfm_report(const char *path);

/* Sampled, threaded featurization (msfm_build.c). Downsamples and summarizes in
 * one pass: each cell is inflated once and every replicate is drawn from that
 * copy, summarized by scatter-add over the CpG->pattern map rather than a
 * genome-wide scan. rep_sample[r] is the target covered-CpG count for replicate
 * r (0 = keep native coverage). Output is (n_reps * n_cells) rows, replicate
 * major. Caller frees *beta_out, *levels_out, and *names_out (and its strings). */
void ms_msfm_build_sampled(const char *query, const char *mrmp, uint32_t patterns,
                           const uint32_t *rep_sample, uint32_t n_reps,
                           int binarize, uint32_t min_cpgs,
                           uint64_t seed, unsigned threads, int binarize_feat,
                           uint16_t **beta_out, uint32_t **levels_out,
                           char ***names_out, uint32_t *n_cells_out,
                           uint32_t *ncol_out);

/* Same, over N MRMP sets in ONE pass over the query. Column layout is
 * set-major: set s owns [set_col0[s], set_col0[s] + patterns[s]], the last of
 * those being its PNA. With n_sets == 1 this is byte-identical to
 * ms_msfm_build_sampled, which calls it.
 *
 * The reason this exists rather than N calls: a cell's decompress dominates the
 * cost (the header note above), so scoring one inflated copy against every set
 * turns N passes into one. Merging the masks instead would not work -- a .cm
 * gives each CpG exactly one pattern, so overlapping sets would fight over the
 * CpGs they share, and those are precisely the informative ones.
 *
 * set_col0_out (optional) receives the first column of each set. */
void ms_msfm_build_sampled_multi(const char *query, const char *const *mrmps,
                           const uint64_t *mrmp_base, const uint64_t *mrmp_len,
                           const uint32_t *patterns, uint32_t n_sets,
                           const uint32_t *rep_sample, uint32_t n_reps,
                           int binarize, uint32_t min_cpgs,
                           uint64_t seed, unsigned threads, int binarize_feat,
                           uint16_t **beta_out, uint32_t **levels_out,
                           char ***names_out, uint32_t *n_cells_out,
                           uint32_t *ncol_out, uint32_t *set_col0_out);

/* Subcommand entry point. */
int  main_classify_featurize(int argc, char *argv[]);

#endif /* MSFM_H */
