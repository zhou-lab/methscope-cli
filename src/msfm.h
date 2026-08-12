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

/* header.flags -- how the PATTERN columns are coded. Recorded because the two
 * scoring routes build features differently: classify --data reads this matrix,
 * but classify on a raw .cg goes through ms_matrix_build(), which computes a
 * continuous mean and has no cut. A model trained on {0,1,NA} scored against
 * betas in [0,1] is a silent feature-space mismatch -- measured at 2 of 42
 * cells correct. So the coding travels with the artifact, and from there into
 * the booster (MS_ATTR_BINARIZE), instead of being re-derived. 0 = continuous,
 * which is also what every artifact written before 2026-08-09 implies. */
#define MSFM_FLAG_BIN_FLAT  1u   /* pattern columns cut at a flat 0.5 */
#define MSFM_FLAG_BIN_PAT   2u   /* cut at per-pattern midpoints (--thresh-pattern) */
/* Satellite CONTRAST columns. A 2-class set's two patterns have opposite
 * polarity, so "is P1 above P2" answers the same question as "is P1 above 0.5"
 * -- but relatively, so anything shifting both patterns together cancels. That
 * is the violation rule's argmin cancellation, expressed as one feature.
 *
 * Measured on Bian colorectal carcinoma, where global hypomethylation
 * compresses both patterns toward 0.5 (P1 0.845 -> 0.537, P2 0.100 -> 0.429):
 * the flat cut calls colon on 55.3% of left-colon cells, the paired contrast on
 * 87.7%, and on adjacent-normal cells the two agree exactly (95.4%). The
 * contrast costs nothing in distribution and holds up out of it. */
#define MSFM_FLAG_CONTRAST_ADD  4u   /* satellites emit P1, P2 AND the contrast */
#define MSFM_FLAG_CONTRAST_ONLY 8u   /* satellites emit the contrast INSTEAD */
/* Rank features: one column PER CLASS of a set, holding "this cell's evidence
 * for class c outranks its evidence against c" -- the mean over the set's
 * patterns calling c '1' against the mean over those calling c '0'. It is the
 * 2-class contrast generalised to any k, and it survives a global shift for the
 * same reason: the shift enters both means and cancels in the difference.
 *
 * Measured on 2,616 Gaiti CLL cells against the 33-class human root: the
 * patterns expecting B.Cell '1' sit at 0.777 and those expecting B.Cell '0' at
 * 0.135, so the order says B.Cell in 98.9% of cells with both sides observed --
 * against 90.8% for the same cells' binarised absolute calls. */
#define MSFM_FLAG_RANK_ADD      16u  /* per-class rank columns ALONGSIDE patterns */
#define MSFM_FLAG_RANK_ONLY     32u  /* per-class rank columns INSTEAD of them */
/* Ties (beta exactly on the cut) are MSFM_NA under both; see msfm_build.c. */

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
  uint64_t mrmp_offset;     /* the MRMP CHAIN this matrix was featurized against,
                             * as [uint64 bytes][chain], or 0 when absent.
                             *
                             * Was a reserved pad, zero in every artifact ever
                             * written, so old files read as "no chain" -- the
                             * truth about them -- and need no version bump.
                             *
                             * Carrying it makes the matrix self-describing: the
                             * column layout is (chain + flags), so a consumer
                             * can find any set's columns from the file alone,
                             * and a matrix CANNOT be paired with the wrong tree
                             * because it brings its own. */
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
/* Read just header.flags from PATH (see MSFM_FLAG_*). Fatal if unreadable. */
uint32_t ms_msfm_flags(const char *path);

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
/* Where each set's columns live in a fused .msfm.
 *
 * (chain + flags) determines the column layout exactly -- set-major in chain
 * order, block pattern-rank within a set, and under --rank-features one column
 * per SEPARABLE class pair. Nothing else is needed, so nothing else is stored.
 *
 * This is the ONE place that rule lives. ms_msfm_build_sampled_multi() asserts
 * its own emitted count against it, because two implementations of one rule is
 * the failure this codebase keeps hitting: an n_sets == 1 fast path that never
 * grew what the multi-set path did, two filter legs running the identical
 * comparison at different thresholds, and an analysis assuming column 0 was the
 * A-high pattern. All three were silent.
 *
 * Chain inputs only: --patterns caps the FIRST set of a loose .cm list, which
 * this cannot see. From a chain every block reports its own count. */
typedef struct {
  uint32_t   n_sets;
  char     **name;      /* n_sets, owned */
  uint32_t  *col0;      /* first column of each set */
  uint32_t  *ncol;      /* columns it owns */
  uint32_t   total;     /* == sum(ncol) == the matrix width */
} ms_msfm_layout_t;

/* `flags` are MSFM_FLAG_* as recorded in a written artifact's header, so a
 * layout can be recovered from (chain, header) with no other context. */
/* The chain a matrix was featurized against, written to a temp file whose path
 * is returned (caller frees the string and unlinks). NULL when the artifact
 * predates embedding. Everything that needs the layout can then work from the
 * .msfm alone. */
char *ms_msfm_chain(const char *msfm);

ms_msfm_layout_t *ms_msfm_layout(const char *chain, uint32_t flags);
void ms_msfm_layout_free(ms_msfm_layout_t *l);

void ms_msfm_build_sampled_multi(const char *query, const char *const *mrmps,
                           const uint64_t *mrmp_base, const uint64_t *mrmp_len,
                           const uint32_t *patterns, uint32_t n_sets,
                           const uint32_t *rep_sample, uint32_t n_reps,
                           int binarize, uint32_t min_cpgs,
                           uint64_t seed, unsigned threads, int binarize_feat,
                           int contrast, int rank,
                           uint16_t **beta_out, uint32_t **levels_out,
                           char ***names_out, uint32_t *n_cells_out,
                           uint32_t *ncol_out, uint32_t *set_col0_out);

/* Subcommand entry point. */
int  main_classify_featurize(int argc, char *argv[]);

#endif /* MSFM_H */
