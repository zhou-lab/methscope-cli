// SPDX-License-Identifier: AGPL-3.0-or-later
/* Native MRMP (methylation reference membership pattern) construction.
 *
 * `methscope mrmp-build` / `mrmp-export` replace the YAME + awk + sort + text
 * pipeline that used to define the MRMP mask. mrmp-build reads a discretized
 * (pseudo-binary) reference .cg, reproduces `rowop -o binstring` per-CpG
 * resolution deterministically, counts exact membership patterns, ranks them,
 * and serializes a self-describing MRMPIDX1 artifact. `methscope inspect`
 * reports on it.
 *
 * A pattern is one fixed-length ternary string over the reference samples:
 *   '0' unmethylated, '1' methylated, '2' missing/ambiguous.
 * binstring emits '2' only as an all-or-nothing per-CpG sentinel, so the sole
 * pattern containing a '2' is the all-'2' string (PNA). Every other pattern is
 * pure {0,1}.
 *
 * A pattern is packed base-3, 40 trits per uint64 (3^40 < 2^64 <= 3^41), into
 * ceil(n_samples/40) words: word 0 holds samples 0..39, word 1 the next 40, and
 * within a word the most-significant digit is the lowest sample index. Word
 * order therefore compares the same way the sample string does. The reference
 * used to be capped at 40 samples; the width is now derived from `n_samples`,
 * which the header already carries, so every artifact written under the cap has
 * exactly one word and stays byte-identical -- no version bump, no compat
 * shim. Use mrmp_key_words() / mrmp_pattern_stride() rather than assuming. */
#ifndef MS_MRMP_H
#define MS_MRMP_H

#include <stdint.h>

#define MRMPIDX_MAGIC "MRMPIDX1"
#define MRMPIDX_VERSION 1u
#define MRMP_PNA_MEMBERSHIP 0xFFFFFFFFu   /* per-CpG sentinel: all-'2' / PNA */

/* all-0 and all-1 are candidates. Always set now that --no-include-homogeneous
 * is gone; kept in the header so older artifacts stay readable. */
#define MRMP_FLAG_INCLUDE_HOMOGENEOUS 1u

/* Per-pattern binarisation midpoints are present at thresh_offset.
 *
 * The threshold belongs with the patterns, not with the featurizer. It is the
 * midpoint between the mean reference beta of the classes a pattern calls 1 and
 * of those it calls 0, so only mrmp-build -- which has the reference open -- can
 * compute it, and any later recomputation risks disagreeing with the patterns it
 * is applied to. Storing it here lets it travel .mrmp -> .msfm -> .clfx, so a
 * shipped model binarises a new query exactly as training did.
 *
 * It must not be 0.5: inside a satellite the member classes are close
 * relatives, and a pattern can sit entirely above 0.5 where that cut carries no
 * information at all. NaN marks a pattern whose two groups are not both
 * populated, or whose reference puts the expected-0 group above the expected-1
 * group; such a pattern is unusable and consumers drop it. */
#define MRMP_FLAG_THRESH 2u

/* 128-byte fixed header; all little-endian, offsets are absolute file bytes. */
typedef struct {
  char     magic[8];          /* "MRMPIDX1" */
  uint32_t version;           /* MRMPIDX_VERSION */
  uint32_t n_samples;         /* reference samples == pattern length */
  uint32_t n_selected;        /* selectable patterns; == n_candidates since the
                               * top-K cut moved to the consumers. Older
                               * artifacts carry their build-time K, and every
                               * reader takes min(n_selected, its own K). */
  uint32_t flags;             /* MRMP_FLAG_* */
  uint64_t n_cpg;             /* genomic CpGs (per-CpG membership entries) */
  uint64_t n_candidates;      /* distinct {0,1} patterns (ranked; excl. PNA) */
  uint64_t pna_key;           /* base-3 key of the all-'2' sentinel, word 0.
                               * Informational: the sentinel is fully determined
                               * by n_samples, and PNA CpGs are flagged by the
                               * membership sentinel, not by matching this. */
  uint64_t pna_cpg;           /* CpGs resolved to the PNA sentinel */
  uint32_t mincov;            /* binstring -c (min coverage) */
  uint32_t pad0;
  float    beta_threshold;    /* binstring -b */
  float    max_ambig_frac;    /* binstring -m */
  float    min_major_fold;    /* binstring -M */
  float    pad1;
  uint64_t refname_offset;    /* NUL-terminated reference path */
  uint64_t names_offset;      /* n_samples NUL-terminated sample names */
  uint64_t patterns_offset;   /* n_candidates * mrmp_pattern_t, rank order */
  uint64_t membership_offset; /* n_cpg * uint32 rank (PNA sentinel = -1) */
  uint64_t content_checksum;  /* FNV-1a over the per-CpG key stream */
  uint64_t thresh_offset;     /* n_candidates float32 binarisation midpoints,
                               * or 0 in an artifact written before they were
                               * stored. See MRMP_FLAG_THRESH. */
} mrmp_header_t;

/* One ranked candidate pattern (rank == array index; label = P(rank+1)).
 *
 * On disk a record is mrmp_key_words(n_samples) key words followed by `count`,
 * i.e. mrmp_pattern_stride(n_samples) bytes. This struct is that record for the
 * one-word case, which is every artifact written before the width became
 * derived -- readers must stride by mrmp_pattern_stride(), not sizeof(). */
typedef struct {
  uint64_t key;    /* base-3 packed pattern (word 0) */
  uint64_t count;  /* CpGs carrying this exact pattern */
} mrmp_pattern_t;

#define MRMP_TRITS_PER_WORD 40u   /* 3^40 < 2^64 <= 3^41 */

/* uint64 words a pattern of n_samples occupies (>= 1, so 0 samples is benign). */
static inline uint32_t mrmp_key_words(uint32_t n_samples) {
  return n_samples ? (n_samples + MRMP_TRITS_PER_WORD - 1) / MRMP_TRITS_PER_WORD : 1u;
}

/* Bytes per on-disk pattern record: the key words plus the count. */
static inline uint64_t mrmp_pattern_stride(uint32_t n_samples) {
  return (uint64_t)mrmp_key_words(n_samples) * sizeof(uint64_t) + sizeof(uint64_t);
}

int main_mrmp_build(int argc, char *argv[]);
/* One 2-class satellite per (thin class, partner), as one MRMPSET1 container.
 * Thin == store labels minus the global's, so the split needs no side file. */
int main_mrmp_build_thin(int argc, char *argv[]);
int main_mrmp_export(int argc, char *argv[]);
int main_mrmp_inspect(int argc, char *argv[]);
int main_mrmp_pack(int argc, char *argv[]);
/* Pool sets into one container AND cut to a shared column budget. Distinct
 * from mrmp-pack, which concatenates without selecting. */
int main_mrmp_pool(int argc, char *argv[]);
int main_mrmpset_inspect(const char *path);   /* the MRMPIDX1 arm of `inspect` */

/* The artifact is the build pipeline's currency: `upscale-featurize`,
 * `upscale-set-units`, and `upscale-train` all read it, so the per-CpG mask and
 * the group map they use cannot drift apart.  The .cm exists only as the
 * runtime form, materialized into the model bundle. */

/* Nonzero if PATH is a MRMPIDX1 artifact rather than an exported .cm mask. */
int ms_mrmp_is_artifact(const char *path);

/* Write the artifact's per-CpG P1..P<top_k> / PNA labels as a YAME format-2
 * .cm. The caller owns top_k: for a model bundle it is the model's MRMP input
 * count, so the shipped mask cannot disagree with the input dimension.
 * Fatal on error. */
void ms_mrmp_write_mask(const char *artifact, const char *out_cm,
                        const char *pna_label, uint32_t top_k);

/* Fill group[0..n_cpg-1] with each CpG's 1-based selected-pattern index, or 0
 * for PNA and ranks at or beyond `patterns`. Fatal on error or size mismatch. */
void ms_mrmp_group_map(const char *artifact, uint16_t *group, uint64_t n_cpg,
                       uint32_t patterns);

/* Decoded top-K view: the per-pattern binstrings and their CpG counts, plus the
 * reference sample names. For a per-cell-type reference those names ARE the
 * class labels, and binstring position order matches them -- which is what lets
 * the `violation` framework build a classifier straight from the artifact, with
 * no training data and no fitted parameter. */
typedef struct {
  uint32_t  n_samples;    /* binstring length == number of labels */
  uint32_t  n_patterns;   /* min(top_k, n_candidates) */
  char    **labels;       /* n_samples reference sample names */
  char    **binstring;    /* n_patterns strings, n_samples chars each */
  uint64_t *count;        /* n_patterns CpG counts */
} mrmp_top_t;

/* Read the top `top_k` ranked patterns. Fatal on error; free with the below. */
mrmp_top_t *ms_mrmp_top_read(const char *artifact, uint32_t top_k);
void ms_mrmp_top_free(mrmp_top_t *t);

/* ---------------- MRMPSET1: several MRMP sets in one artifact -------------
 *
 * One global MRMP cannot serve many similar cell types: its CpG filter is a
 * conjunction across every class, so the chance a CpG is spoiled by at least
 * one class is 1-(1-p)^N -- 98% at N=41 against 26% at N=3. The answer is a
 * global set plus small "satellite" sets over confusable blocks, each free to
 * be strict precisely because it constrains few classes. That turns the MRMP
 * from one object into several, and they have to travel together: a feature
 * vector fused across sets is meaningless if the sets can drift apart.
 *
 * The container is deliberately dumb. Each block is a complete MRMPIDX1
 * artifact, byte-for-byte, at some offset -- so every reader above works on a
 * block once told where it starts, and a one-set container is exactly today's
 * artifact plus a wrapper. A block's internal offsets stay relative to that
 * block, which is what makes "write standalone, then concatenate" correct.
 */
#define MRMPSET_MAGIC "MRMPSET1"
#define MRMPSET_VERSION 1u

typedef struct {
  uint64_t block_offset;   /* absolute file offset of a complete MRMPIDX1 */
  uint64_t block_bytes;
} mrmpset_entry_t;

/* 64-byte fixed header; little-endian, offsets absolute. */
typedef struct {
  char     magic[8];       /* "MRMPSET1" */
  uint32_t version;
  uint32_t n_sets;
  uint32_t flags;
  uint32_t reserved;
  uint64_t table_offset;   /* n_sets * mrmpset_entry_t */
  uint64_t names_offset;   /* n_sets NUL-terminated set names, set order */
  uint64_t file_bytes;
  uint64_t reserved2[2];
} mrmpset_header_t;

/* Opened container: names and block extents, copied out so the file can close. */
typedef struct {
  uint32_t   n_sets;
  char     **name;         /* n_sets */
  uint64_t  *block_off;    /* n_sets */
  uint64_t  *block_bytes;  /* n_sets */
} ms_mrmpset_t;

/* Nonzero if PATH is a MRMPSET1 container rather than a bare MRMPIDX1. */
int ms_mrmpset_is(const char *path);

/* Open/close. Fatal on a bad or truncated container. */
ms_mrmpset_t *ms_mrmpset_open(const char *path);
void ms_mrmpset_free(ms_mrmpset_t *s);

/* Write one. `block[i]` are complete MRMPIDX1 images with their own lengths;
 * the container copies them and owns nothing afterwards. Fatal on I/O error. */
void ms_mrmpset_write(const char *out, uint32_t n_sets, const char *const *name,
                      const void *const *block, const uint64_t *block_bytes);

/* Block-addressed forms of the two readers above. `base` is a block_offset from
 * the container; 0 addresses a bare MRMPIDX1, which is what the un-suffixed
 * functions pass. */
mrmp_top_t *ms_mrmp_top_read_at(const char *path, uint64_t base, uint32_t top_k);
void ms_mrmp_group_map_at(const char *path, uint64_t base, uint16_t *group,
                          uint64_t n_cpg, uint32_t patterns);

#endif /* MS_MRMP_H */
