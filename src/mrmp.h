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
 * Each is the midpoint between the mean reference beta of the classes a pattern
 * calls 1 and of those it calls 0, so only mrmp-build -- which has the reference
 * open -- can compute it. NaN marks a pattern whose two groups are not both
 * populated, or whose reference puts the expected-0 group above the expected-1
 * group; such a pattern is unusable and consumers drop it.
 *
 * NOT the default cut. classify-featurize binarises pattern betas at a flat 0.5
 * unless asked for these with --thresh-pattern. 0.5 is an absolute call, so it
 * means the same thing on a cohort this reference never saw; a midpoint fitted
 * to this reference's two groups separates a close pair better but travels
 * worse, and travelling is the point once a global shift in methylation moves
 * every beta at once. Kept stored so the comparison can be made, and because
 * the violation framework scores against them directly. */
#define MRMP_FLAG_THRESH 2u

/* The membership section is RLE-compressed (a YAME format-2 payload) and is
 * `membership_bytes` long; without this flag it is the legacy dense array of
 * n_cpg uint32. Readers inflate on open, so `membership` is an owned array
 * either way and every consumer indexes it identically.
 *
 * Dense membership is one uint32 per genomic CpG whether or not the set has
 * anything to say there, and ~99% of it is the PNA sentinel -- so a 2-class
 * satellite describing ~1,200 CpGs cost the same 87 MB as a 34-class global.
 * That was the whole size of a pooled container (8.2 GB for 100 sets) against
 * 6 MB for the same sets exported as .cm, which is the same RLE applied one
 * step later. Compressing in the artifact closes that gap.
 *
 * The flag is absent in artifacts written before this, and `membership_bytes`
 * reuses a formerly zeroed pad, so old files stay readable with no version
 * bump -- the same compatibility discipline the derived key width used. */
#define MRMP_FLAG_MEMB_RLE 4u

/* The RLE membership payload is additionally BGZF-compressed, and the section
 * is [uint64 rle_bytes][BGZF blocks] -- the length is carried inline because the
 * 128-byte header has no room left.
 *
 * Deflate is where the remaining size is: measured on a 34-class global, the RLE
 * alone is 1,717,137 bytes and BGZF takes it to ~620 K, a 2.74x that dwarfs
 * every other lever. Folding non-selected patterns into the background -- the
 * one thing an exported .cm does that this does not -- is worth only 4.4%, which
 * is why a .mrmp is not pruned and does not need to be.
 *
 * BGZF rather than a raw deflate stream so the payload is the same framing YAME
 * writes everywhere else; the block index it enables is unused here, since the
 * section is always inflated whole. */
#define MRMP_FLAG_MEMB_BGZF 8u

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
  uint32_t membership_bytes;  /* on-disk size of the membership section when
                               * MRMP_FLAG_MEMB_RLE is set. 0 (a zeroed pad in
                               * every older artifact) means the dense form,
                               * whose size is n_cpg * 4. */
  float    beta_threshold;    /* binstring -b */
  float    max_ambig_frac;    /* binstring -m */
  float    min_major_fold;    /* binstring -M */
  uint32_t name_offset;       /* block-relative offset of the set's NUL-terminated
                               * name, or 0 for an unnamed set (which is what
                               * every artifact written before this reads as, and
                               * what a reader labels positionally).
                               *
                               * The name lives in the block because the block is
                               * the unit that travels: sets are concatenated, so
                               * anything held outside a block would be lost the
                               * moment two files are cat'd together. */
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
/* One 2-class satellite per (thin class, partner), as one chain.
 * Thin == store labels minus the global's, so the split needs no side file. */
int main_mrmp_build_thin(int argc, char *argv[]);
/* One 2-class satellite per (class, near neighbour) over the classes the global
 * already covers, as one chain. Overlapping pairs, NOT a partition:
 * a class appears in as many sets as it has close neighbours. */
int main_mrmp_build_neighbor(int argc, char *argv[]);
int main_mrmp_export(int argc, char *argv[]);
int main_mrmp_inspect(int argc, char *argv[]);
/* Pool sets into one chain AND cut to a shared column budget. Concatenation
 * alone is just `cat`; this is the step that selects. */
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

/* Block-addressed form: `base`/`blk_bytes` are a block_offset/block_bytes pair
 * from a walked chain, and (0, 0) addresses the first block. This is what
 * lets a POOLED artifact reach the featurizer: mrmp-pool emits a container, and
 * classify-featurize consumes .cm masks, so without a per-block export the four
 * -command workflow has no path from one to the other. */
void ms_mrmp_write_mask_at(const char *artifact, uint64_t base,
                           uint64_t blk_bytes, const char *out_cm,
                           const char *pna_label, uint32_t top_k);

/* Per-pattern binarisation midpoints for a block, in rank order. Fills up to
 * n_want and returns how many were written; 0 if the artifact predates
 * MRMP_FLAG_THRESH. A NaN entry marks a pattern whose two groups are not both
 * populated -- unusable, and consumers drop it rather than guess. */
uint32_t ms_mrmp_thresholds_at(const char *path, uint64_t base, uint64_t blk_bytes,
                               uint32_t n_want, float *out);

/* Walk the membership as RUNS rather than expanding it.
 *
 * The section is run-length coded and overwhelmingly PNA -- a 34-class global is
 * 20,015 runs over 21.8M CpGs, and a satellite fewer still -- so a consumer that
 * only cares where patterns ARE can skip the background in a few dozen steps
 * instead of touching every genomic position. That is the difference between
 * setup costing O(n_sets * n_cpg) and O(total memberships): 2.18 billion visits
 * against 865 thousand on a 100-set artifact.
 *
 * `cb` is called once per run, in position order, including PNA runs (rank ==
 * MRMP_PNA_MEMBERSHIP) so a caller can track position without arithmetic. */
typedef void (*ms_mrmp_run_cb)(void *ctx, uint64_t start, uint64_t len,
                               uint32_t rank);
void ms_mrmp_membership_runs(const char *path, uint64_t base, uint64_t blk_bytes,
                             ms_mrmp_run_cb cb, void *ctx);

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

/* ---------------- several sets in one file: a CHAIN, not a container ------
 *
 * One global MRMP cannot serve many similar cell types: its CpG filter is a
 * conjunction across every class, so the chance a CpG is spoiled by at least
 * one class is 1-(1-p)^N -- 98% at N=41 against 26% at N=3. The answer is a
 * global set plus small "satellite" sets over confusable blocks, each free to
 * be strict precisely because it constrains few classes. That turns the MRMP
 * from one object into several, and they have to travel together: a feature
 * vector fused across sets is meaningless if the sets can drift apart.
 *
 * They travel by CONCATENATION. A file is one or more complete MRMPIDX1 blocks
 * laid end to end -- no wrapper, no table, no second magic. A reader walks it:
 * read a 128-byte header, check the magic, take the block's own size, seek past
 * it, repeat until EOF. A single-set file is exactly one block, so "one set" and
 * "many sets" are the same format and there is no .mrmp / .mrmpset ambiguity.
 *
 * This replaces an earlier MRMPSET1 wrapper that put an offset table and a name
 * list at the front. Three things fall out of dropping it:
 *
 *   cat a.mrmp b.mrmp > c.mrmp  is a legal, exact combine, so `mrmp-pack` is
 *   just `cat` and no longer exists.
 *
 *   Appending is O(1) rather than a rewrite of a front table.
 *
 *   A set's NAME rides inside its block (see name_offset), so it survives the
 *   concatenation. A front name list could not.
 *
 * A block's size is DERIVED, not stored: sections are laid out in header order,
 * so the block ends after the last one, rounded up to 8. See ms_mrmp_block_bytes.
 * The rounding is what keeps the next block's header 8-aligned, which a reader
 * casting the header in place requires -- and it is why every writer pads.
 *
 * There is deliberately no end-of-chain marker. A trailer at EOF would sit in
 * the MIDDLE of the chain after a cat, which is the one property worth
 * protecting. Truncation is instead caught by the bounds check: a file that ends
 * mid-block has a block claiming more bytes than remain. Only a cut landing
 * exactly on a block boundary is silent, and that is not what a failed write or
 * a full disk produces.
 */

/* Bytes this block occupies, including the pad that 8-aligns the next one.
 * Sections sit in header order, so the end is the furthest section end; taking
 * the max rather than assuming which is last keeps this correct if the layout
 * ever gains a section. */
/* Where the block's last section ends, WITHOUT the alignment pad. A block
 * written before padding existed stops exactly here, so this is what a bounds
 * check must compare against -- rounding first makes the final block of an old
 * file look one byte past EOF and rejects a perfectly good artifact. */
static inline uint64_t ms_mrmp_block_end(const mrmp_header_t *h) {
  uint64_t end = sizeof(mrmp_header_t);
  uint64_t memb = (h->flags & MRMP_FLAG_MEMB_RLE)
                ? h->membership_bytes : h->n_cpg * sizeof(uint32_t);
  uint64_t cand = h->patterns_offset + h->n_candidates * mrmp_pattern_stride(h->n_samples);
  if (cand > end) end = cand;
  if (h->membership_offset + memb > end) end = h->membership_offset + memb;
  if ((h->flags & MRMP_FLAG_THRESH) &&
      h->thresh_offset + (uint64_t)h->n_candidates * sizeof(float) > end)
    end = h->thresh_offset + (uint64_t)h->n_candidates * sizeof(float);
  return end;
}

/* The same, rounded up to the 8-byte boundary the NEXT block starts on. Every
 * writer pads to this, so it is the stride through a chain. */
static inline uint64_t ms_mrmp_block_bytes(const mrmp_header_t *h) {
  return (ms_mrmp_block_end(h) + 7u) & ~7ull;
}

/* A walked chain: one entry per block, in file order. Same shape the old
 * container view had, so every consumer of it is unchanged. */
typedef struct {
  uint32_t   n_sets;
  char     **name;         /* n_sets; synthesized "set<N>" when unnamed */
  uint64_t  *block_off;    /* n_sets */
  uint64_t  *block_bytes;  /* n_sets */
} ms_mrmpset_t;

/* Walk PATH's chain. Fatal on a bad magic or a block running past EOF. */
ms_mrmpset_t *ms_mrmpset_open(const char *path);
void ms_mrmpset_free(ms_mrmpset_t *s);

/* Write a chain: `block[i]` are complete MRMPIDX1 images, each already padded to
 * a multiple of 8 by its builder, written back to back. Nothing is added -- the
 * result is byte-for-byte what `cat` of the same blocks would produce, which is
 * the property that makes `cat` a supported combine. Fatal on I/O error. */
void ms_mrmp_chain_write(const char *out, uint32_t n_sets,
                         const void *const *block, const uint64_t *block_bytes);

/* Block-addressed forms of the two readers above. `base` is a block_off from a
 * walked chain; 0 addresses the first (or only) block. */
mrmp_top_t *ms_mrmp_top_read_at(const char *path, uint64_t base, uint32_t top_k);
void ms_mrmp_group_map_at(const char *path, uint64_t base, uint16_t *group,
                          uint64_t n_cpg, uint32_t patterns);

#endif /* MS_MRMP_H */
