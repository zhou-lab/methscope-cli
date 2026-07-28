// SPDX-License-Identifier: AGPL-3.0-or-later
/* MSURAW on-disk layout, shared by the writer (upscale_prepare.c) and both
 * training backends (upunit_cpu.c, upunit_cuda.cu) so the three cannot drift.
 *
 * MSURAW2 -- fixed-width records.  Every record is
 *   [beta xP f32][count xP u32][selected xN u32]
 * with one `sampled_per_cell` (N) and one `record_bytes` in the header, so a
 * record is addressed as records_offset + row * record_bytes.  Mixing coverages
 * in one msur meant padding every short record out to the widest level, which
 * on a 143-CpG replicate is 99.9% filler.
 *
 * MSURAW3 -- indexed variable-length records.  Sparsity is a per-replicate
 * property, so every record *within* a replicate is still the same size and
 * O(1) addressing survives with two small tables instead of padding:
 *   record(rep, cell) = rep[rep].offset + cell * rep[rep].record_bytes
 * The tables cost 24 bytes per replicate (a few KB) and replace tens of GB of
 * 0xFFFFFFFF.  Header keeps the 72-byte MSURAW2 prefix and appends one offset;
 * `sampled_per_cell` becomes the widest level (informational) and `record_bytes`
 * is 0, a sentinel so a version-blind reader cannot silently mis-address. */
#ifndef MSUR_H
#define MSUR_H

#include <stdint.h>
#include <string.h>

#define MSUR2_MAGIC "MSURAW2\0"
#define MSUR3_MAGIC "MSURAW3\0"

#define MSUR_F_TRUTH_U16    1u   /* embedded uint16 truth matrix (trainable) */
#define MSUR_F_BINARIZED    2u   /* observed betas Bernoulli-drawn, one read/CpG */
#define MSUR_F_MIXED_SAMPLE 4u   /* replicates differ in --sample */

/* 72-byte prefix, identical in v2 and v3. */
typedef struct {
  char magic[8];
  uint32_t version, n_cells, n_reps, n_patterns;
  uint64_t n_cpg;
  uint32_t sampled_per_cell, flags;
  uint64_t groups_offset, truth_offset, records_offset, record_bytes;
} msur_header_t;

/* v3 only: appended after the prefix, so the struct is 80 bytes on disk. */
typedef struct {
  msur_header_t h;
  uint64_t rep_table_offset;
} msur_header3_t;

/* Observed-set encoding, per replicate (msur_rep_t.flags).
 *
 * A sorted u32 list costs 4 bytes per observed CpG, so it is right while a
 * replicate is sparse -- but it grows without bound, and past n_cpg/32
 * observed CpGs a bitmap over the whole genome is smaller AND turns the
 * membership test from a binary search into one bit read. Deciding per
 * replicate keeps the sparse levels cheap and caps the dense ones: on the D2
 * ladder (500 reps, 71 -> 10% of hg38) records go 194.7 -> 127.2 GiB with only
 * the top 104 replicates switching.
 *
 * v2 files, and any v3 file written before this existed, carry flags == 0 ==
 * MSUR_ENC_LIST and so keep reading exactly as before. */
#define MSUR_ENC_LIST   0u
#define MSUR_ENC_BITMAP 1u

static inline uint64_t msur_bitmap_bytes(uint64_t n_cpg) { return (n_cpg + 7) / 8; }

/* One per replicate, at rep_table_offset. */
typedef struct {
  uint32_t sample;        /* observed CpGs per cell in this replicate */
  uint32_t flags;         /* MSUR_ENC_* -- observed-set encoding */
  uint64_t record_bytes;  /* patterns*8 + (list ? sample*4 : bitmap bytes) */
  uint64_t offset;        /* absolute byte offset of this replicate's block */
} msur_rep_t;

static inline int msur_is_v3(const msur_header_t *h) {
  return !memcmp(h->magic, MSUR3_MAGIC, 8) && h->version == 3;
}
static inline int msur_is_v2(const msur_header_t *h) {
  return !memcmp(h->magic, MSUR2_MAGIC, 8) && h->version == 2;
}

static inline const msur_rep_t *msur_reps(const void *base) {
  const msur_header3_t *h3 = (const msur_header3_t *)base;
  return (const msur_rep_t *)((const unsigned char *)base + h3->rep_table_offset);
}

/* Observed CpGs for a replicate -- constant in v2, per-replicate in v3. */
static inline uint32_t msur_sample_of(const void *base, const msur_header_t *h,
                                      uint32_t rep) {
  return msur_is_v3(h) ? msur_reps(base)[rep].sample : h->sampled_per_cell;
}

/* Start of one (replicate, cell) record. */
static inline const unsigned char *msur_record(const void *base,
                                               const msur_header_t *h,
                                               uint32_t rep, uint32_t cell) {
  const unsigned char *p = (const unsigned char *)base;
  if (msur_is_v3(h)) {
    const msur_rep_t *r = msur_reps(base) + rep;
    return p + r->offset + (uint64_t)cell * r->record_bytes;
  }
  return p + h->records_offset +
    ((uint64_t)rep * h->n_cells + cell) * h->record_bytes;
}

/* Observed-set encoding for a replicate (v2 predates it: always a list). */
static inline uint32_t msur_encoding_of(const void *base, const msur_header_t *h,
                                        uint32_t rep) {
  return msur_is_v3(h) ? msur_reps(base)[rep].flags : MSUR_ENC_LIST;
}

/* Start of the observed set within a record (it follows the feature block). */
static inline const unsigned char *msur_observed_set(const unsigned char *rec,
                                                     const msur_header_t *h) {
  return rec + (size_t)h->n_patterns * 8;
}

/* Is CpG `pos` one of this record's observed CpGs? `n_sel` is used only by the
 * list encoding, where the entries are sorted ascending. */
static inline int msur_observed(const unsigned char *obs, uint32_t enc,
                                uint32_t n_sel, uint32_t pos) {
  if (enc == MSUR_ENC_BITMAP) return (obs[pos >> 3] >> (pos & 7)) & 1u;
  const uint32_t *a = (const uint32_t *)obs;
  uint32_t lo = 0, hi = n_sel;
  while (lo < hi) { uint32_t m = lo + ((hi - lo) >> 1); if (a[m] < pos) lo = m + 1; else hi = m; }
  return lo < n_sel && a[lo] == pos;
}

/* Total bytes the records occupy, for the truncation check at open. */
static inline uint64_t msur_records_end(const void *base,
                                        const msur_header_t *h) {
  if (msur_is_v3(h)) {
    const msur_rep_t *r = msur_reps(base) + (h->n_reps - 1);
    return r->offset + (uint64_t)h->n_cells * r->record_bytes;
  }
  return h->records_offset +
    (uint64_t)h->n_cells * h->n_reps * h->record_bytes;
}

#endif /* MSUR_H */
