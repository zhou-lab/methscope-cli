// SPDX-License-Identifier: AGPL-3.0-or-later
/* Native MRMP (methylation reference membership pattern) construction.
 *
 * `methscope mrmp build/inspect/export` replaces the YAME + awk + sort + text
 * pipeline that used to define the top-K MRMP mask. It reads a discretized
 * (pseudo-binary) reference .cg, reproduces YAME's `rowop -o binstring`
 * per-CpG resolution deterministically, counts exact membership patterns,
 * ranks them, and serializes a self-describing MRMPIDX1 artifact.
 *
 * A pattern is one fixed-length ternary string over the reference samples:
 *   '0' unmethylated, '1' methylated, '2' missing/ambiguous.
 * binstring emits '2' only as an all-or-nothing per-CpG sentinel, so the sole
 * pattern containing a '2' is the all-'2' string (PNA). Every other pattern is
 * pure {0,1}. Each pattern is packed as a base-3 uint64 key (3^35 < 2^64),
 * most-significant digit = sample 0. */
#ifndef MS_MRMP_H
#define MS_MRMP_H

#include <stdint.h>

#define MRMPIDX_MAGIC "MRMPIDX1"
#define MRMPIDX_VERSION 1u
#define MRMP_PNA_MEMBERSHIP 0xFFFFFFFFu   /* per-CpG sentinel: all-'2' / PNA */

#define MRMP_FLAG_INCLUDE_HOMOGENEOUS 1u  /* all-0 and all-1 are candidates */

/* 128-byte fixed header; all little-endian, offsets are absolute file bytes. */
typedef struct {
  char     magic[8];          /* "MRMPIDX1" */
  uint32_t version;           /* MRMPIDX_VERSION */
  uint32_t n_samples;         /* reference samples == pattern length */
  uint32_t n_selected;        /* K: patterns labelled P1..PK */
  uint32_t flags;             /* MRMP_FLAG_* */
  uint64_t n_cpg;             /* genomic CpGs (per-CpG membership entries) */
  uint64_t n_candidates;      /* distinct {0,1} patterns (ranked; excl. PNA) */
  uint64_t pna_key;           /* base-3 key of the all-'2' sentinel */
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
  uint64_t reserved;          /* pad to 128 bytes */
} mrmp_header_t;

/* One ranked candidate pattern (rank == array index; label = P(rank+1)). */
typedef struct {
  uint64_t key;    /* base-3 packed pattern */
  uint64_t count;  /* CpGs carrying this exact pattern */
} mrmp_pattern_t;

int main_mrmp(int argc, char *argv[]);

#endif /* MS_MRMP_H */
