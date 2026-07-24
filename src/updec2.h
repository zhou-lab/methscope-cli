// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPDEC2_H
#define METHSCOPE_UPDEC2_H

#include <stddef.h>
#include <stdint.h>

#define MS_UPDEC2_MAGIC "UPDEC2\0\0"
#define MS_UPDEC2_DIRECT 0u
#define MS_UPDEC2_FACTOR 1u
#define MS_UPDEC2_LINEAR 0u
#define MS_UPDEC2_LEAKY_RELU 1u
#define MS_UPDEC2_FLAG_GENOMIC 1u
#define MS_UPDEC2_FLAG_COUNT 2u
#define MS_UPDEC2_FLAG_TRUNK 4u
#define MS_UPDEC2_FLAG_BETA_ONLY 8u

#pragma pack(push,1)
typedef struct {
  char magic[8];
  uint32_t version, flags, patterns, input_dim;
  uint32_t n_units, n_memberships, target_unit_cpgs, activation;
  uint64_t n_cpg;
  uint64_t mean_offset, scale_offset, unit_offset, cpg_offset;
  uint64_t membership_offset, param_offset, file_bytes;
  uint64_t index_checksum, parameter_checksum, reserved0;
} ms_updec2_header_t;

typedef struct {
  uint64_t output_offset;
  uint64_t param_offset;
  uint64_t param_bytes;
  uint32_t cpg_count;
  uint32_t membership_count;
  uint16_t mode;
  uint16_t bottleneck_dim;
  uint32_t flags;
} ms_updec2_unit_t;

typedef struct {
  uint64_t pattern_key;
  uint64_t output_offset;
  uint32_t count;
  uint32_t unit;
} ms_updec2_membership_t;
#pragma pack(pop)

typedef struct {
  int fd;
  void *mapping;
  size_t mapping_bytes;
  uint64_t model_offset, model_bytes;
  const unsigned char *base;
  const ms_updec2_header_t *header;
  const float *mean, *scale;
  const ms_updec2_unit_t *units;
  const uint32_t *cpg;
  const ms_updec2_membership_t *memberships;
  uint32_t trunk_dim;
  const float *trunk_w1, *trunk_b1, *trunk_w2, *trunk_b2;
} ms_updec2_t;

/* Open either a bare UPDEC2 (`offset=0`) or a model section inside a bundle. */
int ms_updec2_open(ms_updec2_t *m, const char *path,
                   uint64_t offset, uint64_t length,
                   char *error, size_t error_cap);
void ms_updec2_close(ms_updec2_t *m);

/* Public input is P MRMP betas plus their observed-CpG counts. Version 2
 * models ignore count and retain the historical missing-indicator channel. */
void ms_updec2_prepare_input(const ms_updec2_t *m, const double *beta,
                             const int *count, float *x);

/* Evaluate every unit and scatter probabilities into genomic CpG order.
 * Workspace must hold max_bottleneck floats for v2/no-trunk models and
 * 2*trunk_dim+max_bottleneck floats for shared-trunk models. */
int ms_updec2_forward(const ms_updec2_t *m, const float *x, float *prob,
                      float *workspace, size_t workspace_cap);

#endif
