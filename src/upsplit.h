// SPDX-License-Identifier: AGPL-3.0-or-later
#ifndef METHSCOPE_UPSPLIT_H
#define METHSCOPE_UPSPLIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Cell-split labels, matching the byte encoding UPFAC3 stores per cell. */
#define MS_UPSPLIT_TRAIN 0u
#define MS_UPSPLIT_VAL 1u
#define MS_UPSPLIT_TEST 2u

/* Read an explicit train/val/test assignment for the n_cells source cells of a
 * MSURAW2 sidecar and fill label[0..n_cells-1] with MS_UPSPLIT_*.  Any error is
 * fatal; `who` is the calling subcommand used in the message prefix. */
void ms_upsplit_load(const char *who, const char *path, uint32_t n_cells,
                     uint8_t *label);

/* Validate a split file against the cell count of a MSURAW2 sidecar, so a bad
 * split fails on the login node instead of after CUDA has been claimed. */
void ms_upsplit_check(const char *who, const char *msur_path, const char *path);

#ifdef __cplusplus
}
#endif
#endif
