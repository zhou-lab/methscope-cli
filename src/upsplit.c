// SPDX-License-Identifier: AGPL-3.0-or-later
/* Explicit train/val/test split of the source cells used by upscale training.
 *
 * The default split shuffles cells by seed and cuts 70/15/15, which can leave a
 * whole cell type out of training when the atlas has only two or three donors
 * per type.  A curated split file pins the assignment instead, so the same
 * held-out cells can be reused across models and compared with an external
 * baseline trained on that split.
 *
 * The msur identifies cells only by position, so a row keys on the 0-based
 * cell index -- the sample order of the truth .cg the msur was prepared
 * from.  Trailing columns are ignored, which is where the generator puts the
 * sample name so the file stays readable:
 *
 *   cell_index  split  sample_id                        <- optional header
 *   0           val    GSM5652176_Adipocytes-Z000000T7
 *   1           test   GSM5652177_Adipocytes-Z000000T9
 *
 * Every cell must appear exactly once; train and val must be non-empty. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "upsplit.h"

static void sdie(const char *who, const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] %s --split: %s: %s\n", who, msg, arg);
  else fprintf(stderr, "[methscope] %s --split: %s\n", who, msg);
  exit(1);
}

/* Trailing newline plus anything from the third column on. */
static void trim_row(char *line) {
  char *tab = strchr(line, '\t');
  if (tab) tab = strchr(tab + 1, '\t');
  if (tab) *tab = '\0';
  line[strcspn(line, "\r\n")] = '\0';
}

void ms_upsplit_load(const char *who, const char *path, uint32_t n_cells,
                     uint8_t *label) {
  FILE *fp = fopen(path, "r");
  if (!fp) sdie(who, "cannot open split file", path);
  uint8_t *seen = calloc(n_cells ? n_cells : 1, 1);
  if (!seen) sdie(who, "out of memory", NULL);

  char line[4096];
  uint32_t n_seen = 0, count[3] = {0, 0, 0};
  int first_row = 1;
  while (fgets(line, sizeof(line), fp)) {
    trim_row(line);
    if (!line[0] || line[0] == '#') continue;
    char *tab = strchr(line, '\t');
    if (!tab) sdie(who, "row is not <cell_index>TAB<train|val|test>", line);
    *tab = '\0';
    errno = 0;
    char *end = NULL;
    unsigned long long idx = strtoull(line, &end, 10);
    if (errno || end == line || *end) {
      /* One leading non-numeric row is the column header. */
      if (first_row) { first_row = 0; continue; }
      sdie(who, "cell index is not a non-negative integer", line);
    }
    first_row = 0;
    if (idx >= n_cells) sdie(who, "cell index is beyond the msur", line);
    const char *what = tab + 1;
    uint8_t lab;
    if (!strcmp(what, "train")) lab = MS_UPSPLIT_TRAIN;
    else if (!strcmp(what, "val")) lab = MS_UPSPLIT_VAL;
    else if (!strcmp(what, "test")) lab = MS_UPSPLIT_TEST;
    else sdie(who, "split must be train, val, or test", what);
    if (seen[idx]) sdie(who, "cell index is assigned twice", line);
    seen[idx] = 1;
    label[idx] = lab;
    count[lab]++;
    n_seen++;
  }
  if (ferror(fp)) sdie(who, "read failed", path);
  if (fclose(fp)) sdie(who, "close failed", path);

  if (n_seen != n_cells) {
    char msg[128];
    snprintf(msg, sizeof(msg), "expected one row per source cell (%u), got %u",
             n_cells, n_seen);
    sdie(who, msg, path);
  }
  free(seen);
    /* Both are required: training needs rows, and early stopping measures on
     * held-out cells. A run that folds val into train was measured and dropped
     * -- see the 20260822 org entry -- so an empty val is a mistake, not a mode. */
    if (!count[MS_UPSPLIT_TRAIN] || !count[MS_UPSPLIT_VAL])
      sdie(who, "train and val must both be non-empty", path);
  fprintf(stderr, "[methscope] %s: split %s train=%u val=%u test=%u\n",
          who, path, count[MS_UPSPLIT_TRAIN], count[MS_UPSPLIT_VAL],
          count[MS_UPSPLIT_TEST]);
}

void ms_upsplit_check(const char *who, const char *msur_path,
                      const char *path) {
  /* MSURAW2/3 header prefix (identical in both): magic[8], version, n_cells (see upscale_prepare.c). */
  struct { char magic[8]; uint32_t version, n_cells; } head;
  FILE *fp = fopen(msur_path, "rb");
  if (!fp) sdie(who, "cannot open training msur", msur_path);
  if (fread(&head, 1, sizeof(head), fp) != sizeof(head))
    sdie(who, "truncated training msur", msur_path);
  if (fclose(fp)) sdie(who, "close failed", msur_path);
  if (memcmp(head.magic, "MSURAW2\0", 8) && memcmp(head.magic, "MSURAW3\0", 8))
    sdie(who, "not a MSURAW2/3 training msur", msur_path);
  uint8_t *label = malloc(head.n_cells ? head.n_cells : 1);
  if (!label) sdie(who, "out of memory", NULL);
  ms_upsplit_load(who, path, head.n_cells, label);
  free(label);
}
