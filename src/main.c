// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * methscope-cli — pure-C analysis of sparse DNA methylomes via MRMP encoding.
 * Single multi-call binary; dispatches to subcommands (cf. yame's main.c).
 *
 * Copyright (C) 2025 Hongxiang Fu and Wanding Zhou
 * GNU Affero General Public License v3.0 or later.
 */
#include <stdio.h>
#include <string.h>
#include "methscope.h"
#include "yame_version.h"   /* YAME version this binary was built against */

static int usage(void) {
  fprintf(stderr,
    "\n"
    "methscope-cli (v%s) — ultra-fast sparse DNA methylome analysis via MRMP encoding\n"
    "Built against YAME %s\n"
    "\n"
    "Usage:\n"
    "  methscope <command> [options] [args]\n"
    "\n"
    "Commands:\n"
    "  predict      Predict a label (cell type, sex, ...): query.cg + model.ubjx -> labels + confidence\n"
    "  matrix       Build the record x pattern beta matrix (TSV; --refx -> a .refx reference)\n"
    "  deconv       Estimate cell-type proportions (NNLS): mixture.cg + panel.refx\n"
    "  upscale      Upscale an MRMP embedding to CpG-level methylation (MLP decoder)\n"
    "  train        Train a label classifier (xgboost/threshold/logistic) -> model.ubjx\n"
    "  inspect      Report a bundle's framework mark, on-disk layout, and model summary\n"
    "  bundle       Wrap a model + its MRMP (+ labels via -l) -> self-contained bundle\n"
    "  unbundle     Unpack a bundle back into its model, MRMP, and outcpg mask\n"
    "\n"
    "Run 'methscope <command> -h' for command-specific options.\n"
    "\n",
    METHSCOPE_VERSION, YAME_VERSION);
  return 1;
}

int main(int argc, char *argv[]) {
  if (argc < 2) return usage();
  if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) return usage();
  if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0) {
    printf("methscope %s (yame %s)\n", METHSCOPE_VERSION, YAME_VERSION);
    return 0;
  }
  if (strcmp(argv[1], "predict")    == 0) return main_predict(argc - 1, argv + 1);
  if (strcmp(argv[1], "matrix")     == 0) return main_matrix(argc - 1, argv + 1);
  if (strcmp(argv[1], "deconv")     == 0) return main_deconv(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale")    == 0) return main_upscale(argc - 1, argv + 1);
  if (strcmp(argv[1], "train")      == 0) return main_train(argc - 1, argv + 1);
  if (strcmp(argv[1], "inspect")    == 0) return main_inspect(argc - 1, argv + 1);
  if (strcmp(argv[1], "bundle")     == 0) return main_bundle(argc - 1, argv + 1);
  if (strcmp(argv[1], "unbundle")   == 0) return main_unbundle(argc - 1, argv + 1);

  fprintf(stderr, "[methscope] unrecognized command '%s'\n", argv[1]);
  return 1;
}
