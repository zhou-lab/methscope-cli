// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * methscope — pure-C analysis of sparse DNA methylomes via MRMP encoding.
 * Single multi-call binary; dispatches to subcommands (cf. yame's main.c).
 *
 * Copyright (C) 2025 Hongxiang Fu and Wanding Zhou
 * GNU Affero General Public License v3.0 or later.
 */
#include <stdio.h>
#include <string.h>
#include <unistd.h>       /* isatty */
#include "methscope.h"
#include "yame_version.h" /* YAME version this binary was built against */

/* Grouped, ANSI-styled overview. Colors are emitted only when stderr is a TTY,
 * so redirected/piped output stays plain (cf. the ls-alias foot-gun). */
static int usage(void) {
  int tty = isatty(STDERR_FILENO);
  const char *A = tty ? "\033[1;36m" : ""; /* accent: title + command names */
  const char *B = tty ? "\033[1m"    : ""; /* section headers */
  const char *D = tty ? "\033[2m"    : ""; /* dim: version, hints */
  const char *R = tty ? "\033[0m"    : "";
#define CMD(n, d) fprintf(stderr, "  %s%-13s%s %s\n", A, n, R, d)
  fprintf(stderr, "\n%smethscope%s %sv%s · built against YAME %s%s\n",
          A, R, D, METHSCOPE_VERSION, YAME_VERSION, R);
  fprintf(stderr, "%spure-C analysis of sparse DNA methylomes via MRMP encoding%s\n\n",
          D, R);
  fprintf(stderr, "%sUsage%s  methscope <command> [options] [args]\n\n", D, R);

  fprintf(stderr, "%sReference%s %s— the MRMP foundation%s\n", B, R, D, R);
  CMD("mrmp",         "Build / inspect / export the MRMP membership-pattern mask");

  fprintf(stderr, "\n%sAnalyze a query .cg%s\n", B, R);
  CMD("predict",      "Classify a methylome (cell type, sex, ...) -> labels + confidence");
  CMD("deconv",       "Estimate cell-type proportions (NNLS) from a mixture");
  CMD("upscale",      "Impute genome-wide CpG methylation from a sparse methylome");
  CMD("matrix",       "Build the record x pattern beta matrix (or a .refx reference)");

  fprintf(stderr, "\n%sBuild models%s\n", B, R);
  CMD("train",        "Fit a label classifier (xgboost / threshold / logistic)");
  CMD("upscale-train","Train the whole-genome upscale decoder (CUDA)");
  CMD("bundle",       "Wrap a model + its MRMP into a self-contained bundle");
  CMD("unbundle",     "Unpack a bundle into its model, MRMP, and outcpg mask");

  fprintf(stderr, "\n%sInspect%s\n", B, R);
  CMD("inspect",      "Report a bundle's framework, layout, and model summary");

  fprintf(stderr, "\n%sRun 'methscope <command> -h' for command-specific options.%s\n\n",
          D, R);
#undef CMD
  return 1;
}

int main(int argc, char *argv[]) {
  if (argc < 2) return usage();
  if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
    usage();
    return 0;
  }
  if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0) {
    printf("methscope %s (yame %s)\n", METHSCOPE_VERSION, YAME_VERSION);
    return 0;
  }
  if (strcmp(argv[1], "predict")    == 0) return main_predict(argc - 1, argv + 1);
  if (strcmp(argv[1], "matrix")     == 0) return main_matrix(argc - 1, argv + 1);
  if (strcmp(argv[1], "deconv")     == 0) return main_deconv(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale")    == 0) return main_upscale(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale-train") == 0) return main_upscale_train(argc - 1, argv + 1);
  if (strcmp(argv[1], "_upscale") == 0) return main_upscale_internal(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale-factor-train") == 0 ||
      strcmp(argv[1], "upscale-residual-train") == 0) {
    fprintf(stderr, "[methscope] '%s' is retired; use the unified 'upscale-train'\n", argv[1]);
    return 1;
  }
  if (strcmp(argv[1], "upscale-prepare") == 0) {
    fprintf(stderr, "[methscope] internal tool moved to '_upscale prepare'\n");
    return 1;
  }
  if (strcmp(argv[1], "upscale-residual-index") == 0) {
    fprintf(stderr, "[methscope] internal tool moved to '_upscale index'\n");
    return 1;
  }
  if (strcmp(argv[1], "upscale-hybrid-eval") == 0) {
    fprintf(stderr, "[methscope] research evaluator moved to '_upscale eval'\n");
    return 1;
  }
  if (strcmp(argv[1], "mrmp")       == 0) return main_mrmp(argc - 1, argv + 1);
  if (strcmp(argv[1], "train")      == 0) return main_train(argc - 1, argv + 1);
  if (strcmp(argv[1], "inspect")    == 0) return main_inspect(argc - 1, argv + 1);
  if (strcmp(argv[1], "bundle")     == 0) return main_bundle(argc - 1, argv + 1);
  if (strcmp(argv[1], "unbundle")   == 0) return main_unbundle(argc - 1, argv + 1);

  fprintf(stderr, "[methscope] unrecognized command '%s'\n", argv[1]);
  return 1;
}
