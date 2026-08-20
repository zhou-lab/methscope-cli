// SPDX-License-Identifier: AGPL-3.0-or-later
/**
 * methscope — pure-C analysis of sparse DNA methylomes via MRMP encoding.
 * Single multi-call binary; dispatches to subcommands (cf. yame's main.c).
 *
 * Copyright (C) 2025 Hongxiang Fu and Wanding Zhou
 * GNU Affero General Public License v3.0 or later.
 */
#include <stdio.h>
#include <stdlib.h>       /* getenv */
#include <string.h>
#include <unistd.h>       /* isatty */
#include "methscope.h"
#include "mrmp.h"
#include "msfm.h"
#include "assets.h"       /* yame_assets_root -- the shared store methscope reads */
#include "yame_version.h" /* YAME version this binary was built against */

/* Grouped, ANSI-styled overview. Colors are emitted only when stderr is a TTY,
 * so redirected/piped output stays plain (cf. the ls-alias foot-gun). */
static int usage(void) {
  int tty = isatty(STDERR_FILENO);
  const char *A = tty ? "\033[1;36m" : ""; /* accent: title + command names */
  const char *B = tty ? "\033[1m"    : ""; /* section headers */
  const char *D = tty ? "\033[2m"    : ""; /* dim: version, hints */
  const char *R = tty ? "\033[0m"    : "";
#define CMD(n, d) fprintf(stderr, "  %s%-17s%s %s\n", A, n, R, d)
  char store[4096];
  yame_assets_root(NULL, NULL, store, sizeof(store));
  const char *dh = getenv("YAME_DATA_HOME");
  fprintf(stderr, "\n%smethscope%s %sv%s%s\n", A, R, D, METHSCOPE_VERSION, R);
  fprintf(stderr, "%sDNA methylome analysis via MRMP encoding%s\n", D, R);
  fprintf(stderr, "%sbuilt against YAME %s%s\n", D, YAME_VERSION, R);
  fprintf(stderr, "%sYAME_DATA_HOME%s  %s %s%s%s\n", B, R, store, D,
          dh && *dh ? "(from $YAME_DATA_HOME)" : "(unset; -d overrides)", R);
  fprintf(stderr, "\n%sUsage%s  methscope <command> [options] [args]\n\n", D, R);

  fprintf(stderr, "%sModels & data%s %s— fetched with 'yame fetch methscope/...'%s\n",
          B, R, D, R);

  fprintf(stderr, "\n%sMRMP construction%s %s— the feature foundation%s\n", B, R, D, R);
  CMD("mrmp-build",   "Build the MRMP routing tree (--flat for one flat set)");
  CMD("mrmp-build-thin","One 2-class satellite per (thin class, nearest partner)");
  CMD("mrmp-build-neighbor","One 2-class satellite per (class, near neighbour)");
  CMD("mrmp-export",  "Emit the runtime .cm mask (and pattern / count tables)");
  CMD("mrmp-planes",  "Persist per-class q-filter bits so a subset rebuilds cheaply");
  CMD("mrmp-pool",    "Combine MRMP sets and cut them to a shared column budget");

  fprintf(stderr, "\n%sClassification%s %s(cell type, sex, ...)%s\n", B, R, D, R);
  CMD("classify",     "Classify a methylome -> labels + confidence");
  CMD("classify-train","Fit a label classifier (xgboost / threshold / logistic)");
  CMD("classify-featurize","Prebuild the .msfm feature matrix (parallel, reusable)");

  fprintf(stderr, "\n%sDeconvolution%s\n", B, R);
  CMD("deconv-build-ref","Pack a cell-type store into the .msdref deconvolution reference");
  CMD("deconv",           "Estimate cell-type proportions, rebuilding the MRMP per query");

  fprintf(stderr, "\n%sUpscaling%s %s(imputation)%s\n", B, R, D, R);
  CMD("upscale",      "Impute genome-wide CpG methylation from a sparse methylome");
  CMD("upscale-featurize", "Build the MSURAW2/3 training msur from a truth .cg");
  CMD("upscale-set-units", "Build the MSUIDX1 processing-unit index from a .mrmp");
  CMD("upscale-train","Train the whole-genome upscale decoder (CUDA)");

  fprintf(stderr, "\n%sModel bundles%s\n", B, R);
  CMD("bundle",       "Wrap a model + its MRMP into a self-contained bundle");
  CMD("relabel",      "Rename a class label in a trained model, no retraining");
  CMD("unbundle",     "Unpack a bundle into its model, MRMP, and outcpg mask");
  CMD("inspect",      "Describe any artifact: bundle, .mrmp, .msui, .msur, or .msfm");

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
  if (strcmp(argv[1], "classify")    == 0) return main_predict(argc - 1, argv + 1);
  if (strcmp(argv[1], "classify-featurize") == 0) return main_classify_featurize(argc - 1, argv + 1);
  if (strcmp(argv[1], "deconv-build-ref") == 0) return main_deconv_build_ref(argc - 1, argv + 1);
  if (strcmp(argv[1], "deconv") == 0) return main_deconv(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale")    == 0) return main_upscale(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale-train") == 0) return main_upscale_train(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale-featurize") == 0) return main_upscale_prepare(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale-set-units") == 0) return main_upscale_set_units(argc - 1, argv + 1);
  if (strcmp(argv[1], "_upscale") == 0) return main_upscale_internal(argc - 1, argv + 1);
  if (strcmp(argv[1], "upscale-factor-train") == 0 ||
      strcmp(argv[1], "upscale-residual-train") == 0) {
    fprintf(stderr, "[methscope] '%s' is retired; use the unified 'upscale-train'\n", argv[1]);
    return 1;
  }
  if (strcmp(argv[1], "upscale-prepare") == 0) {
    fprintf(stderr, "[methscope] 'upscale-prepare' was renamed to 'upscale-featurize'\n");
    return 1;
  }
  if (strcmp(argv[1], "upscale-residual-index") == 0) {
    fprintf(stderr, "[methscope] 'upscale-residual-index' was renamed to 'upscale-set-units'\n");
    return 1;
  }
  if (strcmp(argv[1], "upscale-hybrid-eval") == 0) {
    fprintf(stderr, "[methscope] 'upscale-hybrid-eval' was removed (deprecated hybrid model)\n");
    return 1;
  }
  if (strcmp(argv[1], "fetch") == 0) {   /* retired: YAME's registry now covers methscope data */
    fprintf(stderr, "[methscope] 'methscope fetch' was retired; use YAME's shared store:\n"
                    "  yame fetch methscope/hg38/models    # or hg38/data, mm10/models\n");
    return 1;
  }
  if (strcmp(argv[1], "mrmp-build")   == 0) return main_mrmp_build(argc - 1, argv + 1);
  if (strcmp(argv[1], "mrmp-build-thin") == 0) return main_mrmp_build_thin(argc - 1, argv + 1);
  if (strcmp(argv[1], "mrmp-build-neighbor") == 0) return main_mrmp_build_neighbor(argc - 1, argv + 1);
  if (strcmp(argv[1], "mrmp-tree")    == 0) return main_mrmp_build(argc - 1, argv + 1);  /* old name */
  if (strcmp(argv[1], "mrmp-export")  == 0) return main_mrmp_export(argc - 1, argv + 1);
  if (strcmp(argv[1], "mrmp-planes")  == 0) return main_mrmp_planes(argc - 1, argv + 1);
  if (strcmp(argv[1], "mrmp-pool")    == 0) return main_mrmp_pool(argc - 1, argv + 1);
  if (strcmp(argv[1], "classify-train-tree") == 0) return main_train_tree(argc - 1, argv + 1);
  if (strcmp(argv[1], "classify-train")      == 0) return main_train(argc - 1, argv + 1);
  if (strcmp(argv[1], "inspect")    == 0) return main_inspect(argc - 1, argv + 1);
  if (strcmp(argv[1], "bundle")     == 0) return main_bundle(argc - 1, argv + 1);
  if (strcmp(argv[1], "relabel")    == 0) return main_relabel(argc - 1, argv + 1);
  if (strcmp(argv[1], "unbundle")   == 0) return main_unbundle(argc - 1, argv + 1);

  fprintf(stderr, "[methscope] unrecognized command '%s'\n", argv[1]);
  return 1;
}
