// SPDX-License-Identifier: AGPL-3.0-or-later
/* Hidden developer/analysis entry points for the whole-genome upscale pipeline. */
#include <stdio.h>
#include <string.h>
#include "methscope.h"

static int internal_usage(void) {
  ms_help(stderr,
    "Usage: methscope _upscale <command> [options]\n\n"
    "Internal development tools; interfaces and file formats may change:\n"
    "  trunk-train Train a research shared 512-dimensional decoder trunk\n\n"
    "The pipeline steps are public: upscale-featurize, upscale-set-units.\n\n"
    "These commands are intentionally omitted from the public command list.\n");
  return 1;
}

int main_upscale_internal(int argc, char **argv) {
  if (argc < 2) return internal_usage();
  if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
    internal_usage();
    return 0;
  }
  if (!strcmp(argv[1], "trunk-train"))
    return main_upscale_trunk_train(argc - 1, argv + 1);
  fprintf(stderr, "[methscope] _upscale: unrecognized internal command '%s'\n",
          argv[1]);
  return 1;
}
