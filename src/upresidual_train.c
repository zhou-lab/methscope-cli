// SPDX-License-Identifier: AGPL-3.0-or-later
/* CLI wrapper for the native membership-first residual CUDA trainer. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "methscope.h"
#include "upresidual_cuda.h"

static int usage(void) {
  fprintf(stderr,
    "Usage: methscope upscale-residual-train -i DATA.msur --encoder BASE.upfac\\\n"
    "       --index RESIDUAL.msri -o MODEL.upres [options]\n\n"
    "Train membership-first residual decoder heads with the UPFAC3 shared\n"
    "encoder frozen. Training and evaluation use native CUDA/cuBLAS.\n\n"
    "Options:\n"
    "  -i PATH          MSURAW2 training sidecar (required)\n"
    "  --encoder PATH   trained UPFAC3 model providing frozen encoder (required)\n"
    "  --index PATH     MSRIDX1 residual membership/head index (required)\n"
    "  -o PATH          output UPRES1 residual model (required)\n"
    "  --rank N         local factors per decoder head (default 32)\n"
    "  --steps N        AdamW update steps (default 100000)\n"
    "  --batch N        CpG targets per update (default 8192)\n"
    "  --eval-rows-per-head N  validation/test rows per head (default 1)\n"
    "  --log-every N    validation interval (default 10000)\n"
    "  --learning-rate X (default 0.001)\n"
    "  --weight-decay X  (default 0.00001)\n"
    "  --seed N          deterministic training seed (default: encoder seed)\n"
    "  --device N        CUDA device (default 0)\n"
    "  -h, --help        show this help\n");
  return 1;
}

static uint64_t u64(const char *s, const char *what) {
  errno = 0; char *end = NULL; unsigned long long v = strtoull(s, &end, 10);
  if (errno || end == s || *end) {
    fprintf(stderr, "[methscope] upscale-residual-train: invalid %s: %s\n", what, s);
    exit(1);
  }
  return (uint64_t)v;
}

static double real(const char *s, const char *what) {
  errno = 0; char *end = NULL; double v = strtod(s, &end);
  if (errno || end == s || *end) {
    fprintf(stderr, "[methscope] upscale-residual-train: invalid %s: %s\n", what, s);
    exit(1);
  }
  return v;
}

int main_upscale_residual_train(int argc, char **argv) {
  ms_upresidual_config_t c = {0};
  c.rank = 32; c.steps = 100000; c.batch = 8192;
  c.eval_rows_per_head = 1; c.log_every = 10000;
  c.learning_rate = 1e-3; c.weight_decay = 1e-5;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) return usage();
    else if (!strcmp(argv[i], "-i") && i + 1 < argc) c.data_path = argv[++i];
    else if (!strcmp(argv[i], "--encoder") && i + 1 < argc) c.encoder_path = argv[++i];
    else if (!strcmp(argv[i], "--index") && i + 1 < argc) c.index_path = argv[++i];
    else if (!strcmp(argv[i], "-o") && i + 1 < argc) c.model_path = argv[++i];
    else if (!strcmp(argv[i], "--rank") && i + 1 < argc) c.rank = (uint32_t)u64(argv[++i], "--rank");
    else if (!strcmp(argv[i], "--steps") && i + 1 <argc) c.steps = (uint32_t)u64(argv[++i], "--steps");
    else if (!strcmp(argv[i], "--batch") && i + 1 < argc) c.batch = (uint32_t)u64(argv[++i], "--batch");
    else if (!strcmp(argv[i], "--eval-rows-per-head") && i + 1 < argc)
      c.eval_rows_per_head = (uint32_t)u64(argv[++i], "--eval-rows-per-head");
    else if (!strcmp(argv[i], "--log-every") && i + 1 < argc)
      c.log_every = (uint32_t)u64(argv[++i], "--log-every");
    else if (!strcmp(argv[i], "--learning-rate") && i + 1 < argc)
      c.learning_rate = real(argv[++i], "--learning-rate");
    else if (!strcmp(argv[i], "--weight-decay") && i + 1 < argc)
      c.weight_decay = real(argv[++i], "--weight-decay");
    else if (!strcmp(argv[i], "--seed") && i + 1 < argc) c.seed = u64(argv[++i], "--seed");
    else if (!strcmp(argv[i], "--device") && i + 1 < argc) c.device = (int)u64(argv[++i], "--device");
    else {
      usage();
      fprintf(stderr, "[methscope] upscale-residual-train: bad option: %s\n", argv[i]);
      return 1;
    }
  }
  if (!c.data_path || !c.encoder_path || !c.index_path || !c.model_path ||
      !c.rank || !c.steps || !c.batch || !c.eval_rows_per_head || !c.log_every)
    return usage();
  if (!ms_upresidual_cuda_available()) {
    fprintf(stderr, "[methscope] upscale-residual-train: native CUDA backend unavailable; rebuild with make CUDA=1\n");
    return 1;
  }
  return ms_upresidual_train_cuda(&c);
}
