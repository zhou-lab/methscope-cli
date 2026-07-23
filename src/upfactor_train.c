// SPDX-License-Identifier: AGPL-3.0-or-later
/* CLI wrapper for the native global MRMP-factorized CUDA trainer. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "methscope.h"
#include "upfactor_cuda.h"

static int usage(void) {
  fprintf(stderr,
    "Usage: methscope upscale-factor-train -i DATA.msur -o MODEL.upfac [options]\n\n"
    "Train the non-bin global factorized decoder with native CUDA/cuBLAS.\n"
    "The sidecar must be MSURAW2 and must include --embed-truth.\n\n"
    "Options:\n"
    "  -i PATH          MSURAW2 training sidecar (required)\n"
    "  -o PATH          output native factor model (required)\n"
    "  --patterns N     use P1..PN (default: all sidecar patterns)\n"
    "  --rank N         factors per MRMP (default 16)\n"
    "  --hidden N       shared encoder width (default 512)\n"
    "  --steps N        AdamW update steps (default 5000)\n"
    "  --batch N        CpG targets per update (default 65536)\n"
    "  --eval-batches N fixed batches per validation/test (default 100)\n"
    "  --log-every N    validation interval (default 250)\n"
    "  --learning-rate X (default 0.001)\n"
    "  --weight-decay X  (default 0.00001)\n"
    "  --homogeneous-groups LIST  comma-separated 1-based all-0/all-1 groups\n"
    "  --homogeneous-fraction X   fraction of target batch from those groups (default 0.1)\n"
    "  --seed N          deterministic split/training seed (default 1)\n"
    "  --device N        CUDA device (default 0)\n"
    "  -h, --help        show this help\n");
  return 1;
}

static uint64_t u64(const char *s, const char *what) {
  errno = 0; char *end = NULL; unsigned long long v = strtoull(s, &end, 10);
  if (errno || end == s || *end) { fprintf(stderr, "[methscope] upscale-factor-train: invalid %s: %s\n", what, s); exit(1); }
  return (uint64_t)v;
}
static double real(const char *s, const char *what) {
  errno = 0; char *end = NULL; double v = strtod(s, &end);
  if (errno || end == s || *end) { fprintf(stderr, "[methscope] upscale-factor-train: invalid %s: %s\n", what, s); exit(1); }
  return v;
}

int main_upscale_factor_train(int argc, char *argv[]) {
  ms_upfactor_config_t c = {0};
  c.rank=16; c.hidden=512; c.steps=5000; c.batch=65536;
  c.eval_batches=100; c.log_every=250; c.seed=1; c.learning_rate=1e-3;
  c.weight_decay=1e-5;
  c.homogeneous_fraction=0.1;
  for (int i=1;i<argc;++i) {
    if (!strcmp(argv[i],"-h") || !strcmp(argv[i],"--help")) return usage();
    else if (!strcmp(argv[i],"-i") && i+1<argc) c.data_path=argv[++i];
    else if (!strcmp(argv[i],"-o") && i+1<argc) c.model_path=argv[++i];
    else if (!strcmp(argv[i],"--patterns") && i+1<argc) c.patterns=(uint32_t)u64(argv[++i],"--patterns");
    else if (!strcmp(argv[i],"--rank") && i+1<argc) c.rank=(uint32_t)u64(argv[++i],"--rank");
    else if (!strcmp(argv[i],"--hidden") && i+1<argc) c.hidden=(uint32_t)u64(argv[++i],"--hidden");
    else if (!strcmp(argv[i],"--steps") && i+1<argc) c.steps=(uint32_t)u64(argv[++i],"--steps");
    else if (!strcmp(argv[i],"--batch") && i+1<argc) c.batch=(uint32_t)u64(argv[++i],"--batch");
    else if (!strcmp(argv[i],"--eval-batches") && i+1<argc) c.eval_batches=(uint32_t)u64(argv[++i],"--eval-batches");
    else if (!strcmp(argv[i],"--log-every") && i+1<argc) c.log_every=(uint32_t)u64(argv[++i],"--log-every");
    else if (!strcmp(argv[i],"--seed") && i+1<argc) c.seed=u64(argv[++i],"--seed");
    else if (!strcmp(argv[i],"--device") && i+1<argc) c.device=(int)u64(argv[++i],"--device");
    else if (!strcmp(argv[i],"--learning-rate") && i+1<argc) c.learning_rate=real(argv[++i],"--learning-rate");
    else if (!strcmp(argv[i],"--weight-decay") && i+1<argc) c.weight_decay=real(argv[++i],"--weight-decay");
    else if (!strcmp(argv[i],"--homogeneous-groups") && i+1<argc) c.homogeneous_groups=argv[++i];
    else if (!strcmp(argv[i],"--homogeneous-fraction") && i+1<argc) c.homogeneous_fraction=real(argv[++i],"--homogeneous-fraction");
    else { usage(); fprintf(stderr,"[methscope] upscale-factor-train: bad option: %s\n",argv[i]); return 1; }
  }
  if (!c.data_path || !c.model_path || !c.rank || !c.hidden || !c.steps || !c.batch || !c.eval_batches) return usage();
  if (c.homogeneous_fraction < 0.0 || c.homogeneous_fraction > 1.0) {
    fprintf(stderr,"[methscope] upscale-factor-train: --homogeneous-fraction must be in [0,1]\n");
    return 1;
  }
  if (!ms_upfactor_cuda_available()) {
    fprintf(stderr,"[methscope] upscale-factor-train: native CUDA backend unavailable; rebuild with make CUDA=1\n");
    return 1;
  }
  return ms_upfactor_train_cuda(&c);
}
