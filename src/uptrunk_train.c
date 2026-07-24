// SPDX-License-Identifier: AGPL-3.0-or-later
/* Hidden research trainer for a shared 512-dimensional UPDEC2 trunk.
 * The output is a UPFAC3 training artifact; upscale-train extracts and embeds
 * only its preprocessing and W1,b1,W2,b2 trunk parameters. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "upfactor_cuda.h"

static int usage(void) {
  fprintf(stderr,
    "Usage: methscope _upscale trunk-train -i DATA.msur -o TRUNK.upfac [options]\n\n"
    "  --features MODE   beta, missing, or count (default count)\n"
    "  --patterns N      MRMPs (default 1000)\n"
    "  --hidden N        shared width (default 512)\n"
    "  --rank N          training-head rank (default 16)\n"
    "  --steps N         updates (default 100000)\n"
    "  --batch N         CpG targets/update (default 65536)\n"
    "  --eval-batches N  validation/test batches (default 100)\n"
    "  --log-every N     validation interval (default 5000)\n"
    "  --learning-rate X (default 0.0003)\n"
    "  --weight-decay X  (default 0.00001)\n"
    "  --homogeneous-groups LIST\n"
    "  --homogeneous-fraction X (default 0.1)\n"
    "  --seed N          default 1\n"
    "  --device N        default 0\n");
  return 1;
}

static uint64_t u64(const char *s, const char *name) {
  errno = 0; char *e = NULL; unsigned long long x = strtoull(s, &e, 10);
  if (errno || e == s || *e) {
    fprintf(stderr, "[methscope] trunk-train: invalid %s: %s\n", name, s);
    exit(1);
  }
  return (uint64_t)x;
}

static double real(const char *s, const char *name) {
  errno = 0; char *e = NULL; double x = strtod(s, &e);
  if (errno || e == s || *e) {
    fprintf(stderr, "[methscope] trunk-train: invalid %s: %s\n", name, s);
    exit(1);
  }
  return x;
}

int main_upscale_trunk_train(int argc, char **argv) {
  ms_upfactor_config_t c = {0};
  c.patterns = 1000; c.rank = 16; c.hidden = 512; c.steps = 100000;
  c.batch = 65536; c.eval_batches = 100; c.log_every = 5000;
  c.seed = 1; c.learning_rate = 3e-4; c.weight_decay = 1e-5;
  c.homogeneous_fraction = 0.1; c.feature_mode = MS_UPFEATURE_COUNT;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) return usage();
    else if (!strcmp(a, "-i") && i + 1 < argc) c.data_path = argv[++i];
    else if (!strcmp(a, "-o") && i + 1 < argc) c.model_path = argv[++i];
    else if (!strcmp(a, "--features") && i + 1 < argc) {
      const char *v = argv[++i];
      if (!strcmp(v, "missing")) c.feature_mode = MS_UPFEATURE_MISSING;
      else if (!strcmp(v, "count")) c.feature_mode = MS_UPFEATURE_COUNT;
      else if (!strcmp(v, "beta")) c.feature_mode = MS_UPFEATURE_BETA;
      else { fprintf(stderr, "[methscope] trunk-train: --features must be beta, missing, or count\n"); return 1; }
    }
    else if (!strcmp(a, "--patterns") && i + 1 < argc) c.patterns = (uint32_t)u64(argv[++i], a);
    else if (!strcmp(a, "--rank") && i + 1 < argc) c.rank = (uint32_t)u64(argv[++i], a);
    else if (!strcmp(a, "--hidden") && i + 1 < argc) c.hidden = (uint32_t)u64(argv[++i], a);
    else if (!strcmp(a, "--steps") && i + 1 < argc) c.steps = (uint32_t)u64(argv[++i], a);
    else if (!strcmp(a, "--batch") && i + 1 < argc) c.batch = (uint32_t)u64(argv[++i], a);
    else if (!strcmp(a, "--eval-batches") && i + 1 < argc) c.eval_batches = (uint32_t)u64(argv[++i], a);
    else if (!strcmp(a, "--log-every") && i + 1 < argc) c.log_every = (uint32_t)u64(argv[++i], a);
    else if (!strcmp(a, "--learning-rate") && i + 1 < argc) c.learning_rate = real(argv[++i], a);
    else if (!strcmp(a, "--weight-decay") && i + 1 < argc) c.weight_decay = real(argv[++i], a);
    else if (!strcmp(a, "--homogeneous-groups") && i + 1 < argc) c.homogeneous_groups = argv[++i];
    else if (!strcmp(a, "--homogeneous-fraction") && i + 1 < argc) c.homogeneous_fraction = real(argv[++i], a);
    else if (!strcmp(a, "--seed") && i + 1 < argc) c.seed = u64(argv[++i], a);
    else if (!strcmp(a, "--device") && i + 1 < argc) c.device = (int)u64(argv[++i], a);
    else { usage(); fprintf(stderr, "[methscope] trunk-train: bad option: %s\n", a); return 1; }
  }
  if (!c.data_path || !c.model_path || !c.patterns || !c.rank || !c.hidden ||
      !c.steps || !c.batch || !c.eval_batches || !c.log_every ||
      c.homogeneous_fraction < 0 || c.homogeneous_fraction > 1)
    return usage();
  if (!ms_upfactor_cuda_available()) {
    fprintf(stderr, "[methscope] trunk-train: CUDA backend unavailable\n");
    return 1;
  }
  return ms_upfactor_train_cuda(&c);
}
