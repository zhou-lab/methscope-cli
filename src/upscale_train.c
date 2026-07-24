// SPDX-License-Identifier: AGPL-3.0-or-later
/* Public trainer for the unified UPDEC2 processing-unit model. */
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "bundle.h"
#include "methscope.h"
#include "updec2.h"
#include "upunit_cuda.h"

static void terr(const char *msg, const char *arg) {
  fprintf(stderr, "[methscope] upscale-train: %s%s%s\n",
          msg, arg ? ": " : "", arg ? arg : "");
  exit(1);
}

static uint64_t u64(const char *s, const char *name) {
  errno = 0; char *e = NULL;
  unsigned long long x = strtoull(s, &e, 10);
  if (errno || e == s || *e || *s == '-') terr("invalid integer option", name);
  return (uint64_t)x;
}

static uint32_t u32(const char *s, const char *name) {
  uint64_t x = u64(s, name);
  if (x > UINT32_MAX) terr("integer option is too large", name);
  return (uint32_t)x;
}

static double real(const char *s, const char *name) {
  errno = 0; char *e = NULL; double x = strtod(s, &e);
  if (errno || e == s || *e) terr("invalid real option", name);
  return x;
}

static int usage(void) {
  fprintf(stderr,
    "Usage: methscope upscale-train -i DATA.msur --index UNITS.msui\n"
    "       --mrmp TOP1000.cm -o MODEL.updecx --work-dir DIR [options]\n\n"
    "Train whole-genome UPDEC2 processing units on CUDA. Each MRMP contributes\n"
    "beta plus log1p(observed-CpG count); count zero represents missingness.\n"
    "An optional frozen learned trunk is shared by every processing unit.\n\n"
    "Required:\n"
    "  -i, --data PATH          embedded-truth MSURAW2 training sidecar\n"
    "  --index PATH             whole-genome MSUIDX1 processing-unit index\n"
    "  --mrmp PATH              top-P MRMP .cm bundled into the output\n"
    "  -o PATH                  self-contained output .updecx\n"
    "  --work-dir DIR           resumable unit checkpoints and bare UPDEC2\n\n"
    "Architecture:\n"
    "  --patterns N             MRMP inputs (default 1000)\n"
    "  --features MODE          beta, count, or missing (default count)\n"
    "  --trunk PATH             optional frozen UPFAC3 shared trunk\n"
    "  --pure-bottleneck N      one-membership unit dimension (default 16)\n"
    "  --mixed-bottleneck N     mixed/PNA unit dimension (default 32)\n"
    "  --mixed-mode MODE        factor or direct (default factor)\n"
    "  --activation MODE        linear or leaky (default linear)\n"
    "  --adaptive-rank          size/variability-aware per-unit rank\n"
    "  --max-rank N             cap for large variable units (default 64)\n"
    "  --homogeneous-rank N     rank for flat all-0/all-1 units (default 8)\n"
    "  --large-cpgs N           >= this many CpGs -> max-rank (default 1000000)\n"
    "  --medium-cpgs N          >= this many CpGs -> 32 (default 100000)\n"
    "  --var-floor X            mean per-CpG variance below this = flat (0.005)\n\n"
    "Optimization:\n"
    "  --min-steps N            earliest stopping point (default 500)\n"
    "  --max-steps N            maximum updates per unit (default 2000)\n"
    "  --eval-every N           validation interval (default 100)\n"
    "  --patience N             non-improving checks (default 3)\n"
    "  --batch N                target CpGs per update (default 8192)\n"
    "  --eval-rows N            fixed validation rows per unit (default 8)\n"
    "  --learning-rate X        AdamW rate (default 0.001)\n"
    "  --weight-decay X         AdamW decay (default 0.00001)\n"
    "  --seed N                 deterministic seed (default 1)\n"
    "  --device N               CUDA device (default 0)\n"
    "  --pilot-units FILE       train listed unit IDs only; do not assemble\n"
    "  --force                  replace final output and manifest\n"
    "  --dry-run                validate options and print configuration\n"
    "  -h, --help               show this help\n");
  return 1;
}

static void ensure_dir(const char *path) {
  struct stat st;
  if (!stat(path, &st)) {
    if (!S_ISDIR(st.st_mode)) terr("--work-dir is not a directory", path);
    return;
  }
  if (errno != ENOENT || mkdir(path, 0775)) terr("cannot create --work-dir", path);
}

int main_upscale_train(int argc, char **argv) {
  const char *data = NULL, *index = NULL, *mrmp = NULL, *out = NULL, *work = NULL;
  int force = 0, dry = 0;
  ms_upunit_config_t c;
  memset(&c, 0, sizeof(c));
  c.patterns = 1000; c.pure_bottleneck = 16; c.mixed_bottleneck = 32;
  c.feature_mode = MS_UPFEATURE_COUNT;
  c.activation = MS_UPDEC2_LINEAR;
  c.min_steps = 500; c.max_steps = 2000; c.eval_every = 100;
  c.patience = 3; c.batch = 8192; c.eval_rows = 8;
  c.seed = 1; c.device = 0; c.learning_rate = 1e-3; c.weight_decay = 1e-5;
  c.adaptive_rank = 0; c.max_rank = 64; c.homogeneous_rank = 8;
  c.large_cpgs = 1000000; c.medium_cpgs = 100000; c.var_floor = 0.005;

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
    else if ((!strcmp(a, "-i") || !strcmp(a, "--data")) && i + 1 < argc) data = argv[++i];
    else if (!strcmp(a, "--index") && i + 1 < argc) index = argv[++i];
    else if (!strcmp(a, "--mrmp") && i + 1 < argc) mrmp = argv[++i];
    else if (!strcmp(a, "-o") && i + 1 < argc) out = argv[++i];
    else if (!strcmp(a, "--work-dir") && i + 1 < argc) work = argv[++i];
    else if (!strcmp(a, "--patterns") && i + 1 < argc) c.patterns = u32(argv[++i], a);
    else if (!strcmp(a, "--features") && i + 1 < argc) {
      const char *x = argv[++i];
      if (!strcmp(x, "count")) c.feature_mode = MS_UPFEATURE_COUNT;
      else if (!strcmp(x, "missing")) c.feature_mode = MS_UPFEATURE_MISSING;
      else if (!strcmp(x, "beta")) c.feature_mode = MS_UPFEATURE_BETA;
      else terr("--features must be beta, count, or missing", x);
    }
    else if (!strcmp(a, "--trunk") && i + 1 < argc) c.trunk_path = argv[++i];
    else if (!strcmp(a, "--pure-bottleneck") && i + 1 < argc) c.pure_bottleneck = u32(argv[++i], a);
    else if (!strcmp(a, "--mixed-bottleneck") && i + 1 < argc) c.mixed_bottleneck = u32(argv[++i], a);
    else if (!strcmp(a, "--mixed-mode") && i + 1 < argc) {
      const char *x = argv[++i];
      if (!strcmp(x, "factor")) c.mixed_direct = 0;
      else if (!strcmp(x, "direct")) c.mixed_direct = 1;
      else terr("--mixed-mode must be factor or direct", x);
    } else if (!strcmp(a, "--activation") && i + 1 < argc) {
      const char *x = argv[++i];
      if (!strcmp(x, "linear")) c.activation = MS_UPDEC2_LINEAR;
      else if (!strcmp(x, "leaky")) c.activation = MS_UPDEC2_LEAKY_RELU;
      else terr("--activation must be linear or leaky", x);
    } else if (!strcmp(a, "--adaptive-rank")) c.adaptive_rank = 1;
    else if (!strcmp(a, "--max-rank") && i + 1 < argc) c.max_rank = u32(argv[++i], a);
    else if (!strcmp(a, "--homogeneous-rank") && i + 1 < argc) c.homogeneous_rank = u32(argv[++i], a);
    else if (!strcmp(a, "--large-cpgs") && i + 1 < argc) c.large_cpgs = u32(argv[++i], a);
    else if (!strcmp(a, "--medium-cpgs") && i + 1 < argc) c.medium_cpgs = u32(argv[++i], a);
    else if (!strcmp(a, "--var-floor") && i + 1 < argc) c.var_floor = real(argv[++i], a);
    else if (!strcmp(a, "--min-steps") && i + 1 < argc) c.min_steps = u32(argv[++i], a);
    else if (!strcmp(a, "--max-steps") && i + 1 < argc) c.max_steps = u32(argv[++i], a);
    else if (!strcmp(a, "--eval-every") && i + 1 < argc) c.eval_every = u32(argv[++i], a);
    else if (!strcmp(a, "--patience") && i + 1 < argc) c.patience = u32(argv[++i], a);
    else if (!strcmp(a, "--batch") && i + 1 < argc) c.batch = u32(argv[++i], a);
    else if (!strcmp(a, "--eval-rows") && i + 1 < argc) c.eval_rows = u32(argv[++i], a);
    else if (!strcmp(a, "--learning-rate") && i + 1 < argc) c.learning_rate = real(argv[++i], a);
    else if (!strcmp(a, "--weight-decay") && i + 1 < argc) c.weight_decay = real(argv[++i], a);
    else if (!strcmp(a, "--seed") && i + 1 < argc) c.seed = u64(argv[++i], a);
    else if (!strcmp(a, "--device") && i + 1 < argc) c.device = (int)u32(argv[++i], a);
    else if (!strcmp(a, "--pilot-units") && i + 1 < argc) c.pilot_units_path = argv[++i];
    else if (!strcmp(a, "--force")) force = 1;
    else if (!strcmp(a, "--dry-run")) dry = 1;
    else { usage(); terr("unrecognized or incomplete option", a); }
  }
  if (!data || !index || !mrmp || !out || !work) return usage();
  if (!c.patterns || !c.pure_bottleneck || !c.mixed_bottleneck || !c.min_steps ||
      !c.max_steps || c.min_steps > c.max_steps || !c.eval_every ||
      !c.patience || !c.batch || !c.eval_rows ||
      !(c.learning_rate > 0) || c.weight_decay < 0)
    terr("invalid training configuration", NULL);
  if (c.adaptive_rank && (!c.max_rank || !c.homogeneous_rank ||
      c.medium_cpgs > c.large_cpgs || c.var_floor < 0))
    terr("invalid adaptive-rank configuration", NULL);
  if (!force && !access(out, F_OK)) terr("output exists (use --force)", out);

  char bare[4096], manifest[4096];
  if (snprintf(bare, sizeof(bare), "%s/model.updec2", work) >= (int)sizeof(bare) ||
      snprintf(manifest, sizeof(manifest), "%s.train.tsv", out) >= (int)sizeof(manifest))
    terr("output path is too long", NULL);
  fprintf(stderr,
    "[methscope] upscale-train: data=%s index=%s MRMP=%s\n"
    "[methscope] upscale-train: output=%s work=%s\n"
    "[methscope] upscale-train: P=%u features=%s input=%u trunk=%s pure=%u mixed=%s/%u activation=%s\n"
    "[methscope] upscale-train: steps=%u..%u eval=%u patience=%u batch=%u seed=%" PRIu64 "\n",
    data, index, mrmp, out, work, c.patterns,
    c.feature_mode == MS_UPFEATURE_COUNT ? "beta+count" :
      c.feature_mode == MS_UPFEATURE_MISSING ? "beta+missing" : "beta-only",
    (c.feature_mode == MS_UPFEATURE_BETA ? 1 : 2) * c.patterns,
    c.trunk_path ? c.trunk_path : "none",
    c.pure_bottleneck, c.mixed_direct ? "direct" : "factor",
    c.mixed_bottleneck, c.activation ? "leaky" : "linear",
    c.min_steps, c.max_steps, c.eval_every, c.patience, c.batch, c.seed);
  if (c.adaptive_rank)
    fprintf(stderr,
      "[methscope] upscale-train: adaptive rank: homogeneous=%u medium(>=%u)=32 "
      "large(>=%u)=max=%u var_floor=%.4g\n",
      c.homogeneous_rank, c.medium_cpgs, c.large_cpgs, c.max_rank, c.var_floor);
  if (dry) {
    fprintf(stderr, "[methscope] upscale-train: dry run complete\n");
    return 0;
  }
  ensure_dir(work);
  if (!ms_upunit_cuda_available())
    terr("CUDA backend unavailable; rebuild with make CUDA=1 on a GPU node", NULL);
  c.data_path = data; c.index_path = index; c.model_path = bare; c.work_dir = work;
  int train_rc = ms_upunit_train_cuda(&c);
  if (train_rc == 2) {
    fprintf(stderr, "[methscope] upscale-train: pilot complete; no bundle assembled\n");
    return 0;
  }
  if (train_rc) terr("CUDA training failed", NULL);

  /* Bundle assembly streams the model, so peak RAM is independent of model size. */
  ms_bundle_pack(out, "upscale", bare, mrmp, NULL);
  FILE *mf = fopen(manifest, "w");
  if (!mf) terr("cannot create training manifest", manifest);
  fprintf(mf,
    "format\tUPDEC2\nmodel\t%s\ndata\t%s\nindex\t%s\nmrmp\t%s\nwork_dir\t%s\n"
    "patterns\t%u\ninput_dim\t%u\nfeatures\t%s\ntrunk\t%s\npure_bottleneck\t%u\nmixed_mode\t%s\n"
    "mixed_bottleneck\t%u\nactivation\t%s\nmin_steps\t%u\nmax_steps\t%u\n"
    "eval_every\t%u\npatience\t%u\nbatch\t%u\neval_rows\t%u\nseed\t%" PRIu64
    "\nlearning_rate\t%.9g\nweight_decay\t%.9g\nadaptive_rank\t%u\n"
    "max_rank\t%u\nhomogeneous_rank\t%u\nlarge_cpgs\t%u\nmedium_cpgs\t%u\nvar_floor\t%.9g\n",
    out, data, index, mrmp, work, c.patterns,
    (c.feature_mode == MS_UPFEATURE_BETA ? 1 : 2) * c.patterns,
    c.feature_mode == MS_UPFEATURE_COUNT ? "beta_log1p_count" :
      c.feature_mode == MS_UPFEATURE_MISSING ? "beta_missing" : "beta_only",
    c.trunk_path ? c.trunk_path : "",
    c.pure_bottleneck, c.mixed_direct ? "direct" : "factor",
    c.mixed_bottleneck, c.activation ? "leaky" : "linear",
    c.min_steps, c.max_steps, c.eval_every, c.patience, c.batch,
    c.eval_rows, c.seed, c.learning_rate, c.weight_decay,
    c.adaptive_rank, c.max_rank, c.homogeneous_rank, c.large_cpgs,
    c.medium_cpgs, c.var_floor);
  if (fclose(mf)) terr("cannot close training manifest", manifest);
  fprintf(stderr, "[methscope] upscale-train: complete -> %s\n", out);
  return 0;
}
