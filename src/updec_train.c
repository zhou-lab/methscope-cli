// SPDX-License-Identifier: AGPL-3.0-or-later
/* `upscale-train`: pure-C training for BottleneckDecoder UPDEC1 models. */
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>
#include "methscope.h"
#include "updec.h"
#include "updec_cuda.h"
#include "updec_nn.h"

#define UPSCALE_N_IN 101
#define UPSCALE_N_OUT 10000

typedef struct {
  size_t n;
  float *x;                 /* n x UPSCALE_N_IN; transformed in place */
  uint8_t *target;          /* n x UPSCALE_N_OUT; 2 means missing */
} split_t;

static void tdie(const char *msg, const char *arg) {
  if (arg) fprintf(stderr, "[methscope] upscale-train: %s: %s\n", msg, arg);
  else     fprintf(stderr, "[methscope] upscale-train: %s\n", msg);
  exit(1);
}

static size_t xmul(size_t a, size_t b, const char *what) {
  if (a && b > SIZE_MAX / a) tdie("size overflow", what);
  return a * b;
}

static void *xmalloc(size_t n, const char *what) {
  void *p = malloc(n ? n : 1);
  if (!p) tdie("out of memory", what);
  return p;
}

static void *xcalloc(size_t n, size_t z, const char *what) {
  void *p = calloc(n ? n : 1, z);
  if (!p) tdie("out of memory", what);
  return p;
}

/* gzgets with dynamic growth: the real 10k target header is about 120 KiB. */
static int gz_getline(gzFile fp, char **buf, size_t *cap, const char *path) {
  if (!*buf) { *cap = 4096; *buf = xmalloc(*cap, "gzip line"); }
  size_t len = 0; (*buf)[0] = '\0';
  for (;;) {
    if (*cap - len < 2) {
      if (*cap > SIZE_MAX / 2) tdie("input line too long", path);
      *cap *= 2;
      char *q = realloc(*buf, *cap);
      if (!q) tdie("out of memory growing input line", path);
      *buf = q;
    }
    size_t room = *cap - len;
    int ask = room > INT_MAX ? INT_MAX : (int)room;
    char *got = gzgets(fp, *buf + len, ask);
    if (!got) {
      if (len) return 1;
      if (gzeof(fp)) return 0;
      int ze; const char *zs = gzerror(fp, &ze);
      tdie(zs ? zs : "gzip read error", path);
    }
    size_t add = strlen(*buf + len);
    len += add;
    if (len && ((*buf)[len-1] == '\n' || (*buf)[len-1] == '\r')) return 1;
    if (gzeof(fp)) return 1;
    if (!add) tdie("NUL byte in gzip TSV", path);
  }
}

static char *next_field(char **cursor) {
  if (!*cursor) return NULL;
  char *s = *cursor, *p = s;
  while (*p && *p != '\t' && *p != '\n' && *p != '\r') ++p;
  if (*p == '\t') { *p = '\0'; *cursor = p + 1; }
  else { *p = '\0'; *cursor = NULL; }
  return s;
}

static void validate_header(char *line, const char *path) {
  char *p = line, expect[64];
  for (int j = 0; j < UPSCALE_N_IN; ++j) {
    char *f = next_field(&p); snprintf(expect, sizeof(expect), "feat_%d", j + 1);
    if (!f || strcmp(f, expect)) tdie("expected header column", expect);
  }
  for (int k = 0; k < UPSCALE_N_OUT; ++k) {
    char *f = next_field(&p); snprintf(expect, sizeof(expect), "target_%d", k + 1);
    if (!f || strcmp(f, expect)) tdie("expected header column", expect);
  }
  if (next_field(&p)) tdie("header has extra columns", path);
}

static int blank_line(const char *s) {
  while (*s == '\n' || *s == '\r') ++s;
  return *s == '\0';
}

static float parse_feature(const char *s, const char *path) {
  if (!*s || !strcmp(s, "NA") || !strcmp(s, "nan") || !strcmp(s, "NaN")) return NAN;
  errno = 0; char *end = NULL; float v = strtof(s, &end);
  if (errno || end == s || *end || !isfinite(v)) tdie("invalid feature value", path);
  return v;
}

static uint8_t parse_target(const char *s, const char *path) {
  errno = 0; char *end = NULL; long v = strtol(s, &end, 10);
  if (errno || end == s || *end || v < 0 || v > 2) tdie("target must be 0, 1, or 2", path);
  return (uint8_t)v;
}

static void parse_data_row(char *line, float *x, uint8_t *target, const char *path) {
  char *p = line;
  for (int j = 0; j < UPSCALE_N_IN; ++j) {
    char *f = next_field(&p); if (!f) tdie("data row has too few columns", path);
    x[j] = parse_feature(f, path);
  }
  for (int k = 0; k < UPSCALE_N_OUT; ++k) {
    char *f = next_field(&p); if (!f) tdie("data row has too few columns", path);
    target[k] = parse_target(f, path);
  }
  if (next_field(&p)) tdie("data row has extra columns", path);
}

static split_t split_load(const char *path) {
  gzFile fp = gzopen(path, "rb");
  if (!fp) tdie("cannot open split", path);
  char *line = NULL; size_t cap = 0;
  if (!gz_getline(fp, &line, &cap, path)) tdie("empty split", path);
  validate_header(line, path);
  size_t n = 0;
  while (gz_getline(fp, &line, &cap, path)) if (!blank_line(line)) ++n;
  gzclose(fp);
  if (!n) tdie("split has no data rows", path);

  split_t s = {0}; s.n = n;
  s.x = xmalloc(xmul(xmul(n, UPSCALE_N_IN, "features"), sizeof(float), "features"), "features");
  s.target = xmalloc(xmul(n, UPSCALE_N_OUT, "targets"), "targets");
  fp = gzopen(path, "rb");
  if (!fp) tdie("cannot reopen split", path);
  if (!gz_getline(fp, &line, &cap, path)) tdie("empty split", path);
  size_t r = 0;
  while (gz_getline(fp, &line, &cap, path)) {
    if (blank_line(line)) continue;
    if (r >= n) tdie("split changed while reading", path);
    parse_data_row(line, s.x + r * UPSCALE_N_IN,
                   s.target + r * UPSCALE_N_OUT, path);
    ++r;
  }
  gzclose(fp); free(line);
  if (r != n) tdie("split changed while reading", path);
  fprintf(stderr, "[methscope] upscale-train: loaded %zu rows <- %s\n", n, path);
  return s;
}

static void split_free(split_t *s) {
  free(s->x); free(s->target); memset(s, 0, sizeof(*s));
}

static void transform_split(split_t *s, const double *imp,
                            const double *mean, const double *scale) {
  for (size_t r = 0; r < s->n; ++r)
    for (int j = 0; j < UPSCALE_N_IN; ++j) {
      size_t q = r * UPSCALE_N_IN + j;
      double v = s->x[q]; if (isnan(v)) v = imp[j];
      s->x[q] = (float)((v - mean[j]) / scale[j]);
    }
}

static void fit_preprocessing(split_t *train, split_t *val, split_t *test,
                              ms_updec_t *m) {
  double *imp = xcalloc(UPSCALE_N_IN, sizeof(*imp), "imputer means");
  double *mean = xcalloc(UPSCALE_N_IN, sizeof(*mean), "scaler means");
  double *scale = xcalloc(UPSCALE_N_IN, sizeof(*scale), "scaler scales");
  size_t count[UPSCALE_N_IN] = {0};
  for (size_t r = 0; r < train->n; ++r)
    for (int j = 0; j < UPSCALE_N_IN; ++j) {
      float v = train->x[r * UPSCALE_N_IN + j];
      if (!isnan(v)) { imp[j] += v; ++count[j]; }
    }
  for (int j = 0; j < UPSCALE_N_IN; ++j) {
    if (count[j]) imp[j] /= count[j];
    else fprintf(stderr, "[methscope] upscale-train: warning: feat_%d is all-NA in train; imputing 0\n", j + 1);
    double sum = 0.0;
    for (size_t r = 0; r < train->n; ++r) {
      double v = train->x[r * UPSCALE_N_IN + j]; if (isnan(v)) v = imp[j];
      sum += v;
    }
    mean[j] = sum / train->n;
    double ss = 0.0;
    for (size_t r = 0; r < train->n; ++r) {
      double v = train->x[r * UPSCALE_N_IN + j]; if (isnan(v)) v = imp[j];
      double d = v - mean[j]; ss += d * d;
    }
    scale[j] = sqrt(ss / train->n); if (scale[j] == 0.0) scale[j] = 1.0;
    m->imp_mean[j] = (float)imp[j]; m->sc_mean[j] = (float)mean[j];
    m->sc_scale[j] = (float)scale[j];
  }
  transform_split(train, imp, mean, scale);
  transform_split(val, imp, mean, scale);
  if (test) transform_split(test, imp, mean, scale);
  free(imp); free(mean); free(scale);
}

/* Fixed PCG32 stream: deterministic independent of libc rand(). */
typedef struct { uint64_t state, inc; } pcg_t;
static uint32_t pcg32(pcg_t *r) {
  uint64_t old = r->state; r->state = old * UINT64_C(6364136223846793005) + r->inc;
  uint32_t x = (uint32_t)(((old >> 18u) ^ old) >> 27u), rot = old >> 59u;
  return (x >> rot) | (x << ((-rot) & 31));
}
static void pcg_seed(pcg_t *r, uint64_t seed) {
  r->state = 0; r->inc = UINT64_C(1442695040888963407); pcg32(r);
  r->state += seed; pcg32(r);
}
static uint32_t pcg_bounded(pcg_t *r, uint32_t n) {
  uint32_t threshold = (uint32_t)(-n) % n;
  for (;;) { uint32_t x = pcg32(r); if (x >= threshold) return x % n; }
}
static float pcg_uniform(pcg_t *r, double bound) {
  double u = (double)pcg32(r) / 4294967296.0;
  return (float)((2.0 * u - 1.0) * bound);
}

static void init_model(ms_updec_t *m, pcg_t *rng) {
  size_t HI = (size_t)m->n_hidden * m->n_in, OH = (size_t)m->n_out * m->n_hidden;
  double b1 = 1.0 / sqrt(m->n_in), b2 = 1.0 / sqrt(m->n_hidden);
  for (size_t q = 0; q < HI; ++q) m->W1[q] = pcg_uniform(rng, b1);
  for (int q = 0; q < m->n_hidden; ++q) m->b1[q] = pcg_uniform(rng, b1);
  for (int q = 0; q < m->n_hidden; ++q) { m->bn_g[q] = 1.0f; m->bn_v[q] = 1.0f; }
  for (size_t q = 0; q < OH; ++q) m->W2[q] = pcg_uniform(rng, b2);
  for (int q = 0; q < m->n_out; ++q) m->b2[q] = pcg_uniform(rng, b2);
}

static void shuffle(size_t *idx, size_t n, pcg_t *rng) {
  for (size_t i = n; i > 1; --i) {
    if (i > UINT32_MAX) tdie("too many training rows for PRNG shuffle", NULL);
    size_t j = pcg_bounded(rng, (uint32_t)i), t = idx[i-1]; idx[i-1] = idx[j]; idx[j] = t;
  }
}

typedef struct { float *theta; double *g, *m, *v; size_t n; } param_t;
static param_t param_make(float *theta, size_t n, const char *what) {
  param_t p = {theta, xcalloc(n, sizeof(double), what),
               xcalloc(n, sizeof(double), what), xcalloc(n, sizeof(double), what), n};
  return p;
}
static void param_free(param_t *p) { free(p->g); free(p->m); free(p->v); }

typedef struct {
  int maxB, I, H, O;
  float *x; uint8_t *target;
  double *a1, *xhat, *bn, *logits, *invstd, *mean, *var, *dh;
} work_t;

static work_t work_make(int maxB, int I, int H, int O) {
  work_t w = {0}; w.maxB=maxB; w.I=I; w.H=H; w.O=O;
  size_t BI=xmul(maxB,I,"batch features"), BH=xmul(maxB,H,"batch hidden");
  size_t BO=xmul(maxB,O,"batch logits");
  w.x=xmalloc(xmul(BI,sizeof(float),"batch features"),"batch features");
  w.target=xmalloc(BO,"batch targets");
  w.a1=xmalloc(xmul(BH,sizeof(double),"a1"),"a1");
  w.xhat=xmalloc(xmul(BH,sizeof(double),"xhat"),"xhat");
  w.bn=xmalloc(xmul(BH,sizeof(double),"batchnorm"),"batchnorm");
  w.logits=xmalloc(xmul(BO,sizeof(double),"logits"),"logits");
  w.invstd=xmalloc(xmul(H,sizeof(double),"invstd"),"invstd");
  w.mean=xmalloc(xmul(H,sizeof(double),"bn mean"),"bn mean");
  w.var=xmalloc(xmul(H,sizeof(double),"bn var"),"bn var");
  w.dh=xmalloc(xmul(BH,sizeof(double),"hidden gradient"),"hidden gradient");
  return w;
}
static void work_free(work_t *w) {
  free(w->x); free(w->target); free(w->a1); free(w->xhat); free(w->bn);
  free(w->logits); free(w->invstd); free(w->mean); free(w->var); free(w->dh);
}

static void gather(work_t *w, const split_t *s, const size_t *idx, size_t pos, int B) {
  for (int b = 0; b < B; ++b) {
    size_t r = idx ? idx[pos + b] : pos + b;
    memcpy(w->x + (size_t)b*w->I, s->x + r*w->I, (size_t)w->I*sizeof(float));
    memcpy(w->target + (size_t)b*w->O, s->target + r*w->O, (size_t)w->O);
  }
}

static double forward_train(ms_updec_t *m, work_t *w, int B, double momentum) {
  ms_nn_linear_fwd_f32(w->a1,w->x,m->W1,m->b1,B,w->I,w->H);
  ms_nn_relu(w->a1,(size_t)B*w->H);
  ms_nn_bn_train(w->bn,w->xhat,w->invstd,w->mean,w->var,w->a1,
                 m->bn_g,m->bn_b,m->bn_m,m->bn_v,B,w->H,m->bn_eps,momentum);
  ms_nn_linear_fwd_f64(w->logits,w->bn,m->W2,m->b2,B,w->H,w->O);
  double loss=ms_nn_bce_logits(w->logits,w->target,w->logits,(size_t)B*w->O);
  return loss;
}

static double forward_eval(const ms_updec_t *m, work_t *w, int B) {
  ms_nn_linear_fwd_f32(w->a1,w->x,m->W1,m->b1,B,w->I,w->H);
  ms_nn_relu(w->a1,(size_t)B*w->H);
  ms_nn_bn_eval(w->bn,w->a1,m->bn_g,m->bn_b,m->bn_m,m->bn_v,B,w->H,m->bn_eps);
  ms_nn_linear_fwd_f64(w->logits,w->bn,m->W2,m->b2,B,w->H,w->O);
  return ms_nn_bce_logits(w->logits,w->target,NULL,(size_t)B*w->O);
}

static void copy_model(ms_updec_t *dst, const ms_updec_t *src) {
  size_t I=src->n_in,H=src->n_hidden,O=src->n_out;
  dst->bn_eps=src->bn_eps;
#define CPY(F,N) memcpy(dst->F,src->F,(N)*sizeof(float))
  CPY(imp_mean,I); CPY(sc_mean,I); CPY(sc_scale,I); CPY(W1,H*I); CPY(b1,H);
  CPY(bn_g,H); CPY(bn_b,H); CPY(bn_m,H); CPY(bn_v,H); CPY(W2,O*H); CPY(b2,O);
#undef CPY
}

static double validation_loss(const ms_updec_t *m, const split_t *val,
                              work_t *w, int batch) {
  double sum=0.0;
  for (size_t pos=0; pos<val->n; pos+=batch) {
    int B=(int)((val->n-pos<(size_t)batch)?val->n-pos:(size_t)batch);
    gather(w,val,NULL,pos,B); sum += forward_eval(m,w,B);
  }
  return sum;
}

static void evaluate_test(const ms_updec_t *m, const split_t *test,
                          work_t *w, int batch) {
  uint64_t valid=0, missing=0, correct=0; double abs_sum=0.0;
  for (size_t pos=0; pos<test->n; pos+=batch) {
    int B=(int)((test->n-pos<(size_t)batch)?test->n-pos:(size_t)batch);
    gather(w,test,NULL,pos,B); forward_eval(m,w,B);
    size_t n=(size_t)B*w->O;
    for (size_t q=0;q<n;++q) {
      uint8_t y=w->target[q]; if (y==2) { ++missing; continue; }
      double p=ms_nn_sigmoid(w->logits[q]); ++valid;
      correct += (uint8_t)(p>0.5)==y; abs_sum += fabs(p-y);
    }
  }
  fprintf(stderr,"[methscope] upscale-train: test accuracy %.6f  MAE %.6f  valid %llu  missing %llu\n",
          valid?(double)correct/valid:NAN,valid?abs_sum/valid:NAN,
          (unsigned long long)valid,(unsigned long long)missing);
}

static int train_usage(void) {
  fprintf(stderr,
    "\nUsage:\n"
    "  methscope upscale-train -o OUT.updec --train TRAIN.tsv.gz --val VAL.tsv.gz [options]\n\n"
    "Options:\n"
    "  --test FILE       evaluate the best model on a test split\n"
    "  --epochs N        training epochs (default 3)\n"
    "  --batch N         mini-batch size (default 64)\n"
    "  --lr X            Adam learning rate (default 1e-3)\n"
    "  --wd X            coupled L2 weight decay (default 1e-5)\n"
    "  --hidden N        bottleneck width (default 512)\n"
    "  --eps X           BatchNorm epsilon (default 1e-5)\n"
    "  --momentum X      BatchNorm running-stat momentum (default 0.1)\n"
    "  --seed N          deterministic PRNG seed (default 1)\n"
    "  --device DEVICE   cpu, cuda, or cuda:N (default cpu)\n"
    "  -h, --help        show this help\n\n"
    "Input columns must be feat_1..feat_101 then target_1..target_10000; target 2 is missing.\n");
  return 1;
}

static long parse_long(const char *s, const char *name) {
  errno=0; char *e=NULL; long v=strtol(s,&e,10);
  if(errno||e==s||*e)tdie("invalid integer option",name);
  return v;
}
static double parse_double(const char *s, const char *name) {
  errno=0; char *e=NULL; double v=strtod(s,&e);
  if(errno||e==s||*e||!isfinite(v))tdie("invalid numeric option",name);
  return v;
}
static uint64_t parse_seed(const char *s) {
  if (*s == '-') tdie("invalid seed",s);
  errno=0; char *e=NULL; unsigned long long v=strtoull(s,&e,10);
  if(errno||e==s||*e)tdie("invalid seed",s);
  return (uint64_t)v;
}
static int parse_int_option(const char *s, const char *name, int max) {
  long v=parse_long(s,name);
  if(v<1||v>max)tdie("integer option out of range",name);
  return (int)v;
}

static int parse_cuda_device(const char *s) {
  if(!strcmp(s,"cpu"))return -1;
  if(!strcmp(s,"cuda"))return 0;
  if(!strncmp(s,"cuda:",5)) {
    long v=parse_long(s+5,"--device");
    if(v<0||v>INT_MAX)tdie("CUDA device index out of range",s);
    return (int)v;
  }
  tdie("--device must be cpu, cuda, or cuda:N",s);
  return -1;
}

int main_upscale_train(int argc, char **argv) {
  const char *out=NULL,*train_path=NULL,*val_path=NULL,*test_path=NULL,*device="cpu";
  int epochs=3,batch=64,hidden=512; double lr=1e-3,wd=1e-5,eps=1e-5,momentum=0.1;
  uint64_t seed=1;
  for(int i=1;i<argc;++i) {
    if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help"))return train_usage();
#define OPT(S,V) if(!strcmp(argv[i],S)&&i+1<argc){V=argv[++i];continue;}
    OPT("-o",out) OPT("--train",train_path) OPT("--val",val_path) OPT("--test",test_path)
    OPT("--device",device)
#undef OPT
    if(!strcmp(argv[i],"--epochs")&&i+1<argc){epochs=parse_int_option(argv[++i],"--epochs",INT_MAX);continue;}
    if(!strcmp(argv[i],"--batch")&&i+1<argc){batch=parse_int_option(argv[++i],"--batch",INT_MAX-1);continue;}
    if(!strcmp(argv[i],"--hidden")&&i+1<argc){hidden=parse_int_option(argv[++i],"--hidden",INT_MAX);continue;}
    if(!strcmp(argv[i],"--lr")&&i+1<argc){lr=parse_double(argv[++i],"--lr");continue;}
    if(!strcmp(argv[i],"--wd")&&i+1<argc){wd=parse_double(argv[++i],"--wd");continue;}
    if(!strcmp(argv[i],"--eps")&&i+1<argc){eps=parse_double(argv[++i],"--eps");continue;}
    if(!strcmp(argv[i],"--momentum")&&i+1<argc){momentum=parse_double(argv[++i],"--momentum");continue;}
    if(!strcmp(argv[i],"--seed")&&i+1<argc){seed=parse_seed(argv[++i]);continue;}
    fprintf(stderr,"[methscope] upscale-train: unknown or incomplete option: %s\n",argv[i]);
    return train_usage();
  }
  if(!out||!train_path||!val_path)return train_usage();
  if(epochs<1)tdie("--epochs must be at least 1",NULL);
  if(batch<2)tdie("--batch must be at least 2",NULL);
  if(hidden<1)tdie("--hidden must be at least 1",NULL);
  if(!(lr>0))tdie("--lr must be positive",NULL);
  if(wd<0)tdie("--wd must be nonnegative",NULL);
  if(!(eps>0))tdie("--eps must be positive",NULL);
  if(momentum<0||momentum>1)tdie("--momentum must be in [0,1]",NULL);
  int cuda_device=parse_cuda_device(device);

  split_t train=split_load(train_path),val=split_load(val_path),test={0};
  if(test_path)test=split_load(test_path);
  if(train.n<2)tdie("training split needs at least two rows for BatchNorm",train_path);

  ms_updec_t *model=ms_updec_alloc(UPSCALE_N_IN,hidden,UPSCALE_N_OUT);
  ms_updec_t *best=ms_updec_alloc(UPSCALE_N_IN,hidden,UPSCALE_N_OUT);
  model->bn_eps=(float)eps; best->bn_eps=(float)eps;
  fit_preprocessing(&train,&val,test_path?&test:NULL,model);
  pcg_t rng; pcg_seed(&rng,seed); init_model(model,&rng);

  if(cuda_device>=0) {
    if(!ms_updec_cuda_available())
      tdie("CUDA backend unavailable (rebuild with make CUDA=1, or use --device cpu)",device);
    if(train.n>UINT32_MAX)tdie("CUDA backend supports at most UINT32_MAX training rows",train_path);
    size_t nperm=xmul((size_t)epochs,train.n,"CUDA epoch permutations");
    uint32_t *all_idx=xmalloc(xmul(nperm,sizeof(*all_idx),"CUDA epoch permutations"),
                              "CUDA epoch permutations");
    size_t *idx=xmalloc(xmul(train.n,sizeof(*idx),"shuffle indices"),"shuffle indices");
    for(size_t i=0;i<train.n;++i)idx[i]=i;
    for(int ep=0;ep<epochs;++ep) {
      shuffle(idx,train.n,&rng);
      for(size_t i=0;i<train.n;++i)all_idx[(size_t)ep*train.n+i]=(uint32_t)idx[i];
    }
    free(idx);
    ms_updec_cuda_config_t cfg={
      .train_x=train.x,.val_x=val.x,.test_x=test_path?test.x:NULL,
      .train_target=train.target,.val_target=val.target,
      .test_target=test_path?test.target:NULL,
      .n_train=train.n,.n_val=val.n,.n_test=test_path?test.n:0,
      .epoch_indices=all_idx,.epochs=epochs,.batch=batch,.device=cuda_device,
      .lr=lr,.weight_decay=wd,.bn_momentum=momentum
    };
    ms_updec_cuda_result_t result={0};
    if(ms_updec_train_cuda(&cfg,model,best,&result))
      tdie("CUDA training failed",device);
    if(test_path)
      fprintf(stderr,"[methscope] upscale-train: test accuracy %.6f  MAE %.6f  valid %llu  missing %llu\n",
              result.test_accuracy,result.test_mae,
              (unsigned long long)result.test_valid,
              (unsigned long long)result.test_missing);
    ms_updec_write(out,best);
    fprintf(stderr,"[methscope] upscale-train: wrote best model (val_loss %.9g) -> %s\n",
            result.best_val,out);
    free(all_idx);
    ms_updec_free(model);ms_updec_free(best);split_free(&train);split_free(&val);
    if(test_path)split_free(&test);
    return 0;
  }

  size_t HI=(size_t)hidden*UPSCALE_N_IN,OH=(size_t)UPSCALE_N_OUT*hidden;
  param_t ps[6]={param_make(model->W1,HI,"Adam W1"),param_make(model->b1,hidden,"Adam b1"),
    param_make(model->bn_g,hidden,"Adam gamma"),param_make(model->bn_b,hidden,"Adam beta"),
    param_make(model->W2,OH,"Adam W2"),param_make(model->b2,UPSCALE_N_OUT,"Adam b2")};
  size_t *idx=xmalloc(train.n*sizeof(*idx),"shuffle indices");
  for(size_t i=0;i<train.n;++i)idx[i]=i;
  size_t maxBn=train.n<(size_t)batch+1?train.n:(size_t)batch+1;
  size_t valBn=val.n<(size_t)batch?val.n:(size_t)batch;
  if(valBn>maxBn)maxBn=valBn;
  if(test_path) {
    size_t testBn=test.n<(size_t)batch?test.n:(size_t)batch;
    if(testBn>maxBn)maxBn=testBn;
  }
  int maxB=(int)maxBn;
  work_t work=work_make(maxB,UPSCALE_N_IN,hidden,UPSCALE_N_OUT);
  double best_val=INFINITY,sched_best=INFINITY; int bad=0; uint64_t step=0;

  for(int ep=0;ep<epochs;++ep) {
    shuffle(idx,train.n,&rng);
    for(size_t pos=0;pos<train.n;) {
      size_t remain=train.n-pos; int B=remain<(size_t)batch?(int)remain:batch;
      if(remain==(size_t)batch+1)B=batch+1;
      gather(&work,&train,idx,pos,B); forward_train(model,&work,B,momentum);
      ms_nn_linear_bwd_f64(work.logits,work.bn,model->W2,ps[4].g,ps[5].g,work.dh,B,hidden,UPSCALE_N_OUT);
      ms_nn_bn_bwd(work.dh,ps[2].g,ps[3].g,work.dh,work.xhat,work.invstd,model->bn_g,B,hidden);
      ms_nn_relu_bwd(work.dh,work.a1,(size_t)B*hidden);
      ms_nn_linear_bwd_f32(work.dh,work.x,model->W1,ps[0].g,ps[1].g,NULL,B,UPSCALE_N_IN,hidden);
      ++step; double ib1=1.0/(1.0-pow(0.9,(double)step));
      double ib2=1.0/(1.0-pow(0.999,(double)step));
      for(int p=0;p<6;++p)ms_nn_adam_step(ps[p].theta,ps[p].g,ps[p].m,ps[p].v,ps[p].n,lr,wd,ib1,ib2);
      pos+=B;
    }
    double vl=validation_loss(model,&val,&work,batch);
    if(!isfinite(vl))tdie("non-finite validation loss",NULL);
    ms_nn_plateau_step(vl,&sched_best,&bad,&lr);
    if(vl<best_val){best_val=vl;copy_model(best,model);}
    fprintf(stderr,"[methscope] upscale-train: epoch %d/%d  val_loss %.9g  lr %.9g\n",ep+1,epochs,vl,lr);
  }

  if(test_path)evaluate_test(best,&test,&work,batch);
  ms_updec_write(out,best);
  fprintf(stderr,"[methscope] upscale-train: wrote best model (val_loss %.9g) -> %s\n",best_val,out);

  for(int p=0;p<6;++p)param_free(&ps[p]);
  free(idx); work_free(&work);
  ms_updec_free(model); ms_updec_free(best); split_free(&train); split_free(&val);
  if(test_path)split_free(&test);
  return 0;
}
