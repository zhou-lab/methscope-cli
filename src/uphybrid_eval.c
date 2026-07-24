// SPDX-License-Identifier: AGPL-3.0-or-later
/* CLI wrapper for external-cohort evaluation of UPFAC3 + UPRES1. */
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "methscope.h"
#include "uphybrid_eval_cuda.h"

static int usage(void) {
  fprintf(stderr,
    "Usage: methscope _upscale eval -i DATA.msur --encoder BASE.upfac\n"
    "       --residual MODEL.upres -o METRICS.tsv [options]\n\n"
    "Evaluate a frozen UPFAC3 + UPRES1 hybrid on every finite truth value in\n"
    "an external MSURAW2 cohort. Observed input CpGs are included, matching\n"
    "Hao's 2018 Zhou continuous evaluation protocol.\n\n"
    "Options:\n"
    "  -i PATH          external MSURAW2 sidecar with embedded truth\n"
    "  --encoder PATH   trained UPFAC3 global model\n"
    "  --residual PATH  trained UPRES1 residual model\n"
    "  -o PATH          output pooled metrics TSV\n"
    "  --max-cells N    evaluate first N cells (default: all)\n"
    "  --max-reps N     evaluate first N replicates (default: all)\n"
    "  --log-every-cells N progress interval (default 10)\n"
    "  --device N       CUDA device (default 0)\n"
    "  -h, --help       show this help\n");
  return 1;
}

static uint64_t u64(const char *s, const char *what) {
  errno=0;char *e=NULL;unsigned long long v=strtoull(s,&e,10);
  if(errno||e==s||*e){fprintf(stderr,"[methscope] _upscale eval: invalid %s: %s\n",what,s);exit(1);}
  return (uint64_t)v;
}

int main_upscale_hybrid_eval(int argc,char **argv) {
  ms_uphybrid_eval_config_t c={0};c.log_every_cells=10;
  for(int i=1;i<argc;++i) {
    if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){usage();return 0;}
    else if(!strcmp(argv[i],"-i")&&i+1<argc)c.data_path=argv[++i];
    else if(!strcmp(argv[i],"--encoder")&&i+1<argc)c.encoder_path=argv[++i];
    else if(!strcmp(argv[i],"--residual")&&i+1<argc)c.residual_path=argv[++i];
    else if(!strcmp(argv[i],"-o")&&i+1<argc)c.metrics_path=argv[++i];
    else if(!strcmp(argv[i],"--max-cells")&&i+1<argc)c.max_cells=(uint32_t)u64(argv[++i],"--max-cells");
    else if(!strcmp(argv[i],"--max-reps")&&i+1<argc)c.max_reps=(uint32_t)u64(argv[++i],"--max-reps");
    else if(!strcmp(argv[i],"--log-every-cells")&&i+1<argc)c.log_every_cells=(uint32_t)u64(argv[++i],"--log-every-cells");
    else if(!strcmp(argv[i],"--device")&&i+1<argc)c.device=(int)u64(argv[++i],"--device");
    else {usage();fprintf(stderr,"[methscope] _upscale eval: bad option: %s\n",argv[i]);return 1;}
  }
  if(!c.data_path||!c.encoder_path||!c.residual_path||!c.metrics_path||!c.log_every_cells)return usage();
  if(!ms_uphybrid_eval_cuda_available()){
    fprintf(stderr,"[methscope] _upscale eval: native CUDA backend unavailable; rebuild with make CUDA=1\n");
    return 1;
  }
  return ms_uphybrid_eval_cuda(&c);
}
