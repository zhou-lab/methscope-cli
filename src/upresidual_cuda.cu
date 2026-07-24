// SPDX-License-Identifier: AGPL-3.0-or-later
/*
 * Native CUDA/cuBLAS trainer for membership-first residual upscale heads.
 *
 * The trained UPFAC3 encoder is frozen and evaluated once for every simulated
 * input row:
 *
 *   h = frozen_UPFAC3_encoder(x)
 *   z_b = LeakyReLU(A_b h + a_b)       (one local rank-R projection per head)
 *   y_c = sigmoid(e_c . z_b + bias_c)  (linear CpG decoder)
 *
 * Heads are defined by MSRIDX1.  Each update touches one head, one input row,
 * and a unique target subset, so the large CpG embedding is updated sparsely.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#include "upresidual_cuda.h"

static const int THREADS = 256;

static void die(const char *msg) {
  std::fprintf(stderr, "[methscope] upscale-train/residual: %s\n", msg);
  std::exit(1);
}
static void die_path(const char *msg, const char *path) {
  std::fprintf(stderr, "[methscope] upscale-train/residual: %s: %s\n", msg, path);
  std::exit(1);
}
static void cuda_fail(const char *call, cudaError_t e, const char *file, int line) {
  std::fprintf(stderr, "[methscope] upscale-train/residual CUDA: %s failed at %s:%d: %s\n",
               call, file, line, cudaGetErrorString(e));
  std::exit(1);
}
#define CUDA_OK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess) cuda_fail(#x,_e,__FILE__,__LINE__); } while(0)

static const char *blas_msg(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
    default: return "unknown";
  }
}
static void blas_fail(const char *call, cublasStatus_t s, const char *file, int line) {
  std::fprintf(stderr, "[methscope] upscale-train/residual cuBLAS: %s failed at %s:%d: %s\n",
               call, file, line, blas_msg(s));
  std::exit(1);
}
#define BLAS_OK(x) do { cublasStatus_t _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS) blas_fail(#x,_s,__FILE__,__LINE__); } while(0)

static size_t checked_mul(size_t a, size_t b, const char *what) {
  if (a && b > SIZE_MAX / a) die(what);
  return a * b;
}
static float *df(size_t n) {
  float *p = NULL;
  CUDA_OK(cudaMalloc((void **)&p, checked_mul(n, sizeof(float), "device size overflow")));
  return p;
}
static uint32_t *du32(size_t n) {
  uint32_t *p = NULL;
  CUDA_OK(cudaMalloc((void **)&p, checked_mul(n, sizeof(uint32_t), "device size overflow")));
  return p;
}
static uint16_t *du16(size_t n) {
  uint16_t *p = NULL;
  CUDA_OK(cudaMalloc((void **)&p, checked_mul(n, sizeof(uint16_t), "device size overflow")));
  return p;
}
static int blocks(size_t n) {
  size_t b = (n + THREADS - 1) / THREADS;
  return (int)(b > 4096 ? 4096 : (b ? b : 1));
}

struct Pcg { uint64_t state, inc; };
static uint32_t pcg32(Pcg *p) {
  uint64_t o=p->state; p->state=o*UINT64_C(6364136223846793005)+p->inc;
  uint32_t x=(uint32_t)(((o>>18)^o)>>27),r=(uint32_t)(o>>59);
  return (x>>r)|(x<<((-r)&31));
}
static void pcg_seed(Pcg *p, uint64_t s) {
  p->state=0; p->inc=UINT64_C(1442695040888963407);
  pcg32(p); p->state+=s; pcg32(p);
}

struct MsurHeader {
  char magic[8]; uint32_t version,n_cells,n_reps,n_patterns; uint64_t n_cpg;
  uint32_t sampled_per_cell,flags;
  uint64_t groups_offset,truth_offset,records_offset,record_bytes;
};
static_assert(sizeof(MsurHeader)==72, "MSUR header layout");

#pragma pack(push,1)
struct UpfacHeader {
  char magic[8];
  uint32_t version,patterns,rank,hidden,n_input,n_active,n_cells,n_train,n_val,n_test,steps,batch;
  uint64_t n_cpg,seed;
  float val_rmse,val_mae,test_rmse,test_mae;
  uint64_t split_offset,prep_offset,active_offset,param_offset,file_bytes;
};
static_assert(sizeof(UpfacHeader)==128, "UPFAC3 header layout");

struct MsriHeader {
  char magic[8];
  uint32_t version,flags,pattern_length,top_patterns;
  uint32_t target_bin_cpgs,n_bins,n_memberships,reserved32;
  uint64_t n_cpg,n_residual;
  uint64_t bin_offsets_offset,cpg_offset,rank_offset,group_offset,file_bytes;
  uint64_t pattern_checksum,top_checksum,reserved0,reserved1;
};
static_assert(sizeof(MsriHeader)==128, "MSRIDX1 header layout");

struct UpresHeader {
  char magic[8];
  uint32_t version,rank,hidden,n_bins,n_memberships,n_cells;
  uint32_t n_train,n_val,n_test,steps,batch,reserved32;
  uint64_t n_cpg,n_residual,seed;
  float val_rmse,val_mae,val_macro_rmse,val_macro_mae;
  float test_rmse,test_mae,test_macro_rmse,test_macro_mae;
  uint64_t split_offset,bin_offsets_offset,cpg_offset,membership_rank_offset;
  uint64_t param_offset,file_bytes,encoder_checksum,index_checksum;
};
static_assert(sizeof(UpresHeader)==176, "UPRES1 header layout");
#pragma pack(pop)

struct Mapping { int fd; size_t size; uint8_t *base; };
static Mapping map_read(const char *path) {
  Mapping m={-1,0,NULL};
  m.fd=open(path,O_RDONLY);
  if(m.fd<0) die_path("cannot open input",path);
  struct stat st;
  if(fstat(m.fd,&st)||st.st_size<0) die_path("cannot stat input",path);
  m.size=(size_t)st.st_size;
  m.base=(uint8_t*)mmap(NULL,m.size,PROT_READ,MAP_SHARED,m.fd,0);
  if(m.base==MAP_FAILED) die_path("cannot mmap input",path);
  return m;
}
static void unmap_read(Mapping *m) {
  if(m->base&&m->base!=MAP_FAILED) munmap(m->base,m->size);
  if(m->fd>=0) close(m->fd);
}
static bool range_ok(uint64_t off, uint64_t bytes, size_t size) {
  return off <= size && bytes <= (uint64_t)size - off;
}

__device__ static float sigmoid(float x) {
  if(x>=0){float z=expf(-x);return 1.0f/(1.0f+z);}
  float z=expf(x);return z/(1.0f+z);
}
__device__ static uint64_t hash64(uint64_t x) {
  x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
  x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}
__global__ static void init_random(float *x,size_t n,float scale,uint64_t seed) {
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x) {
    uint64_t z=hash64(seed+q*UINT64_C(0x9e3779b97f4a7c15));
    float u=(float)((z>>40)+0.5)*(1.0f/16777216.0f);
    x[q]=(2.0f*u-1.0f)*scale;
  }
}
__global__ static void matrix_leaky_bias(float *x,const float *b,size_t n,int width) {
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x) {
    float v=x[q]+b[q%width]; x[q]=v>0?v:.01f*v;
  }
}
__global__ static void matrix_leaky_residual(float *x,const float *b,const float *skip,size_t n,int width) {
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x) {
    float v=x[q]+b[q%width]; x[q]=skip[q]+(v>0?v:.01f*v);
  }
}
__global__ static void add_leaky_bias(float *x,const float *b,int n) {
  for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x) {
    float v=x[q]+b[q]; x[q]=v>0?v:.01f*v;
  }
}
__global__ static void leaky_back(float *d,const float *z,int n) {
  for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x)
    if(!(z[q]>0)) d[q]*=.01f;
}
__global__ static void accumulate_bias(const uint16_t *truth,const uint32_t *cpg,
                                       float *sum,uint32_t *count,size_t n) {
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x) {
    uint16_t v=truth[cpg[q]];
    if(v!=UINT16_MAX){sum[q]+=(float)v/65534.0f;count[q]++;}
  }
}
__global__ static void finalize_bias(const float *sum,const uint32_t *count,float *bias,size_t n) {
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x) {
    float p=count[q]?sum[q]/count[q]:.5f;
    p=fmaxf(.01f,fminf(.99f,p)); bias[q]=logf(p/(1.0f-p));
  }
}
__global__ static void adam_vec(float *t,float *m,float *v,const float *g,size_t n,
                                float lr,float wd,float ib1,float ib2) {
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x) {
    float z=g[q]+wd*t[q],mm=.9f*m[q]+.1f*z,vv=.999f*v[q]+.001f*z*z;
    m[q]=mm;v[q]=vv;t[q]-=lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);
  }
}
__global__ static void adam_outer(float *t,float *m,float *v,const float *go,const float *in,
                                  int O,int I,float lr,float wd,float ib1,float ib2) {
  size_t n=(size_t)O*I;
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x) {
    int o=(int)(q/I),j=(int)(q%I);float z=go[o]*in[j]+wd*t[q];
    float mm=.9f*m[q]+.1f*z,vv=.999f*v[q]+.001f*z*z;
    m[q]=mm;v[q]=vv;t[q]-=lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);
  }
}

struct Metrics { double sse,sae; unsigned long long n; };
__global__ static void predict_metrics(const float *z,const float *e,const float *eb,
                                       const uint32_t *id,const float *y,int B,int R,Metrics *out) {
  double sse=0,sae=0;unsigned long long n=0;
  for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<B;q+=blockDim.x*gridDim.x) {
    uint32_t c=id[q];float s=eb[c];
    for(int r=0;r<R;++r)s+=e[(size_t)c*R+r]*z[r];
    float d=sigmoid(s)-y[q];sse+=(double)d*d;sae+=fabs((double)d);++n;
  }
  __shared__ double ss[THREADS],sa[THREADS];
  __shared__ unsigned long long sn[THREADS];
  ss[threadIdx.x]=sse;sa[threadIdx.x]=sae;sn[threadIdx.x]=n;__syncthreads();
  for(int d=blockDim.x/2;d;d>>=1) {
    if(threadIdx.x<d){ss[threadIdx.x]+=ss[threadIdx.x+d];sa[threadIdx.x]+=sa[threadIdx.x+d];sn[threadIdx.x]+=sn[threadIdx.x+d];}
    __syncthreads();
  }
  if(threadIdx.x==0){atomicAdd(&out->sse,ss[0]);atomicAdd(&out->sae,sa[0]);atomicAdd(&out->n,sn[0]);}
}
__global__ static void decoder_backward(const float *z,float *e,float *em,float *ev,
                                        float *eb,float *ebm,float *ebv,const uint32_t *id,
                                        const float *y,float *dz,int B,int R,float lr,float wd,
                                        float ib1,float ib2) {
  for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<B;q+=blockDim.x*gridDim.x) {
    uint32_t c=id[q];float s=eb[c];
    for(int r=0;r<R;++r)s+=e[(size_t)c*R+r]*z[r];
    float p=sigmoid(s),dl=2.0f*(p-y[q])*p*(1.0f-p)/B;
    for(int r=0;r<R;++r) {
      size_t eq=(size_t)c*R+r;float old=e[eq];atomicAdd(dz+r,dl*old);
      float gr=dl*z[r]+wd*old,mm=.9f*em[eq]+.1f*gr,vv=.999f*ev[eq]+.001f*gr*gr;
      em[eq]=mm;ev[eq]=vv;e[eq]=old-lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);
    }
    float old=eb[c],gr=dl,mm=.9f*ebm[c]+.1f*gr,vv=.999f*ebv[c]+.001f*gr*gr;
    ebm[c]=mm;ebv[c]=vv;eb[c]=old-lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);
  }
}

struct Param { float *t,*m,*v;size_t n; };
static Param param_random(size_t n,float scale,uint64_t seed) {
  Param p={df(n),df(n),df(n),n};
  init_random<<<blocks(n),THREADS>>>(p.t,n,scale,seed);
  CUDA_OK(cudaMemset(p.m,0,n*sizeof(float)));CUDA_OK(cudaMemset(p.v,0,n*sizeof(float)));
  return p;
}
static Param param_zero(size_t n) {
  Param p={df(n),df(n),df(n),n};
  CUDA_OK(cudaMemset(p.t,0,n*sizeof(float)));CUDA_OK(cudaMemset(p.m,0,n*sizeof(float)));CUDA_OK(cudaMemset(p.v,0,n*sizeof(float)));
  return p;
}
static Param param_take(float *initial,size_t n) {
  Param p={initial,df(n),df(n),n};
  CUDA_OK(cudaMemset(p.m,0,n*sizeof(float)));CUDA_OK(cudaMemset(p.v,0,n*sizeof(float)));
  return p;
}
static void free_param(Param *p) {
  cudaFree(p->t);cudaFree(p->m);cudaFree(p->v);p->t=p->m=p->v=NULL;
}

struct Net { Param a,ab,e,eb; int H,R,Bins; };
struct Work { float *z,*dz,*y;uint32_t *id;Metrics *met; };

static void gemv(cublasHandle_t h,const float *w,const float *x,float *y,int O,int I) {
  const float one=1,zero=0;
  BLAS_OK(cublasSgemv(h,CUBLAS_OP_T,I,O,&one,w,I,x,1,&zero,y,1));
}
static void local_forward(cublasHandle_t h,const Net &n,Work &w,const float *repr,uint32_t bin) {
  gemv(h,n.a.t+(size_t)bin*n.R*n.H,repr,w.z,n.R,n.H);
  add_leaky_bias<<<blocks(n.R),THREADS>>>(w.z,n.ab.t+(size_t)bin*n.R,n.R);
}

static bool selected_has(const uint32_t *p,uint32_t n,uint32_t x) {
  return std::binary_search(p,p+n,x);
}
static uint32_t fill_batch(const MsurHeader *h,const uint8_t *base,const uint64_t *bo,
                           const uint32_t *cpg,uint32_t bin,uint32_t cell,uint32_t rep,
                           uint64_t start,uint32_t want,std::vector<uint32_t> &id,
                           std::vector<float> &y) {
  uint64_t begin=bo[bin],end=bo[bin+1],available=end-begin;
  if(!available||!want)return 0;
  size_t row=(size_t)rep*h->n_cells+cell;
  const uint8_t *rec=base+h->records_offset+row*h->record_bytes;
  const uint32_t *sel=(const uint32_t*)(rec+(size_t)h->n_patterns*(sizeof(float)+sizeof(uint32_t)));
  const uint16_t *truth=(const uint16_t*)(base+h->truth_offset)+(size_t)cell*h->n_cpg;
  uint64_t scanned=0,q=start%available;uint32_t n=0;
  while(n<want&&scanned<available) {
    uint32_t local=(uint32_t)(begin+q),pos=cpg[local];uint16_t v=truth[pos];
    if(v!=UINT16_MAX&&!selected_has(sel,h->sampled_per_cell,pos)) {
      id[n]=local;y[n]=(float)v/65534.0f;++n;
    }
    if(++q==available)q=0;++scanned;
  }
  return n;
}

struct Score { double rmse,mae,macro_rmse,macro_mae;uint64_t n;uint32_t heads; };
static Score evaluate(cublasHandle_t bh,const Net &net,Work &w,const float *repr,
                      const MsurHeader *h,const uint8_t *base,const uint64_t *bo,
                      const uint32_t *cpg,const std::vector<uint32_t> &cells,
                      uint32_t rows_per_head,uint32_t batch,uint64_t seed) {
  Pcg rng;pcg_seed(&rng,seed);Metrics total={0,0,0};
  double macro_rmse=0,macro_mae=0;uint32_t macro_n=0;
  std::vector<uint32_t> id(batch);std::vector<float> y(batch);
  for(uint32_t bin=0;bin<(uint32_t)net.Bins;++bin) {
    Metrics bm={0,0,0};
    for(uint32_t k=0;k<rows_per_head;++k) {
      uint32_t cell=cells[pcg32(&rng)%cells.size()],rep=pcg32(&rng)%h->n_reps;
      uint32_t want=(uint32_t)std::min<uint64_t>(batch,bo[bin+1]-bo[bin]);
      uint32_t B=fill_batch(h,base,bo,cpg,bin,cell,rep,pcg32(&rng),want,id,y);
      if(!B)continue;
      size_t row=(size_t)rep*h->n_cells+cell;
      CUDA_OK(cudaMemcpy(w.id,id.data(),B*sizeof(uint32_t),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(w.y,y.data(),B*sizeof(float),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemset(w.met,0,sizeof(Metrics)));
      local_forward(bh,net,w,repr+row*net.H,bin);
      predict_metrics<<<blocks(B),THREADS>>>(w.z,net.e.t,net.eb.t,w.id,w.y,B,net.R,w.met);
      Metrics m;CUDA_OK(cudaMemcpy(&m,w.met,sizeof(m),cudaMemcpyDeviceToHost));
      bm.sse+=m.sse;bm.sae+=m.sae;bm.n+=m.n;
    }
    if(bm.n) {
      total.sse+=bm.sse;total.sae+=bm.sae;total.n+=bm.n;
      macro_rmse+=std::sqrt(bm.sse/bm.n);macro_mae+=bm.sae/bm.n;++macro_n;
    }
  }
  Score s={NAN,NAN,NAN,NAN,total.n,macro_n};
  if(total.n){s.rmse=std::sqrt(total.sse/total.n);s.mae=total.sae/total.n;}
  if(macro_n){s.macro_rmse=macro_rmse/macro_n;s.macro_mae=macro_mae/macro_n;}
  return s;
}

static double now() {
  struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;
}
static uint64_t fnv_update(uint64_t h,const void *vp,size_t n) {
  const uint8_t *p=(const uint8_t*)vp;
  for(size_t i=0;i<n;++i){h^=p[i];h*=UINT64_C(1099511628211);}
  return h;
}
static void write_all(FILE *f,const void *p,size_t n,const char *path) {
  if(n&&fwrite(p,1,n,f)!=n)die_path("model write failed",path);
}
static void write_device(FILE *f,const float *d,size_t n,const char *path) {
  const size_t chunk=UINT64_C(16)*1024*1024;
  std::vector<float>b(std::min(n,chunk));size_t q=0;
  while(q<n){size_t k=std::min(n-q,chunk);CUDA_OK(cudaMemcpy(b.data(),d+q,k*sizeof(float),cudaMemcpyDeviceToHost));write_all(f,b.data(),k*sizeof(float),path);q+=k;}
}

extern "C" int ms_upresidual_cuda_available(void) {
  int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;
}

extern "C" int ms_upresidual_train_cuda(const ms_upresidual_config_t *cfg) {
  Mapping data=map_read(cfg->data_path),enc=map_read(cfg->encoder_path),idx=map_read(cfg->index_path);
  if(data.size<sizeof(MsurHeader)||enc.size<sizeof(UpfacHeader)||idx.size<sizeof(MsriHeader))
    die("truncated input header");
  const MsurHeader *dh=(const MsurHeader*)data.base;
  const UpfacHeader *eh=(const UpfacHeader*)enc.base;
  const MsriHeader *ih=(const MsriHeader*)idx.base;
  if(std::memcmp(dh->magic,"MSURAW2\0",8)||dh->version!=2)die("training input is not MSURAW2");
  if(!(dh->flags&1)||!dh->truth_offset)die("sidecar lacks embedded truth");
  if(std::memcmp(eh->magic,"UPFAC3\0\0",8)||eh->version!=3)die("encoder is not UPFAC3");
  if(std::memcmp(ih->magic,"MSRIDX1",7)||ih->version!=1)die("index is not MSRIDX1");
  if(dh->n_cpg!=eh->n_cpg||dh->n_cpg!=ih->n_cpg||dh->n_cells!=eh->n_cells)
    die("sidecar, encoder, and residual index dimensions disagree");
  if(eh->patterns>dh->n_patterns||eh->n_input!=3*eh->patterns)die("encoder feature dimensions disagree with sidecar");
  uint64_t records_end=dh->records_offset+(uint64_t)dh->n_cells*dh->n_reps*dh->record_bytes;
  if(records_end>data.size)die("truncated sidecar records");
  uint64_t truth_bytes=(uint64_t)dh->n_cells*dh->n_cpg*sizeof(uint16_t);
  if(!range_ok(dh->truth_offset,truth_bytes,data.size))die("truncated embedded truth");
  if(eh->file_bytes>enc.size||!range_ok(eh->split_offset,eh->n_cells,enc.size)||
     !range_ok(eh->prep_offset,(uint64_t)3*eh->n_input*sizeof(float),enc.size))
    die("truncated UPFAC3 metadata");
  uint64_t frozen_floats=(uint64_t)eh->hidden*eh->n_input+eh->hidden+
                         (uint64_t)eh->hidden*eh->hidden+eh->hidden;
  if(!range_ok(eh->param_offset,frozen_floats*sizeof(float),enc.size))
    die("truncated UPFAC3 encoder parameters");
  if(ih->file_bytes>idx.size||
     !range_ok(ih->bin_offsets_offset,(uint64_t)(ih->n_bins+1)*sizeof(uint64_t),idx.size)||
     !range_ok(ih->cpg_offset,ih->n_residual*sizeof(uint32_t),idx.size)||
     !range_ok(ih->rank_offset,ih->n_residual*sizeof(uint32_t),idx.size))
    die("truncated MSRIDX1 payload");
  if(!ih->n_bins||!ih->n_residual||ih->n_residual>UINT32_MAX)die("empty or oversized residual index");

  const uint64_t *bo=(const uint64_t*)(idx.base+ih->bin_offsets_offset);
  const uint32_t *cpg=(const uint32_t*)(idx.base+ih->cpg_offset);
  const uint32_t *membership=(const uint32_t*)(idx.base+ih->rank_offset);
  if(bo[0]!=0||bo[ih->n_bins]!=ih->n_residual)die("invalid residual head offsets");
  for(uint32_t b=0;b<ih->n_bins;++b)if(bo[b]>=bo[b+1])die("empty or nonmonotone residual head");
  for(uint64_t q=0;q<ih->n_residual;++q)if(cpg[q]>=ih->n_cpg)die("residual CpG index out of range");

  const uint8_t *split=enc.base+eh->split_offset;
  std::vector<uint32_t> train,val,test;
  for(uint32_t c=0;c<eh->n_cells;++c) {
    if(split[c]==0)train.push_back(c);else if(split[c]==1)val.push_back(c);
    else if(split[c]==2)test.push_back(c);else die("invalid UPFAC3 cell split");
  }
  if(train.size()!=eh->n_train||val.size()!=eh->n_val||test.size()!=eh->n_test)
    die("UPFAC3 split counts disagree with header");
  uint64_t seed=cfg->seed?cfg->seed:eh->seed;
  int I=(int)eh->n_input,H=(int)eh->hidden,R=(int)cfg->rank;
  size_t rows=(size_t)dh->n_cells*dh->n_reps;
  std::fprintf(stderr,
    "[methscope] upscale-train/residual: data cells=%u reps=%u CpGs=%llu residual=%llu memberships=%u heads=%u\n"
    "[methscope] upscale-train/residual: frozen encoder input=%d hidden=%d; local rank=%d; split=%zu/%zu/%zu\n",
    dh->n_cells,dh->n_reps,(unsigned long long)dh->n_cpg,
    (unsigned long long)ih->n_residual,ih->n_memberships,ih->n_bins,
    I,H,R,train.size(),val.size(),test.size());

  CUDA_OK(cudaSetDevice(cfg->device));
  cudaDeviceProp prop;CUDA_OK(cudaGetDeviceProperties(&prop,cfg->device));
  std::fprintf(stderr,"[methscope] upscale-train/residual: CUDA device %d: %s (%.0f MiB, cc %d.%d)\n",
               cfg->device,prop.name,(double)prop.totalGlobalMem/(1024*1024),prop.major,prop.minor);
  cublasHandle_t bh;BLAS_OK(cublasCreate(&bh));

  double prep0=now();
  std::vector<float> feat(checked_mul(rows,(size_t)I,"feature matrix overflow"));
  const float *prep=(const float*)(enc.base+eh->prep_offset);
  const float *imp=prep,*mean=prep+I,*scale=prep+2*I;
  for(size_t row=0;row<rows;++row) {
    const uint8_t *rec=data.base+dh->records_offset+row*dh->record_bytes;
    const float *beta=(const float*)rec;
    const uint32_t *count=(const uint32_t*)(rec+(size_t)dh->n_patterns*sizeof(float));
    float *out=feat.data()+row*I;
    for(uint32_t g=0;g<eh->patterns;++g) {
      float raw[3]={beta[g],std::log1p((float)count[g]),std::isnan(beta[g])?1.0f:0.0f};
      for(int k=0;k<3;++k){int j=3*g+k;float v=std::isnan(raw[k])?imp[j]:raw[k];out[j]=(v-mean[j])/scale[j];}
    }
  }

  float *d_feat=df(feat.size()),*d_h1=df(rows*H),*d_repr=df(rows*H);
  float *d_w1=df((size_t)H*I),*d_b1=df(H),*d_w2=df((size_t)H*H),*d_b2=df(H);
  const float *ep=(const float*)(enc.base+eh->param_offset);
  CUDA_OK(cudaMemcpy(d_feat,feat.data(),feat.size()*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_w1,ep,(size_t)H*I*sizeof(float),cudaMemcpyHostToDevice));ep+=(size_t)H*I;
  CUDA_OK(cudaMemcpy(d_b1,ep,H*sizeof(float),cudaMemcpyHostToDevice));ep+=H;
  CUDA_OK(cudaMemcpy(d_w2,ep,(size_t)H*H*sizeof(float),cudaMemcpyHostToDevice));ep+=(size_t)H*H;
  CUDA_OK(cudaMemcpy(d_b2,ep,H*sizeof(float),cudaMemcpyHostToDevice));
  const float one=1,zero=0;
  BLAS_OK(cublasSgemm(bh,CUBLAS_OP_T,CUBLAS_OP_N,H,(int)rows,I,&one,d_w1,I,d_feat,I,&zero,d_h1,H));
  matrix_leaky_bias<<<blocks(rows*H),THREADS>>>(d_h1,d_b1,rows*H,H);
  BLAS_OK(cublasSgemm(bh,CUBLAS_OP_T,CUBLAS_OP_N,H,(int)rows,H,&one,d_w2,H,d_h1,H,&zero,d_repr,H));
  matrix_leaky_residual<<<blocks(rows*H),THREADS>>>(d_repr,d_b2,d_h1,rows*H,H);
  CUDA_OK(cudaDeviceSynchronize());
  feat.clear();feat.shrink_to_fit();cudaFree(d_feat);cudaFree(d_h1);cudaFree(d_w1);cudaFree(d_b1);cudaFree(d_w2);cudaFree(d_b2);

  uint32_t *d_cpg=du32(ih->n_residual),*d_bcnt=du32(ih->n_residual);
  float *d_bsum=df(ih->n_residual),*d_bias=df(ih->n_residual);
  uint16_t *d_truth=du16(ih->n_cpg);
  CUDA_OK(cudaMemcpy(d_cpg,cpg,ih->n_residual*sizeof(uint32_t),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(d_bsum,0,ih->n_residual*sizeof(float)));
  CUDA_OK(cudaMemset(d_bcnt,0,ih->n_residual*sizeof(uint32_t)));
  const uint16_t *all_truth=(const uint16_t*)(data.base+dh->truth_offset);
  for(uint32_t cell:train) {
    CUDA_OK(cudaMemcpy(d_truth,all_truth+(size_t)cell*ih->n_cpg,ih->n_cpg*sizeof(uint16_t),cudaMemcpyHostToDevice));
    accumulate_bias<<<blocks(ih->n_residual),THREADS>>>(d_truth,d_cpg,d_bsum,d_bcnt,ih->n_residual);
  }
  finalize_bias<<<blocks(ih->n_residual),THREADS>>>(d_bsum,d_bcnt,d_bias,ih->n_residual);
  cudaFree(d_truth);cudaFree(d_bsum);cudaFree(d_bcnt);cudaFree(d_cpg);

  size_t an=(size_t)ih->n_bins*R*H,abn=(size_t)ih->n_bins*R,en=(size_t)ih->n_residual*R;
  float ax=std::sqrt(6.0f/(H+R));
  Net net={param_random(an,ax,seed^UINT64_C(0xa0761d6478bd642f)),
           param_zero(abn),
           param_random(en,.02f,seed^UINT64_C(0xe7037ed1a0b428db)),
           param_take(d_bias,ih->n_residual),H,R,(int)ih->n_bins};
  Work w={df(R),df(R),df(cfg->batch),du32(cfg->batch),NULL};
  CUDA_OK(cudaMalloc((void**)&w.met,sizeof(Metrics)));
  CUDA_OK(cudaDeviceSynchronize());
  double prep_seconds=now()-prep0;
  std::fprintf(stderr,
    "[methscope] upscale-train/residual: prepared %zu frozen representations and CpG biases in %.1fs; trainable parameters=%.3f billion\n",
    rows,prep_seconds,(double)(an+abn+en+ih->n_residual)/1e9);

  std::vector<uint32_t> bid(cfg->batch),head_order(ih->n_bins),head_steps(ih->n_bins,0);
  std::vector<float> by(cfg->batch);
  for(uint32_t b=0;b<ih->n_bins;++b)head_order[b]=b;
  Pcg rng;pcg_seed(&rng,seed^UINT64_C(0x8ebc6af09c88c6e3));
  Score vs=evaluate(bh,net,w,d_repr,dh,data.base,bo,cpg,val,cfg->eval_rows_per_head,cfg->batch,seed+101);
  std::fprintf(stderr,
    "[methscope] upscale-train/residual: step 0 val_rmse=%.6f val_mae=%.6f macro_rmse=%.6f macro_mae=%.6f n=%llu heads=%u\n",
    vs.rmse,vs.mae,vs.macro_rmse,vs.macro_mae,(unsigned long long)vs.n,vs.heads);
  double t0=now();
  for(uint32_t step=1;step<=cfg->steps;++step) {
    uint32_t cycle=(step-1)%ih->n_bins;
    if(!cycle)for(uint32_t q=ih->n_bins;q>1;--q)std::swap(head_order[q-1],head_order[pcg32(&rng)%q]);
    uint32_t bin=head_order[cycle],cell=train[pcg32(&rng)%train.size()],rep=pcg32(&rng)%dh->n_reps;
    uint32_t want=(uint32_t)std::min<uint64_t>(cfg->batch,bo[bin+1]-bo[bin]);
    uint32_t B=fill_batch(dh,data.base,bo,cpg,bin,cell,rep,pcg32(&rng),want,bid,by);
    if(!B)die("could not form residual training batch");
    size_t row=(size_t)rep*dh->n_cells+cell;
    CUDA_OK(cudaMemcpy(w.id,bid.data(),B*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(w.y,by.data(),B*sizeof(float),cudaMemcpyHostToDevice));
    local_forward(bh,net,w,d_repr+row*H,bin);
    CUDA_OK(cudaMemset(w.dz,0,R*sizeof(float)));
    uint32_t bt=++head_steps[bin];
    float ib1=1.0f/(1.0f-std::pow(.9f,(float)bt)),ib2=1.0f/(1.0f-std::pow(.999f,(float)bt));
    float lr=(float)cfg->learning_rate,wd=(float)cfg->weight_decay;
    decoder_backward<<<blocks(B),THREADS>>>(w.z,net.e.t,net.e.m,net.e.v,net.eb.t,net.eb.m,net.eb.v,
                                            w.id,w.y,w.dz,B,R,lr,wd,ib1,ib2);
    leaky_back<<<blocks(R),THREADS>>>(w.dz,w.z,R);
    size_t ao=(size_t)bin*R*H,abo=(size_t)bin*R;
    adam_outer<<<blocks((size_t)R*H),THREADS>>>(net.a.t+ao,net.a.m+ao,net.a.v+ao,w.dz,d_repr+row*H,R,H,lr,wd,ib1,ib2);
    adam_vec<<<blocks(R),THREADS>>>(net.ab.t+abo,net.ab.m+abo,net.ab.v+abo,w.dz,R,lr,0,ib1,ib2);
    if(step%cfg->log_every==0||step==cfg->steps) {
      vs=evaluate(bh,net,w,d_repr,dh,data.base,bo,cpg,val,cfg->eval_rows_per_head,cfg->batch,seed+101);
      double elapsed=now()-t0;
      std::fprintf(stderr,
        "[methscope] upscale-train/residual: step %u/%u val_rmse=%.6f val_mae=%.6f macro_rmse=%.6f macro_mae=%.6f elapsed=%.1fs steps/s=%.2f\n",
        step,cfg->steps,vs.rmse,vs.mae,vs.macro_rmse,vs.macro_mae,elapsed,step/elapsed);
    }
  }
  Score ts=evaluate(bh,net,w,d_repr,dh,data.base,bo,cpg,test,cfg->eval_rows_per_head,cfg->batch,seed+202);
  double train_seconds=now()-t0;
  std::fprintf(stderr,
    "[methscope] upscale-train/residual: test_rmse=%.6f test_mae=%.6f macro_rmse=%.6f macro_mae=%.6f n=%llu heads=%u training_seconds=%.1f\n",
    ts.rmse,ts.mae,ts.macro_rmse,ts.macro_mae,(unsigned long long)ts.n,ts.heads,train_seconds);

  uint64_t encoder_checksum=UINT64_C(1469598103934665603);
  encoder_checksum=fnv_update(encoder_checksum,eh,sizeof(*eh));
  encoder_checksum=fnv_update(encoder_checksum,split,eh->n_cells);
  encoder_checksum=fnv_update(encoder_checksum,prep,(size_t)3*I*sizeof(float));
  encoder_checksum=fnv_update(encoder_checksum,enc.base+eh->param_offset,(size_t)frozen_floats*sizeof(float));
  uint64_t index_checksum=fnv_update(UINT64_C(1469598103934665603),idx.base,ih->file_bytes);

  UpresHeader oh={};
  std::memcpy(oh.magic,"UPRES1\0\0",8);oh.version=1;oh.rank=R;oh.hidden=H;
  oh.n_bins=ih->n_bins;oh.n_memberships=ih->n_memberships;oh.n_cells=dh->n_cells;
  oh.n_train=train.size();oh.n_val=val.size();oh.n_test=test.size();oh.steps=cfg->steps;oh.batch=cfg->batch;
  oh.n_cpg=ih->n_cpg;oh.n_residual=ih->n_residual;oh.seed=seed;
  oh.val_rmse=(float)vs.rmse;oh.val_mae=(float)vs.mae;oh.val_macro_rmse=(float)vs.macro_rmse;oh.val_macro_mae=(float)vs.macro_mae;
  oh.test_rmse=(float)ts.rmse;oh.test_mae=(float)ts.mae;oh.test_macro_rmse=(float)ts.macro_rmse;oh.test_macro_mae=(float)ts.macro_mae;
  oh.split_offset=sizeof(oh);oh.bin_offsets_offset=oh.split_offset+dh->n_cells;
  oh.cpg_offset=oh.bin_offsets_offset+(uint64_t)(ih->n_bins+1)*sizeof(uint64_t);
  oh.membership_rank_offset=oh.cpg_offset+ih->n_residual*sizeof(uint32_t);
  oh.param_offset=oh.membership_rank_offset+ih->n_residual*sizeof(uint32_t);
  oh.file_bytes=oh.param_offset+(uint64_t)(an+abn+en+ih->n_residual)*sizeof(float);
  oh.encoder_checksum=encoder_checksum;oh.index_checksum=index_checksum;
  FILE *f=std::fopen(cfg->model_path,"wb");if(!f)die_path("cannot create model",cfg->model_path);
  write_all(f,&oh,sizeof(oh),cfg->model_path);write_all(f,split,dh->n_cells,cfg->model_path);
  write_all(f,bo,(ih->n_bins+1)*sizeof(uint64_t),cfg->model_path);
  write_all(f,cpg,ih->n_residual*sizeof(uint32_t),cfg->model_path);
  write_all(f,membership,ih->n_residual*sizeof(uint32_t),cfg->model_path);
  write_device(f,net.a.t,an,cfg->model_path);write_device(f,net.ab.t,abn,cfg->model_path);
  write_device(f,net.e.t,en,cfg->model_path);write_device(f,net.eb.t,ih->n_residual,cfg->model_path);
  if(std::fclose(f))die_path("error closing model",cfg->model_path);

  char mp[4096];if(std::snprintf(mp,sizeof(mp),"%s.tsv",cfg->model_path)>=(int)sizeof(mp))die("model path too long");
  f=std::fopen(mp,"w");if(!f)die_path("cannot create model manifest",mp);
  std::fprintf(f,
    "format\tUPRES1\nactivation\tleaky_relu_0.01_local_projection_linear_cpg_decoder\n"
    "sidecar\t%s\nencoder\t%s\nindex\t%s\n"
    "residual_cpgs\t%llu\nresidual_memberships\t%u\ndecoder_heads\t%u\n"
    "hidden\t%d\nrank\t%d\ntrain_cells\t%zu\nvalidation_cells\t%zu\ntest_cells\t%zu\n"
    "steps\t%u\nbatch\t%u\neval_rows_per_head\t%u\nseed\t%llu\n"
    "learning_rate\t%.9g\nweight_decay\t%.9g\n"
    "validation_rmse\t%.9g\nvalidation_mae\t%.9g\nvalidation_macro_rmse\t%.9g\nvalidation_macro_mae\t%.9g\n"
    "test_rmse\t%.9g\ntest_mae\t%.9g\ntest_macro_rmse\t%.9g\ntest_macro_mae\t%.9g\n"
    "preparation_seconds\t%.3f\ntraining_seconds\t%.3f\n"
    "encoder_checksum\t%016llx\nindex_checksum\t%016llx\n"
    "parameter_order\tbin_W,bin_b,cpg_embedding,cpg_bias\nfile_bytes\t%llu\n",
    cfg->data_path,cfg->encoder_path,cfg->index_path,(unsigned long long)ih->n_residual,
    ih->n_memberships,ih->n_bins,H,R,train.size(),val.size(),test.size(),
    cfg->steps,cfg->batch,cfg->eval_rows_per_head,(unsigned long long)seed,
    cfg->learning_rate,cfg->weight_decay,
    vs.rmse,vs.mae,vs.macro_rmse,vs.macro_mae,
    ts.rmse,ts.mae,ts.macro_rmse,ts.macro_mae,
    prep_seconds,train_seconds,(unsigned long long)encoder_checksum,(unsigned long long)index_checksum,
    (unsigned long long)oh.file_bytes);
  if(std::fclose(f))die_path("error closing manifest",mp);
  std::fprintf(stderr,"[methscope] upscale-train/residual: wrote %s (%llu bytes) and %s\n",
               cfg->model_path,(unsigned long long)oh.file_bytes,mp);

  free_param(&net.a);free_param(&net.ab);free_param(&net.e);free_param(&net.eb);
  cudaFree(w.z);cudaFree(w.dz);cudaFree(w.y);cudaFree(w.id);cudaFree(w.met);cudaFree(d_repr);
  BLAS_OK(cublasDestroy(bh));unmap_read(&idx);unmap_read(&enc);unmap_read(&data);
  return 0;
}
