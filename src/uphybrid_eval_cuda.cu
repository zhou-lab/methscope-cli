// SPDX-License-Identifier: AGPL-3.0-or-later
/*
 * External-cohort evaluation for the frozen UPFAC3 + UPRES1 hybrid model.
 * Every finite truth value is scored, including the sparse input observations,
 * matching Hao's 2018 Zhou continuous-model evaluation protocol.
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
#include "uphybrid_eval_cuda.h"

static const int THREADS=256;
static const int METRIC_BLOCKS=128;

static void die(const char *msg){std::fprintf(stderr,"[methscope] _upscale eval: %s\n",msg);std::exit(1);}
static void die_path(const char *msg,const char *path){std::fprintf(stderr,"[methscope] _upscale eval: %s: %s\n",msg,path);std::exit(1);}
static void cuda_fail(const char *call,cudaError_t e,const char *file,int line){std::fprintf(stderr,"[methscope] _upscale eval CUDA: %s failed at %s:%d: %s\n",call,file,line,cudaGetErrorString(e));std::exit(1);}
#define CUDA_OK(x) do{cudaError_t _e=(x);if(_e!=cudaSuccess)cuda_fail(#x,_e,__FILE__,__LINE__);}while(0)
static const char *blas_msg(cublasStatus_t s){
  switch(s){case CUBLAS_STATUS_SUCCESS:return"success";case CUBLAS_STATUS_NOT_INITIALIZED:return"not initialized";case CUBLAS_STATUS_ALLOC_FAILED:return"allocation failed";case CUBLAS_STATUS_INVALID_VALUE:return"invalid value";case CUBLAS_STATUS_ARCH_MISMATCH:return"architecture mismatch";case CUBLAS_STATUS_MAPPING_ERROR:return"mapping error";case CUBLAS_STATUS_EXECUTION_FAILED:return"execution failed";case CUBLAS_STATUS_INTERNAL_ERROR:return"internal error";case CUBLAS_STATUS_NOT_SUPPORTED:return"not supported";default:return"unknown";}
}
static void blas_fail(const char *call,cublasStatus_t s,const char *file,int line){std::fprintf(stderr,"[methscope] _upscale eval cuBLAS: %s failed at %s:%d: %s\n",call,file,line,blas_msg(s));std::exit(1);}
#define BLAS_OK(x) do{cublasStatus_t _s=(x);if(_s!=CUBLAS_STATUS_SUCCESS)blas_fail(#x,_s,__FILE__,__LINE__);}while(0)

static size_t checked_mul(size_t a,size_t b,const char *what){if(a&&b>SIZE_MAX/a)die(what);return a*b;}
static float *df(size_t n){float*p=NULL;CUDA_OK(cudaMalloc((void**)&p,checked_mul(n,sizeof(float),"device size overflow")));return p;}
static uint32_t *du32(size_t n){uint32_t*p=NULL;CUDA_OK(cudaMalloc((void**)&p,checked_mul(n,sizeof(uint32_t),"device size overflow")));return p;}
static uint16_t *du16(size_t n){uint16_t*p=NULL;CUDA_OK(cudaMalloc((void**)&p,checked_mul(n,sizeof(uint16_t),"device size overflow")));return p;}
static int blocks(size_t n){size_t b=(n+THREADS-1)/THREADS;return(int)(b>4096?4096:(b?b:1));}

struct MsurHeader{
  char magic[8];uint32_t version,n_cells,n_reps,n_patterns;uint64_t n_cpg;
  uint32_t sampled_per_cell,flags;uint64_t groups_offset,truth_offset,records_offset,record_bytes;
};
static_assert(sizeof(MsurHeader)==72,"MSUR header layout");

#pragma pack(push,1)
struct UpfacHeader{
  char magic[8];uint32_t version,patterns,rank,hidden,n_input,n_active,n_cells,n_train,n_val,n_test,steps,batch;
  uint64_t n_cpg,seed;float val_rmse,val_mae,test_rmse,test_mae;
  uint64_t split_offset,prep_offset,active_offset,param_offset,file_bytes;
};
static_assert(sizeof(UpfacHeader)==128,"UPFAC3 header layout");
struct UpresHeader{
  char magic[8];uint32_t version,rank,hidden,n_bins,n_memberships,n_cells;
  uint32_t n_train,n_val,n_test,steps,batch,reserved32;uint64_t n_cpg,n_residual,seed;
  float val_rmse,val_mae,val_macro_rmse,val_macro_mae,test_rmse,test_mae,test_macro_rmse,test_macro_mae;
  uint64_t split_offset,bin_offsets_offset,cpg_offset,membership_rank_offset;
  uint64_t param_offset,file_bytes,encoder_checksum,index_checksum;
};
static_assert(sizeof(UpresHeader)==176,"UPRES1 header layout");
#pragma pack(pop)

struct Mapping{int fd;size_t size;uint8_t*base;};
static Mapping map_read(const char *path){
  Mapping m={-1,0,NULL};m.fd=open(path,O_RDONLY);if(m.fd<0)die_path("cannot open input",path);
  struct stat st;if(fstat(m.fd,&st)||st.st_size<0)die_path("cannot stat input",path);m.size=(size_t)st.st_size;
  m.base=(uint8_t*)mmap(NULL,m.size,PROT_READ,MAP_SHARED,m.fd,0);if(m.base==MAP_FAILED)die_path("cannot mmap input",path);return m;
}
static void unmap_read(Mapping*m){if(m->base&&m->base!=MAP_FAILED)munmap(m->base,m->size);if(m->fd>=0)close(m->fd);}
static bool range_ok(uint64_t off,uint64_t bytes,size_t size){return off<=size&&bytes<=(uint64_t)size-off;}

__device__ static float sigmoid(float x){if(x>=0){float z=expf(-x);return 1.0f/(1.0f+z);}float z=expf(x);return z/(1.0f+z);}
__global__ static void matrix_leaky_bias(float*x,const float*b,size_t n,int width){
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){float v=x[q]+b[q%width];x[q]=v>0?v:.01f*v;}
}
__global__ static void matrix_leaky_residual(float*x,const float*b,const float*skip,size_t n,int width){
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){float v=x[q]+b[q%width];x[q]=skip[q]+(v>0?v:.01f*v);}
}
__global__ static void add_bias(float*x,const float*b,size_t n){
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x)x[q]+=b[q];
}
__global__ static void add_leaky_bias(float*x,const float*b,size_t n){
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){float v=x[q]+b[q];x[q]=v>0?v:.01f*v;}
}

struct Metrics{
  double abs_sum,sq_sum,gt_sum,pred_sum,gt_sq_sum,pred_sq_sum,cross_sum;
  unsigned long long n,within;
};

template<bool Residual>
__global__ static void score_targets(const float*z,const float*e,const float*eb,
                                     const uint32_t*actual,const uint16_t*group_or_bin,
                                     const uint16_t*truth,size_t n_targets,int rank,Metrics*out){
  double ab=0,sq=0,gs=0,ps=0,gq=0,pq=0,cr=0;unsigned long long nn=0,wi=0;
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n_targets;q+=(size_t)blockDim.x*gridDim.x){
    uint16_t tv=truth[actual[q]];if(tv==UINT16_MAX)continue;
    size_t g=(size_t)group_or_bin[q];float s=eb[q];
    for(int r=0;r<rank;++r)s+=e[q*(size_t)rank+r]*z[g*(size_t)rank+r];
    float p=sigmoid(s),y=(float)tv/65534.0f,d=p-y,ad=fabsf(d);
    ab+=ad;sq+=(double)d*d;gs+=y;ps+=p;gq+=(double)y*y;pq+=(double)p*p;cr+=(double)y*p;++nn;if(ad<=.05f)++wi;
  }
  __shared__ double sab[THREADS],ssq[THREADS],sgs[THREADS],sps[THREADS],sgq[THREADS],spq[THREADS],scr[THREADS];
  __shared__ unsigned long long sn[THREADS],sw[THREADS];
  int t=threadIdx.x;sab[t]=ab;ssq[t]=sq;sgs[t]=gs;sps[t]=ps;sgq[t]=gq;spq[t]=pq;scr[t]=cr;sn[t]=nn;sw[t]=wi;__syncthreads();
  for(int d=blockDim.x/2;d;d>>=1){if(t<d){sab[t]+=sab[t+d];ssq[t]+=ssq[t+d];sgs[t]+=sgs[t+d];sps[t]+=sps[t+d];sgq[t]+=sgq[t+d];spq[t]+=spq[t+d];scr[t]+=scr[t+d];sn[t]+=sn[t+d];sw[t]+=sw[t+d];}__syncthreads();}
  if(!t){atomicAdd(&out->abs_sum,sab[0]);atomicAdd(&out->sq_sum,ssq[0]);atomicAdd(&out->gt_sum,sgs[0]);atomicAdd(&out->pred_sum,sps[0]);atomicAdd(&out->gt_sq_sum,sgq[0]);atomicAdd(&out->pred_sq_sum,spq[0]);atomicAdd(&out->cross_sum,scr[0]);atomicAdd(&out->n,sn[0]);atomicAdd(&out->within,sw[0]);}
}

static void gemv(cublasHandle_t h,const float*w,const float*x,float*y,int O,int I){
  const float one=1,zero=0;BLAS_OK(cublasSgemv(h,CUBLAS_OP_T,I,O,&one,w,I,x,1,&zero,y,1));
}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t fnv_update(uint64_t h,const void*vp,size_t n){const uint8_t*p=(const uint8_t*)vp;for(size_t i=0;i<n;++i){h^=p[i];h*=UINT64_C(1099511628211);}return h;}

struct Score{double mae,mse,rmse,corr,within;unsigned long long n;};
static Score score(const Metrics&m){
  Score s={NAN,NAN,NAN,NAN,NAN,m.n};if(!m.n)return s;
  s.mae=m.abs_sum/m.n;s.mse=m.sq_sum/m.n;s.rmse=std::sqrt(s.mse);s.within=(double)m.within/m.n;
  double gm=m.gt_sum/m.n,pm=m.pred_sum/m.n,cov=m.cross_sum-m.n*gm*pm;
  double gv=m.gt_sq_sum-m.n*gm*gm,pv=m.pred_sq_sum-m.n*pm*pm;
  if(gv>0&&pv>0)s.corr=cov/std::sqrt(gv*pv);return s;
}
static Metrics plus(const Metrics&a,const Metrics&b){
  Metrics z={a.abs_sum+b.abs_sum,a.sq_sum+b.sq_sum,a.gt_sum+b.gt_sum,a.pred_sum+b.pred_sum,
             a.gt_sq_sum+b.gt_sq_sum,a.pred_sq_sum+b.pred_sq_sum,a.cross_sum+b.cross_sum,
             a.n+b.n,a.within+b.within};return z;
}

extern "C" int ms_uphybrid_eval_cuda_available(void){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;}

extern "C" int ms_uphybrid_eval_cuda(const ms_uphybrid_eval_config_t*cfg){
  Mapping data=map_read(cfg->data_path),enc=map_read(cfg->encoder_path),res=map_read(cfg->residual_path);
  if(data.size<sizeof(MsurHeader)||enc.size<sizeof(UpfacHeader)||res.size<sizeof(UpresHeader))die("truncated input header");
  const MsurHeader*dh=(const MsurHeader*)data.base;const UpfacHeader*eh=(const UpfacHeader*)enc.base;const UpresHeader*rh=(const UpresHeader*)res.base;
  if(std::memcmp(dh->magic,"MSURAW2\0",8)||dh->version!=2||!(dh->flags&1))die("evaluation input must be MSURAW2 with embedded truth");
  if(std::memcmp(eh->magic,"UPFAC3\0\0",8)||eh->version!=3)die("encoder is not UPFAC3");
  if(std::memcmp(rh->magic,"UPRES1\0\0",8)||rh->version!=1)die("residual model is not UPRES1");
  if(dh->n_cpg!=eh->n_cpg||dh->n_cpg!=rh->n_cpg)die("CpG dimensions disagree");
  if(eh->patterns>dh->n_patterns||eh->n_input!=3*eh->patterns)die("feature dimensions disagree");
  if(eh->hidden!=rh->hidden||eh->n_active+rh->n_residual!=dh->n_cpg)die("global/residual output dimensions disagree");
  if(!rh->n_bins||rh->n_bins>UINT16_MAX||!rh->rank)die("unsupported residual dimensions");
  uint32_t cells=cfg->max_cells?cfg->max_cells:dh->n_cells,reps=cfg->max_reps?cfg->max_reps:dh->n_reps;
  if(cells>dh->n_cells||reps>dh->n_reps||!cells||!reps)die("--max-cells/--max-reps exceed sidecar");
  uint64_t records_end=dh->records_offset+(uint64_t)dh->n_cells*dh->n_reps*dh->record_bytes;
  uint64_t truth_bytes=(uint64_t)dh->n_cells*dh->n_cpg*sizeof(uint16_t);
  if(records_end>data.size||!range_ok(dh->truth_offset,truth_bytes,data.size))die("truncated evaluation sidecar");
  if(eh->file_bytes>enc.size||rh->file_bytes>res.size)die("truncated model file");
  int I=(int)eh->n_input,H=(int)eh->hidden,G=(int)eh->patterns,GR=(int)eh->rank;
  size_t frozen=(size_t)H*I+H+(size_t)H*H+H;
  if(!range_ok(eh->prep_offset,(uint64_t)3*I*sizeof(float),enc.size)||
     !range_ok(eh->active_offset,(uint64_t)eh->n_active*(sizeof(uint32_t)+sizeof(uint16_t)),enc.size)||
     !range_ok(eh->param_offset,(uint64_t)(frozen+(size_t)G*GR*H+(size_t)G*GR+(size_t)eh->n_active*GR+eh->n_active)*sizeof(float),enc.size))
    die("invalid UPFAC3 offsets");
  if(!range_ok(rh->bin_offsets_offset,(uint64_t)(rh->n_bins+1)*sizeof(uint64_t),res.size)||
     !range_ok(rh->cpg_offset,rh->n_residual*sizeof(uint32_t),res.size)||
     !range_ok(rh->param_offset,((uint64_t)rh->n_bins*rh->rank*H+(uint64_t)rh->n_bins*rh->rank+rh->n_residual*rh->rank+rh->n_residual)*sizeof(float),res.size))
    die("invalid UPRES1 offsets");
  uint64_t encoder_checksum=UINT64_C(1469598103934665603);
  encoder_checksum=fnv_update(encoder_checksum,eh,sizeof(*eh));
  encoder_checksum=fnv_update(encoder_checksum,enc.base+eh->split_offset,eh->n_cells);
  encoder_checksum=fnv_update(encoder_checksum,enc.base+eh->prep_offset,(size_t)3*I*sizeof(float));
  encoder_checksum=fnv_update(encoder_checksum,enc.base+eh->param_offset,frozen*sizeof(float));
  if(encoder_checksum!=rh->encoder_checksum)die("UPRES1 was trained against a different encoder");

  std::fprintf(stderr,
    "[methscope] _upscale eval: external cells=%u/%u reps=%u/%u CpGs=%llu; top=%u residual=%llu\n",
    cells,dh->n_cells,reps,dh->n_reps,(unsigned long long)dh->n_cpg,eh->n_active,(unsigned long long)rh->n_residual);
  CUDA_OK(cudaSetDevice(cfg->device));cudaDeviceProp prop;CUDA_OK(cudaGetDeviceProperties(&prop,cfg->device));
  std::fprintf(stderr,"[methscope] _upscale eval: CUDA device %d: %s (%.0f MiB)\n",cfg->device,prop.name,(double)prop.totalGlobalMem/(1024*1024));
  cublasHandle_t bh;BLAS_OK(cublasCreate(&bh));double t0=now();

  size_t rows=(size_t)dh->n_cells*reps;
  std::vector<float>feat(checked_mul(rows,(size_t)I,"feature matrix overflow"));
  const float*prep=(const float*)(enc.base+eh->prep_offset);const float*imp=prep,*mean=prep+I,*scale=prep+2*I;
  for(uint32_t rep=0;rep<reps;++rep)for(uint32_t cell=0;cell<dh->n_cells;++cell){
    size_t row=(size_t)rep*dh->n_cells+cell;const uint8_t*rec=data.base+dh->records_offset+row*dh->record_bytes;
    const float*beta=(const float*)rec;const uint32_t*count=(const uint32_t*)(rec+(size_t)dh->n_patterns*sizeof(float));float*out=feat.data()+row*I;
    for(uint32_t g=0;g<eh->patterns;++g){float raw[3]={beta[g],std::log1p((float)count[g]),std::isnan(beta[g])?1.0f:0.0f};for(int k=0;k<3;++k){int j=3*g+k;float v=std::isnan(raw[k])?imp[j]:raw[k];out[j]=(v-mean[j])/scale[j];}}
  }
  float*d_feat=df(feat.size()),*d_h1=df(rows*H),*d_repr=df(rows*H),*d_w1=df((size_t)H*I),*d_b1=df(H),*d_w2=df((size_t)H*H),*d_b2=df(H);
  const float*ep=(const float*)(enc.base+eh->param_offset);
  CUDA_OK(cudaMemcpy(d_feat,feat.data(),feat.size()*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_w1,ep,(size_t)H*I*sizeof(float),cudaMemcpyHostToDevice));ep+=(size_t)H*I;
  CUDA_OK(cudaMemcpy(d_b1,ep,H*sizeof(float),cudaMemcpyHostToDevice));ep+=H;
  CUDA_OK(cudaMemcpy(d_w2,ep,(size_t)H*H*sizeof(float),cudaMemcpyHostToDevice));ep+=(size_t)H*H;
  CUDA_OK(cudaMemcpy(d_b2,ep,H*sizeof(float),cudaMemcpyHostToDevice));ep+=H;
  const float one=1,zero=0;
  BLAS_OK(cublasSgemm(bh,CUBLAS_OP_T,CUBLAS_OP_N,H,(int)rows,I,&one,d_w1,I,d_feat,I,&zero,d_h1,H));
  matrix_leaky_bias<<<blocks(rows*H),THREADS>>>(d_h1,d_b1,rows*H,H);
  BLAS_OK(cublasSgemm(bh,CUBLAS_OP_T,CUBLAS_OP_N,H,(int)rows,H,&one,d_w2,H,d_h1,H,&zero,d_repr,H));
  matrix_leaky_residual<<<blocks(rows*H),THREADS>>>(d_repr,d_b2,d_h1,rows*H,H);
  feat.clear();feat.shrink_to_fit();cudaFree(d_feat);cudaFree(d_h1);cudaFree(d_w1);cudaFree(d_b1);cudaFree(d_w2);cudaFree(d_b2);

  const uint32_t*top_actual=(const uint32_t*)(enc.base+eh->active_offset);
  const uint16_t*top_group=(const uint16_t*)(enc.base+eh->active_offset+(uint64_t)eh->n_active*sizeof(uint32_t));
  uint32_t*d_top_actual=du32(eh->n_active);uint16_t*d_top_group=du16(eh->n_active);
  CUDA_OK(cudaMemcpy(d_top_actual,top_actual,(size_t)eh->n_active*sizeof(uint32_t),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_top_group,top_group,(size_t)eh->n_active*sizeof(uint16_t),cudaMemcpyHostToDevice));
  size_t gwn=(size_t)G*GR*H,gbn=(size_t)G*GR,gen=(size_t)eh->n_active*GR;
  float*d_gw=df(gwn),*d_gb=df(gbn),*d_ge=df(gen),*d_geb=df(eh->n_active),*d_gz=df(gbn);
  CUDA_OK(cudaMemcpy(d_gw,ep,gwn*sizeof(float),cudaMemcpyHostToDevice));ep+=gwn;
  CUDA_OK(cudaMemcpy(d_gb,ep,gbn*sizeof(float),cudaMemcpyHostToDevice));ep+=gbn;
  CUDA_OK(cudaMemcpy(d_ge,ep,gen*sizeof(float),cudaMemcpyHostToDevice));ep+=gen;
  CUDA_OK(cudaMemcpy(d_geb,ep,(size_t)eh->n_active*sizeof(float),cudaMemcpyHostToDevice));

  const uint64_t*bo=(const uint64_t*)(res.base+rh->bin_offsets_offset);if(bo[0]||bo[rh->n_bins]!=rh->n_residual)die("invalid UPRES1 head offsets");
  const uint32_t*res_actual=(const uint32_t*)(res.base+rh->cpg_offset);uint32_t*d_res_actual=du32(rh->n_residual);
  CUDA_OK(cudaMemcpy(d_res_actual,res_actual,rh->n_residual*sizeof(uint32_t),cudaMemcpyHostToDevice));
  std::vector<uint16_t>res_bin(rh->n_residual);
  for(uint32_t b=0;b<rh->n_bins;++b){if(bo[b]>=bo[b+1])die("invalid UPRES1 head offsets");for(uint64_t q=bo[b];q<bo[b+1];++q)res_bin[q]=(uint16_t)b;}
  uint16_t*d_res_bin=du16(rh->n_residual);CUDA_OK(cudaMemcpy(d_res_bin,res_bin.data(),rh->n_residual*sizeof(uint16_t),cudaMemcpyHostToDevice));res_bin.clear();res_bin.shrink_to_fit();
  int RR=(int)rh->rank;size_t ran=(size_t)rh->n_bins*RR*H,rabn=(size_t)rh->n_bins*RR,ren=(size_t)rh->n_residual*RR;
  const float*rp=(const float*)(res.base+rh->param_offset);
  float*d_ra=df(ran),*d_rab=df(rabn),*d_re=df(ren),*d_reb=df(rh->n_residual),*d_rz=df(rabn);
  CUDA_OK(cudaMemcpy(d_ra,rp,ran*sizeof(float),cudaMemcpyHostToDevice));rp+=ran;
  CUDA_OK(cudaMemcpy(d_rab,rp,rabn*sizeof(float),cudaMemcpyHostToDevice));rp+=rabn;
  CUDA_OK(cudaMemcpy(d_re,rp,ren*sizeof(float),cudaMemcpyHostToDevice));rp+=ren;
  CUDA_OK(cudaMemcpy(d_reb,rp,rh->n_residual*sizeof(float),cudaMemcpyHostToDevice));

  uint16_t*d_truth=du16(dh->n_cpg);Metrics*d_tm=NULL,*d_rm=NULL;CUDA_OK(cudaMalloc((void**)&d_tm,sizeof(Metrics)));CUDA_OK(cudaMalloc((void**)&d_rm,sizeof(Metrics)));
  CUDA_OK(cudaMemset(d_tm,0,sizeof(Metrics)));CUDA_OK(cudaMemset(d_rm,0,sizeof(Metrics)));CUDA_OK(cudaDeviceSynchronize());
  double prep_seconds=now()-t0,eval0=now();const uint16_t*truth=(const uint16_t*)(data.base+dh->truth_offset);
  for(uint32_t cell=0;cell<cells;++cell){
    CUDA_OK(cudaMemcpy(d_truth,truth+(size_t)cell*dh->n_cpg,dh->n_cpg*sizeof(uint16_t),cudaMemcpyHostToDevice));
    for(uint32_t rep=0;rep<reps;++rep){
      const float*h=d_repr+((size_t)rep*dh->n_cells+cell)*H;
      gemv(bh,d_gw,h,d_gz,G*GR,H);add_bias<<<blocks(gbn),THREADS>>>(d_gz,d_gb,gbn);
      score_targets<false><<<METRIC_BLOCKS,THREADS>>>(d_gz,d_ge,d_geb,d_top_actual,d_top_group,d_truth,eh->n_active,GR,d_tm);
      gemv(bh,d_ra,h,d_rz,rh->n_bins*RR,H);add_leaky_bias<<<blocks(rabn),THREADS>>>(d_rz,d_rab,rabn);
      score_targets<true><<<METRIC_BLOCKS,THREADS>>>(d_rz,d_re,d_reb,d_res_actual,d_res_bin,d_truth,rh->n_residual,RR,d_rm);
    }
    if((cell+1)%cfg->log_every_cells==0||cell+1==cells){CUDA_OK(cudaDeviceSynchronize());double elapsed=now()-eval0;std::fprintf(stderr,"[methscope] _upscale eval: cells %u/%u elapsed=%.1fs rows/s=%.2f\n",cell+1,cells,elapsed,(double)(cell+1)*reps/elapsed);}
  }
  Metrics tm,rm;CUDA_OK(cudaMemcpy(&tm,d_tm,sizeof(tm),cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(&rm,d_rm,sizeof(rm),cudaMemcpyDeviceToHost));
  Metrics cm=plus(tm,rm);Score ts=score(tm),rs=score(rm),cs=score(cm);double eval_seconds=now()-eval0;
  std::fprintf(stderr,
    "[methscope] _upscale eval: top MAE=%.6f RMSE=%.6f n=%llu\n"
    "[methscope] _upscale eval: residual MAE=%.6f RMSE=%.6f n=%llu\n"
    "[methscope] _upscale eval: combined MAE=%.6f RMSE=%.6f corr=%.6f n=%llu\n",
    ts.mae,ts.rmse,ts.n,rs.mae,rs.rmse,rs.n,cs.mae,cs.rmse,cs.corr,cs.n);
  FILE*f=std::fopen(cfg->metrics_path,"w");if(!f)die_path("cannot create metrics",cfg->metrics_path);
  std::fprintf(f,
    "# format\tMSHYBRID-EVAL1\n# sidecar\t%s\n# encoder\t%s\n# residual\t%s\n"
    "# cells\t%u\n# reps\t%u\n# observed_targets\tincluded\n"
    "# preparation_seconds\t%.6f\n# evaluation_seconds\t%.6f\n"
    "target_set\tn_valid\tmae\tmse\trmse\tpearson_corr\twithin_abs_error_le_0.05\n"
    "top1000\t%llu\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n"
    "residual\t%llu\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n"
    "combined\t%llu\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n",
    cfg->data_path,cfg->encoder_path,cfg->residual_path,cells,reps,prep_seconds,eval_seconds,
    ts.n,ts.mae,ts.mse,ts.rmse,ts.corr,ts.within,
    rs.n,rs.mae,rs.mse,rs.rmse,rs.corr,rs.within,
    cs.n,cs.mae,cs.mse,cs.rmse,cs.corr,cs.within);
  if(std::fclose(f))die_path("error closing metrics",cfg->metrics_path);
  std::fprintf(stderr,"[methscope] _upscale eval: wrote %s\n",cfg->metrics_path);

  cudaFree(d_tm);cudaFree(d_rm);cudaFree(d_truth);cudaFree(d_repr);
  cudaFree(d_top_actual);cudaFree(d_top_group);cudaFree(d_gw);cudaFree(d_gb);cudaFree(d_ge);cudaFree(d_geb);cudaFree(d_gz);
  cudaFree(d_res_actual);cudaFree(d_res_bin);cudaFree(d_ra);cudaFree(d_rab);cudaFree(d_re);cudaFree(d_reb);cudaFree(d_rz);
  BLAS_OK(cublasDestroy(bh));unmap_read(&res);unmap_read(&enc);unmap_read(&data);return 0;
}
