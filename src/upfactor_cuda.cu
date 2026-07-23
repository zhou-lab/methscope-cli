// SPDX-License-Identifier: AGPL-3.0-or-later
/* Native CUDA/cuBLAS trainer for the global MRMP-factorized decoder.
 *
 * x (beta, log1p coverage, missing for each MRMP)
 *   -> ReLU(W1 x + b1) -> ReLU(W2 h1 + b2) = h
 *   -> z_g = G_g h + gb_g                         (16 factors / MRMP)
 *   -> sigmoid(e_c . z_group(c) + bias_c)         (linear CpG decoder)
 *
 * Only target CpGs belonging to P1..PN are represented. PNA is background.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
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
#include "upfactor_cuda.h"

static const int THREADS=256;

static void die(const char *msg) {
  std::fprintf(stderr,"[methscope] upscale-factor-train: %s\n",msg); std::exit(1);
}
static void die_path(const char *msg,const char *path) {
  std::fprintf(stderr,"[methscope] upscale-factor-train: %s: %s\n",msg,path); std::exit(1);
}
static void cuda_fail(const char *call,cudaError_t e,const char *file,int line) {
  std::fprintf(stderr,"[methscope] upscale-factor-train CUDA: %s failed at %s:%d: %s\n",call,file,line,cudaGetErrorString(e)); std::exit(1);
}
#define CUDA_OK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess) cuda_fail(#x,_e,__FILE__,__LINE__); } while(0)
static const char *blas_msg(cublasStatus_t s) {
  switch(s){case CUBLAS_STATUS_SUCCESS:return "success";case CUBLAS_STATUS_NOT_INITIALIZED:return "not initialized";case CUBLAS_STATUS_ALLOC_FAILED:return "allocation failed";case CUBLAS_STATUS_INVALID_VALUE:return "invalid value";case CUBLAS_STATUS_ARCH_MISMATCH:return "architecture mismatch";case CUBLAS_STATUS_MAPPING_ERROR:return "mapping error";case CUBLAS_STATUS_EXECUTION_FAILED:return "execution failed";case CUBLAS_STATUS_INTERNAL_ERROR:return "internal error";case CUBLAS_STATUS_NOT_SUPPORTED:return "not supported";default:return "unknown";}
}
static void blas_fail(const char *call,cublasStatus_t s,const char *file,int line) {
  std::fprintf(stderr,"[methscope] upscale-factor-train cuBLAS: %s failed at %s:%d: %s\n",call,file,line,blas_msg(s)); std::exit(1);
}
#define BLAS_OK(x) do { cublasStatus_t _s=(x); if(_s!=CUBLAS_STATUS_SUCCESS) blas_fail(#x,_s,__FILE__,__LINE__); } while(0)

static size_t checked_mul(size_t a,size_t b,const char *what) {
  if(a && b>SIZE_MAX/a) die(what); return a*b;
}
static float *df(size_t n){float*p=NULL;CUDA_OK(cudaMalloc((void**)&p,checked_mul(n,sizeof(float),"device size overflow")));return p;}
static uint32_t *du32(size_t n){uint32_t*p=NULL;CUDA_OK(cudaMalloc((void**)&p,checked_mul(n,sizeof(uint32_t),"device size overflow")));return p;}
static uint16_t *du16(size_t n){uint16_t*p=NULL;CUDA_OK(cudaMalloc((void**)&p,checked_mul(n,sizeof(uint16_t),"device size overflow")));return p;}

struct Pcg { uint64_t state,inc; };
static uint32_t pcg32(Pcg *p){uint64_t o=p->state;p->state=o*UINT64_C(6364136223846793005)+p->inc;uint32_t x=(uint32_t)(((o>>18)^o)>>27),r=(uint32_t)(o>>59);return(x>>r)|(x<<((-r)&31));}
static void pcg_seed(Pcg*p,uint64_t s){p->state=0;p->inc=UINT64_C(1442695040888963407);pcg32(p);p->state+=s;pcg32(p);}
static float rand_unit(Pcg*p){return((pcg32(p)>>8)+0.5f)*(1.0f/16777216.0f);}
static float rand_sym(Pcg*p,float a){return(2.0f*rand_unit(p)-1.0f)*a;}

struct MsurHeader {
  char magic[8]; uint32_t version,n_cells,n_reps,n_patterns; uint64_t n_cpg;
  uint32_t sampled_per_cell,flags;
  uint64_t groups_offset,truth_offset,records_offset,record_bytes;
};
static_assert(sizeof(MsurHeader)==72,"MSUR header layout");

#pragma pack(push,1)
struct ModelHeader {
  char magic[8];
  uint32_t version,patterns,rank,hidden,n_input,n_active,n_cells,n_train,n_val,n_test,steps,batch;
  uint64_t n_cpg,seed;
  float val_rmse,val_mae,test_rmse,test_mae;
  uint64_t split_offset,prep_offset,active_offset,param_offset,file_bytes;
};
#pragma pack(pop)
static_assert(sizeof(ModelHeader)==128,"model header layout");

struct Mapping { int fd; size_t size; uint8_t *base; };
static Mapping map_read(const char *path) {
  Mapping m={-1,0,NULL};m.fd=open(path,O_RDONLY);if(m.fd<0)die_path("cannot open sidecar",path);
  struct stat st;if(fstat(m.fd,&st)||st.st_size<0)die_path("cannot stat sidecar",path);m.size=(size_t)st.st_size;
  m.base=(uint8_t*)mmap(NULL,m.size,PROT_READ,MAP_SHARED,m.fd,0);if(m.base==MAP_FAILED)die_path("cannot mmap sidecar",path);return m;
}
static void unmap_read(Mapping*m){if(m->base&&m->base!=MAP_FAILED)munmap(m->base,m->size);if(m->fd>=0)close(m->fd);}

__device__ static float sigmoid(float x){if(x>=0){float z=expf(-x);return 1.0f/(1.0f+z);}float z=expf(x);return z/(1.0f+z);}
static int blocks(size_t n){size_t b=(n+THREADS-1)/THREADS;return(int)(b>4096?4096:(b?b:1));}

__global__ static void add_leaky_relu(float*x,const float*b,int n){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x){float v=x[q]+b[q];x[q]=v>0?v:.01f*v;}}
__global__ static void add_leaky_residual(float*x,const float*b,const float*skip,int n){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x){float v=x[q]+b[q];x[q]=skip[q]+(v>0?v:.01f*v);}}
__global__ static void add_bias(float*x,const float*b,int n){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x)x[q]+=b[q];}
__global__ static void leaky_relu_back(float*d,const float*a,int n){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x)if(!(a[q]>0))d[q]*=.01f;}
__global__ static void leaky_residual_back(float*d,const float*out,const float*skip,int n){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x)if(!(out[q]-skip[q]>0))d[q]*=.01f;}
__global__ static void add_vector(float*x,const float*y,int n){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x)x[q]+=y[q];}

__global__ static void adam_vec(float*t,float*m,float*v,const float*g,size_t n,float lr,float wd,float ib1,float ib2){
  for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){float z=g[q]+wd*t[q],mm=.9f*m[q]+.1f*z,vv=.999f*v[q]+.001f*z*z;m[q]=mm;v[q]=vv;t[q]-=lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}
}
__global__ static void adam_outer(float*t,float*m,float*v,const float*go,const float*in,int O,int I,float lr,float wd,float ib1,float ib2){
  size_t n=(size_t)O*I;for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){int o=(int)(q/I),j=(int)(q%I);float z=go[o]*in[j]+wd*t[q],mm=.9f*m[q]+.1f*z,vv=.999f*v[q]+.001f*z*z;m[q]=mm;v[q]=vv;t[q]-=lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}
}

struct Metrics { double sse,sae; unsigned long long n; };
__global__ static void predict_metrics(const float*z,const float*e,const float*eb,const uint32_t*id,const uint16_t*g,const float*y,int B,int R,Metrics*out){
  double sse=0,sae=0;unsigned long long n=0;
  for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<B;q+=blockDim.x*gridDim.x){int gg=(int)g[q];uint32_t c=id[q];float s=eb[c];for(int r=0;r<R;++r)s+=e[(size_t)c*R+r]*z[(size_t)gg*R+r];float d=sigmoid(s)-y[q];sse+=(double)d*d;sae+=fabs((double)d);++n;}
  __shared__ double ss[THREADS],sa[THREADS];__shared__ unsigned long long sn[THREADS];ss[threadIdx.x]=sse;sa[threadIdx.x]=sae;sn[threadIdx.x]=n;__syncthreads();
  for(int d=blockDim.x/2;d;d>>=1){if(threadIdx.x<d){ss[threadIdx.x]+=ss[threadIdx.x+d];sa[threadIdx.x]+=sa[threadIdx.x+d];sn[threadIdx.x]+=sn[threadIdx.x+d];}__syncthreads();}
  if(threadIdx.x==0){atomicAdd(&out->sse,ss[0]);atomicAdd(&out->sae,sa[0]);atomicAdd(&out->n,sn[0]);}
}

__global__ static void decoder_backward(const float*z,float*e,float*em,float*ev,float*eb,float*ebm,float*ebv,const uint32_t*id,const uint16_t*g,const float*y,float*dz,int B,int R,float lr,float wd,float ib1,float ib2){
  for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<B;q+=blockDim.x*gridDim.x){int gg=(int)g[q];uint32_t c=id[q];float s=eb[c];for(int r=0;r<R;++r)s+=e[(size_t)c*R+r]*z[(size_t)gg*R+r];float p=sigmoid(s);float dl=2.0f*(p-y[q])*p*(1.0f-p)/B;
    for(int r=0;r<R;++r){size_t eq=(size_t)c*R+r,zq=(size_t)gg*R+r;float old=e[eq];atomicAdd(dz+zq,dl*old);float gr=dl*z[zq]+wd*old,mm=.9f*em[eq]+.1f*gr,vv=.999f*ev[eq]+.001f*gr*gr;em[eq]=mm;ev[eq]=vv;e[eq]=old-lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}
    float old=eb[c],gr=dl,mm=.9f*ebm[c]+.1f*gr,vv=.999f*ebv[c]+.001f*gr*gr;ebm[c]=mm;ebv[c]=vv;eb[c]=old-lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);
  }
}

struct Param { float*t,*m,*v;size_t n; };
static Param param(size_t n,const std::vector<float>&h){Param p={df(n),df(n),df(n),n};CUDA_OK(cudaMemcpy(p.t,h.data(),n*sizeof(float),cudaMemcpyHostToDevice));CUDA_OK(cudaMemset(p.m,0,n*sizeof(float)));CUDA_OK(cudaMemset(p.v,0,n*sizeof(float)));return p;}
static void free_param(Param*p){cudaFree(p->t);cudaFree(p->m);cudaFree(p->v);p->t=p->m=p->v=NULL;}

struct Net { Param w1,b1,w2,b2,gw,gb,e,eb; int I,H,G,R; };
struct Work { float*x,*h1,*h2,*z,*dh2,*dh1,*dskip,*dz,*y;uint32_t*id;uint16_t*g;Metrics*met; };

static void gemv(cublasHandle_t h,const float*w,const float*x,float*y,int O,int I,bool transpose_back=false){
  const float one=1,zero=0;
  if(!transpose_back)BLAS_OK(cublasSgemv(h,CUBLAS_OP_T,I,O,&one,w,I,x,1,&zero,y,1));
  else BLAS_OK(cublasSgemv(h,CUBLAS_OP_N,I,O,&one,w,I,x,1,&zero,y,1));
}
static void forward(cublasHandle_t bh,const Net&n,Work&w){
  gemv(bh,n.w1.t,w.x,w.h1,n.H,n.I);add_leaky_relu<<<blocks(n.H),THREADS>>>(w.h1,n.b1.t,n.H);
  gemv(bh,n.w2.t,w.h1,w.h2,n.H,n.H);add_leaky_residual<<<blocks(n.H),THREADS>>>(w.h2,n.b2.t,w.h1,n.H);
  gemv(bh,n.gw.t,w.h2,w.z,n.G*n.R,n.H);add_bias<<<blocks((size_t)n.G*n.R),THREADS>>>(w.z,n.gb.t,n.G*n.R);
}

static bool selected_has(const uint32_t*p,uint32_t n,uint32_t x){return std::binary_search(p,p+n,x);}
static uint32_t fill_batch(const MsurHeader*h,const uint8_t*base,const std::vector<uint32_t>&active,const std::vector<uint16_t>&ag,const std::vector<uint32_t>*pool,uint32_t cell,uint32_t rep,uint64_t start,uint32_t want,uint32_t out0,std::vector<uint32_t>&id,std::vector<uint16_t>&group,std::vector<float>&y){
  size_t row=(size_t)rep*h->n_cells+cell;const uint8_t*rec=base+h->records_offset+row*h->record_bytes;
  const uint32_t*sel=(const uint32_t*)(rec+(size_t)h->n_patterns*(sizeof(float)+sizeof(uint32_t)));
  const uint16_t*truth=(const uint16_t*)(base+h->truth_offset)+(size_t)cell*h->n_cpg;
  size_t available=pool?pool->size():active.size();if(!available||!want)return 0;
  uint64_t scanned=0,q=start%available;uint32_t n=0;
  while(n<want&&scanned<available){uint32_t local=pool?(*pool)[q]:(uint32_t)q;uint32_t pos=active[local],v=truth[pos];if(v!=UINT16_MAX&&!selected_has(sel,h->sampled_per_cell,pos)){id[out0+n]=local;group[out0+n]=ag[local];y[out0+n]=(float)v/65534.0f;++n;}q++;if(q==available)q=0;++scanned;}
  return n;
}

struct Score { double rmse,mae;uint64_t n; };
static Score evaluate(cublasHandle_t bh,const Net&net,Work&w,const MsurHeader*h,const uint8_t*base,const std::vector<uint32_t>&active,const std::vector<uint16_t>&ag,const std::vector<uint32_t>*pool,const std::vector<uint32_t>&cells,const std::vector<float>&features,uint32_t batches,uint32_t batch,uint64_t seed){
  Pcg rng;pcg_seed(&rng,seed);Metrics total={0,0,0};std::vector<uint32_t>id(batch);std::vector<uint16_t>g(batch);std::vector<float>y(batch);
  for(uint32_t k=0;k<batches;++k){uint32_t cell=cells[pcg32(&rng)%cells.size()],rep=pcg32(&rng)%h->n_reps;uint32_t B=fill_batch(h,base,active,ag,pool,cell,rep,pcg32(&rng),batch,0,id,g,y);if(!B)continue;size_t row=(size_t)rep*h->n_cells+cell;
    CUDA_OK(cudaMemcpy(w.x,features.data()+row*net.I,net.I*sizeof(float),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(w.id,id.data(),B*sizeof(uint32_t),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(w.g,g.data(),B*sizeof(uint16_t),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(w.y,y.data(),B*sizeof(float),cudaMemcpyHostToDevice));CUDA_OK(cudaMemset(w.met,0,sizeof(Metrics)));forward(bh,net,w);predict_metrics<<<blocks(B),THREADS>>>(w.z,net.e.t,net.eb.t,w.id,w.g,w.y,B,net.R,w.met);Metrics m;CUDA_OK(cudaMemcpy(&m,w.met,sizeof(m),cudaMemcpyDeviceToHost));total.sse+=m.sse;total.sae+=m.sae;total.n+=m.n;
  }
  Score s={NAN,NAN,total.n};if(total.n){s.rmse=std::sqrt(total.sse/total.n);s.mae=total.sae/total.n;}return s;
}

static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void write_all(FILE*f,const void*p,size_t n,const char*path){if(n&&fwrite(p,1,n,f)!=n)die_path("model write failed",path);}
static void write_device(FILE*f,const float*d,size_t n,const char*path){const size_t chunk=UINT64_C(16)*1024*1024;std::vector<float>b(std::min(n,chunk));size_t q=0;while(q<n){size_t k=std::min(n-q,chunk);CUDA_OK(cudaMemcpy(b.data(),d+q,k*sizeof(float),cudaMemcpyDeviceToHost));write_all(f,b.data(),k*sizeof(float),path);q+=k;}}

extern "C" int ms_upfactor_cuda_available(void){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;}

extern "C" int ms_upfactor_train_cuda(const ms_upfactor_config_t*cfg){
  Mapping map=map_read(cfg->data_path);if(map.size<sizeof(MsurHeader))die("truncated sidecar");const MsurHeader*h=(const MsurHeader*)map.base;
  if(std::memcmp(h->magic,"MSURAW2\0",8)||h->version!=2)die("input is not MSURAW2");if(!(h->flags&1)||!h->truth_offset)die("sidecar lacks embedded truth; rebuild with --embed-truth");
  uint32_t G=cfg->patterns?cfg->patterns:h->n_patterns;if(G>h->n_patterns)die("--patterns exceeds sidecar patterns");if(!G||G>UINT16_MAX)die("unsupported pattern count");
  uint64_t expected=h->records_offset+(uint64_t)h->n_cells*h->n_reps*h->record_bytes;if(expected>map.size)die("truncated sidecar records");
  const uint16_t*allg=(const uint16_t*)(map.base+h->groups_offset);std::vector<uint32_t>active;std::vector<uint16_t>ag;active.reserve(h->n_cpg/2);ag.reserve(h->n_cpg/2);
  for(uint64_t q=0;q<h->n_cpg;++q)if(allg[q]&&allg[q]<=G){active.push_back((uint32_t)q);ag.push_back((uint16_t)(allg[q]-1));}
  if(active.empty()||active.size()>UINT32_MAX)die("no supported active CpGs");if(cfg->batch>active.size())die("--batch exceeds active CpGs");
  std::vector<uint8_t>is_homogeneous(G,0);
  if(cfg->homogeneous_groups&&*cfg->homogeneous_groups){
    const char*p=cfg->homogeneous_groups;
    while(*p){char*end=NULL;errno=0;unsigned long v=std::strtoul(p,&end,10);if(errno||end==p||v<1||v>G||(*end&&*end!=','))die("invalid --homogeneous-groups");is_homogeneous[v-1]=1;if(!*end)break;p=end+1;if(!*p)die("invalid --homogeneous-groups");}
  }
  std::vector<uint32_t>easy_pool,variable_pool;
  if(cfg->homogeneous_groups&&*cfg->homogeneous_groups){
    easy_pool.reserve(active.size()/2);variable_pool.reserve(active.size()/2);
    for(uint32_t q=0;q<active.size();++q)(is_homogeneous[ag[q]]?easy_pool:variable_pool).push_back(q);
    if(easy_pool.empty()||variable_pool.empty())die("--homogeneous-groups produced an empty target stratum");
  }

  std::vector<uint32_t>order(h->n_cells);for(uint32_t q=0;q<h->n_cells;++q)order[q]=q;Pcg split_rng;pcg_seed(&split_rng,cfg->seed);for(uint32_t q=h->n_cells;q>1;--q)std::swap(order[q-1],order[pcg32(&split_rng)%q]);
  uint32_t nt=(uint32_t)(h->n_cells*.70),nv=(uint32_t)(h->n_cells*.15);if(!nt)nt=1;if(!nv)nv=1;if(nt+nv>=h->n_cells)die("not enough cells for train/validation/test");
  std::vector<uint32_t>train(order.begin(),order.begin()+nt),val(order.begin()+nt,order.begin()+nt+nv),test(order.begin()+nt+nv,order.end());std::vector<uint8_t>split(h->n_cells,2);for(uint32_t c:train)split[c]=0;for(uint32_t c:val)split[c]=1;

  int I=(int)(3*G),H=(int)cfg->hidden,R=(int)cfg->rank;size_t rows=(size_t)h->n_cells*h->n_reps;
  std::fprintf(stderr,"[methscope] upscale-factor-train: data cells=%u reps=%u CpGs=%llu active=%zu patterns=%u\n",h->n_cells,h->n_reps,(unsigned long long)h->n_cpg,active.size(),G);
  if(!easy_pool.empty())std::fprintf(stderr,"[methscope] upscale-factor-train: stratified targets homogeneous=%zu variable=%zu homogeneous_fraction=%.3f\n",easy_pool.size(),variable_pool.size(),cfg->homogeneous_fraction);
  std::fprintf(stderr,"[methscope] upscale-factor-train: split cells train=%zu val=%zu test=%zu; input=%d hidden=%d rank=%d\n",train.size(),val.size(),test.size(),I,H,R);
  std::vector<float>feat(checked_mul(rows,(size_t)I,"feature size overflow"));
  for(size_t row=0;row<rows;++row){const uint8_t*rec=map.base+h->records_offset+row*h->record_bytes;const float*b=(const float*)rec;const uint32_t*c=(const uint32_t*)(rec+(size_t)h->n_patterns*sizeof(float));float*out=feat.data()+row*I;for(uint32_t g=0;g<G;++g){out[3*g]=b[g];out[3*g+1]=std::log1p((float)c[g]);out[3*g+2]=std::isnan(b[g])?1.0f:0.0f;}}
  std::vector<float>imp(I,0),mean(I,0),scale(I,1);std::vector<uint64_t>cnt(I,0);uint64_t train_rows=(uint64_t)train.size()*h->n_reps;
  for(uint32_t rep=0;rep<h->n_reps;++rep)for(uint32_t cell:train){const float*x=feat.data()+((size_t)rep*h->n_cells+cell)*I;for(int j=0;j<I;++j)if(!std::isnan(x[j])){imp[j]+=x[j];cnt[j]++;}}
  for(int j=0;j<I;++j)if(cnt[j])imp[j]/=(float)cnt[j];
  for(uint32_t rep=0;rep<h->n_reps;++rep)for(uint32_t cell:train){float*x=feat.data()+((size_t)rep*h->n_cells+cell)*I;for(int j=0;j<I;++j){float v=std::isnan(x[j])?imp[j]:x[j];mean[j]+=v;}}
  for(int j=0;j<I;++j)mean[j]/=(float)train_rows;
  std::vector<double>ss(I,0);for(uint32_t rep=0;rep<h->n_reps;++rep)for(uint32_t cell:train){float*x=feat.data()+((size_t)rep*h->n_cells+cell)*I;for(int j=0;j<I;++j){double v=std::isnan(x[j])?imp[j]:x[j],d=v-mean[j];ss[j]+=d*d;}}
  for(int j=0;j<I;++j){scale[j]=(float)std::sqrt(ss[j]/train_rows);if(!(scale[j]>1e-8f))scale[j]=1;}
  for(size_t row=0;row<rows;++row){float*x=feat.data()+row*I;for(int j=0;j<I;++j){float v=std::isnan(x[j])?imp[j]:x[j];x[j]=(v-mean[j])/scale[j];}}

  std::vector<float>bias(active.size(),0),bsum(active.size(),0);std::vector<uint16_t>bcnt(active.size(),0);const uint16_t*truth=(const uint16_t*)(map.base+h->truth_offset);
  for(uint32_t cell:train){const uint16_t*tr=truth+(size_t)cell*h->n_cpg;for(size_t q=0;q<active.size();++q){uint16_t v=tr[active[q]];if(v!=UINT16_MAX){bsum[q]+=(float)v/65534.0f;bcnt[q]++;}}}
  for(size_t q=0;q<active.size();++q){float p=bcnt[q]?bsum[q]/bcnt[q]:.5f;p=std::max(.01f,std::min(.99f,p));bias[q]=std::log(p/(1-p));}
  bsum.clear();bcnt.clear();bsum.shrink_to_fit();bcnt.shrink_to_fit();

  CUDA_OK(cudaSetDevice(cfg->device));cudaDeviceProp prop;CUDA_OK(cudaGetDeviceProperties(&prop,cfg->device));std::fprintf(stderr,"[methscope] upscale-factor-train: CUDA device %d: %s (%.0f MiB, cc %d.%d)\n",cfg->device,prop.name,(double)prop.totalGlobalMem/(1024*1024),prop.major,prop.minor);
  Pcg init;pcg_seed(&init,cfg->seed^UINT64_C(0xa0761d6478bd642f));std::vector<float>w1((size_t)H*I),b1(H,0),w2((size_t)H*H),b2(H,0),gw((size_t)G*R*H),gb((size_t)G*R,0),e(active.size()*R);
  float a1=std::sqrt(6.0f/(I+H)),a2=std::sqrt(6.0f/(2*H)),agw=std::sqrt(6.0f/(H+R));for(float&x:w1)x=rand_sym(&init,a1);for(float&x:w2)x=rand_sym(&init,a2);for(float&x:gw)x=rand_sym(&init,agw);for(float&x:e)x=rand_sym(&init,.02f);
  Net net={param(w1.size(),w1),param(b1.size(),b1),param(w2.size(),w2),param(b2.size(),b2),param(gw.size(),gw),param(gb.size(),gb),param(e.size(),e),param(bias.size(),bias),I,H,(int)G,R};w1.clear();w2.clear();gw.clear();e.clear();bias.clear();
  Work w={df(I),df(H),df(H),df((size_t)G*R),df(H),df(H),df(H),df((size_t)G*R),df(cfg->batch),du32(cfg->batch),du16(cfg->batch),NULL};CUDA_OK(cudaMalloc((void**)&w.met,sizeof(Metrics)));
  cublasHandle_t bh;BLAS_OK(cublasCreate(&bh));std::vector<uint32_t>bid(cfg->batch);std::vector<uint16_t>bg(cfg->batch);std::vector<float>by(cfg->batch);Pcg rng;pcg_seed(&rng,cfg->seed^UINT64_C(0xe7037ed1a0b428db));
  Score v0=evaluate(bh,net,w,h,map.base,active,ag,NULL,val,feat,cfg->eval_batches,cfg->batch,cfg->seed+101);std::fprintf(stderr,"[methscope] upscale-factor-train: step 0 val_rmse=%.6f val_mae=%.6f n=%llu\n",v0.rmse,v0.mae,(unsigned long long)v0.n);double t0=now();Score vs=v0;
  for(uint32_t step=1;step<=cfg->steps;++step){uint32_t cell=train[pcg32(&rng)%train.size()],rep=pcg32(&rng)%h->n_reps;uint32_t B=0;if(easy_pool.empty()){B=fill_batch(h,map.base,active,ag,NULL,cell,rep,pcg32(&rng),cfg->batch,0,bid,bg,by);}else{uint32_t ne=(uint32_t)llround(cfg->batch*cfg->homogeneous_fraction);if(ne>cfg->batch)ne=cfg->batch;uint32_t nv=cfg->batch-ne;B=fill_batch(h,map.base,active,ag,&variable_pool,cell,rep,pcg32(&rng),nv,0,bid,bg,by);B+=fill_batch(h,map.base,active,ag,&easy_pool,cell,rep,pcg32(&rng),ne,B,bid,bg,by);}if(!B)die("could not form training batch");size_t row=(size_t)rep*h->n_cells+cell;
    CUDA_OK(cudaMemcpy(w.x,feat.data()+row*I,I*sizeof(float),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(w.id,bid.data(),B*sizeof(uint32_t),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(w.g,bg.data(),B*sizeof(uint16_t),cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(w.y,by.data(),B*sizeof(float),cudaMemcpyHostToDevice));forward(bh,net,w);CUDA_OK(cudaMemset(w.dz,0,(size_t)G*R*sizeof(float)));
    float ib1=1.0f/(1.0f-std::pow(.9f,(float)step)),ib2=1.0f/(1.0f-std::pow(.999f,(float)step)),lr=(float)cfg->learning_rate,wd=(float)cfg->weight_decay;
    decoder_backward<<<blocks(B),THREADS>>>(w.z,net.e.t,net.e.m,net.e.v,net.eb.t,net.eb.m,net.eb.v,w.id,w.g,w.y,w.dz,B,R,lr,wd,ib1,ib2);
    gemv(bh,net.gw.t,w.dz,w.dh2,G*R,H,true);CUDA_OK(cudaMemcpy(w.dskip,w.dh2,H*sizeof(float),cudaMemcpyDeviceToDevice));leaky_residual_back<<<blocks(H),THREADS>>>(w.dh2,w.h2,w.h1,H);
    gemv(bh,net.w2.t,w.dh2,w.dh1,H,H,true);add_vector<<<blocks(H),THREADS>>>(w.dh1,w.dskip,H);leaky_relu_back<<<blocks(H),THREADS>>>(w.dh1,w.h1,H);
    adam_outer<<<blocks(net.gw.n),THREADS>>>(net.gw.t,net.gw.m,net.gw.v,w.dz,w.h2,G*R,H,lr,wd,ib1,ib2);adam_vec<<<blocks(net.gb.n),THREADS>>>(net.gb.t,net.gb.m,net.gb.v,w.dz,net.gb.n,lr,0,ib1,ib2);
    adam_outer<<<blocks(net.w2.n),THREADS>>>(net.w2.t,net.w2.m,net.w2.v,w.dh2,w.h1,H,H,lr,wd,ib1,ib2);adam_vec<<<blocks(net.b2.n),THREADS>>>(net.b2.t,net.b2.m,net.b2.v,w.dh2,H,lr,0,ib1,ib2);
    adam_outer<<<blocks(net.w1.n),THREADS>>>(net.w1.t,net.w1.m,net.w1.v,w.dh1,w.x,H,I,lr,wd,ib1,ib2);adam_vec<<<blocks(net.b1.n),THREADS>>>(net.b1.t,net.b1.m,net.b1.v,w.dh1,H,lr,0,ib1,ib2);
    if(step%cfg->log_every==0||step==cfg->steps){vs=evaluate(bh,net,w,h,map.base,active,ag,NULL,val,feat,cfg->eval_batches,cfg->batch,cfg->seed+101);double elapsed=now()-t0;std::fprintf(stderr,"[methscope] upscale-factor-train: step %u/%u val_rmse=%.6f val_mae=%.6f elapsed=%.1fs steps/s=%.2f\n",step,cfg->steps,vs.rmse,vs.mae,elapsed,step/elapsed);}
  }
  Score ts=evaluate(bh,net,w,h,map.base,active,ag,NULL,test,feat,cfg->eval_batches,cfg->batch,cfg->seed+202);Score te={NAN,NAN,0},tv={NAN,NAN,0};if(!easy_pool.empty()){te=evaluate(bh,net,w,h,map.base,active,ag,&easy_pool,test,feat,cfg->eval_batches,cfg->batch,cfg->seed+303);tv=evaluate(bh,net,w,h,map.base,active,ag,&variable_pool,test,feat,cfg->eval_batches,cfg->batch,cfg->seed+404);}double elapsed=now()-t0;std::fprintf(stderr,"[methscope] upscale-factor-train: test_rmse=%.6f test_mae=%.6f n=%llu training_seconds=%.1f\n",ts.rmse,ts.mae,(unsigned long long)ts.n,elapsed);if(!easy_pool.empty())std::fprintf(stderr,"[methscope] upscale-factor-train: test_homogeneous_rmse=%.6f mae=%.6f; test_variable_rmse=%.6f mae=%.6f\n",te.rmse,te.mae,tv.rmse,tv.mae);

  ModelHeader mh={};std::memcpy(mh.magic,"UPFAC3\0\0",8);mh.version=3;mh.patterns=G;mh.rank=R;mh.hidden=H;mh.n_input=I;mh.n_active=(uint32_t)active.size();mh.n_cells=h->n_cells;mh.n_train=train.size();mh.n_val=val.size();mh.n_test=test.size();mh.steps=cfg->steps;mh.batch=cfg->batch;mh.n_cpg=h->n_cpg;mh.seed=cfg->seed;mh.val_rmse=(float)vs.rmse;mh.val_mae=(float)vs.mae;mh.test_rmse=(float)ts.rmse;mh.test_mae=(float)ts.mae;
  mh.split_offset=sizeof(mh);mh.prep_offset=mh.split_offset+split.size();mh.active_offset=mh.prep_offset+(uint64_t)3*I*sizeof(float);mh.param_offset=mh.active_offset+(uint64_t)active.size()*(sizeof(uint32_t)+sizeof(uint16_t));mh.file_bytes=mh.param_offset+(uint64_t)(net.w1.n+net.b1.n+net.w2.n+net.b2.n+net.gw.n+net.gb.n+net.e.n+net.eb.n)*sizeof(float);
  FILE*f=std::fopen(cfg->model_path,"wb");if(!f)die_path("cannot create model",cfg->model_path);write_all(f,&mh,sizeof(mh),cfg->model_path);write_all(f,split.data(),split.size(),cfg->model_path);write_all(f,imp.data(),I*sizeof(float),cfg->model_path);write_all(f,mean.data(),I*sizeof(float),cfg->model_path);write_all(f,scale.data(),I*sizeof(float),cfg->model_path);write_all(f,active.data(),active.size()*sizeof(uint32_t),cfg->model_path);write_all(f,ag.data(),ag.size()*sizeof(uint16_t),cfg->model_path);
  write_device(f,net.w1.t,net.w1.n,cfg->model_path);write_device(f,net.b1.t,net.b1.n,cfg->model_path);write_device(f,net.w2.t,net.w2.n,cfg->model_path);write_device(f,net.b2.t,net.b2.n,cfg->model_path);write_device(f,net.gw.t,net.gw.n,cfg->model_path);write_device(f,net.gb.t,net.gb.n,cfg->model_path);write_device(f,net.e.t,net.e.n,cfg->model_path);write_device(f,net.eb.t,net.eb.n,cfg->model_path);if(std::fclose(f))die_path("error closing model",cfg->model_path);
  char mp[4096];if(std::snprintf(mp,sizeof(mp),"%s.tsv",cfg->model_path)>=(int)sizeof(mp))die("model path too long");f=std::fopen(mp,"w");if(!f)die_path("cannot create model manifest",mp);
  std::fprintf(f,
    "format\tUPFAC3\nactivation\tleaky_relu_0.01_residual_block\nsidecar\t%s\npatterns\t%u\nrank\t%d\nhidden\t%d\n"
    "input_features\t%d\nactive_cpgs\t%zu\ntrain_cells\t%zu\n"
    "validation_cells\t%zu\ntest_cells\t%zu\nsteps\t%u\nbatch\t%u\n"
    "seed\t%llu\nhomogeneous_groups\t%s\nhomogeneous_fraction\t%.9g\n"
    "homogeneous_cpgs\t%zu\nvariable_cpgs\t%zu\n"
    "validation_rmse\t%.9g\nvalidation_mae\t%.9g\n"
    "test_rmse\t%.9g\ntest_mae\t%.9g\n"
    "test_homogeneous_rmse\t%.9g\ntest_homogeneous_mae\t%.9g\n"
    "test_variable_rmse\t%.9g\ntest_variable_mae\t%.9g\n"
    "training_seconds\t%.3f\n"
    "parameter_order\tW1,b1,W2,b2,group_W,group_b,cpg_embedding,cpg_bias\n",
    cfg->data_path,G,R,H,I,active.size(),train.size(),val.size(),test.size(),
    cfg->steps,cfg->batch,(unsigned long long)cfg->seed,
    cfg->homogeneous_groups?cfg->homogeneous_groups:"",
    cfg->homogeneous_fraction,easy_pool.size(),variable_pool.size(),
    vs.rmse,vs.mae,ts.rmse,ts.mae,te.rmse,te.mae,tv.rmse,tv.mae,elapsed);
  std::fclose(f);
  std::fprintf(stderr,"[methscope] upscale-factor-train: wrote %s (%llu bytes) and %s\n",cfg->model_path,(unsigned long long)mh.file_bytes,mp);
  BLAS_OK(cublasDestroy(bh));free_param(&net.w1);free_param(&net.b1);free_param(&net.w2);free_param(&net.b2);free_param(&net.gw);free_param(&net.gb);free_param(&net.e);free_param(&net.eb);cudaFree(w.x);cudaFree(w.h1);cudaFree(w.h2);cudaFree(w.z);cudaFree(w.dh2);cudaFree(w.dh1);cudaFree(w.dskip);cudaFree(w.dz);cudaFree(w.y);cudaFree(w.id);cudaFree(w.g);cudaFree(w.met);unmap_read(&map);return 0;
}
