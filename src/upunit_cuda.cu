// SPDX-License-Identifier: AGPL-3.0-or-later
/*
 * Independent processing-unit CUDA trainer for UPDEC2.
 *
 * The input is deterministic: every MRMP contributes beta plus either
 * log1p(observed-CpG count) or a missing indicator. An optional frozen
 * two-layer 512-dimensional trunk is shared by all unit-local heads.
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
#include <unistd.h>
#include <vector>
#include <thread>
#include <time.h>
#include "updec2.h"
#include "upunit_cuda.h"

static const int THREADS=256;
static void die(const char*m){std::fprintf(stderr,"[methscope] upscale-train: %s\n",m);std::exit(1);}
static void diep(const char*m,const char*p){std::fprintf(stderr,"[methscope] upscale-train: %s: %s\n",m,p);std::exit(1);}
static void cf(const char*c,cudaError_t e,const char*f,int l){std::fprintf(stderr,"[methscope] upscale-train CUDA: %s failed at %s:%d: %s\n",c,f,l,cudaGetErrorString(e));std::exit(1);}
#define CU(x) do{cudaError_t e=(x);if(e!=cudaSuccess)cf(#x,e,__FILE__,__LINE__);}while(0)
static void bf(const char*c,cublasStatus_t s,const char*f,int l){std::fprintf(stderr,"[methscope] upscale-train cuBLAS: %s failed at %s:%d (%d)\n",c,f,l,(int)s);std::exit(1);}
#define BL(x) do{cublasStatus_t s=(x);if(s!=CUBLAS_STATUS_SUCCESS)bf(#x,s,__FILE__,__LINE__);}while(0)
static int blocks(size_t n){size_t b=(n+THREADS-1)/THREADS;return(int)(b>4096?4096:(b?b:1));}
static float*df(size_t n){float*p=NULL;CU(cudaMalloc((void**)&p,(n?n:1)*sizeof(float)));return p;}
static uint32_t*du(size_t n){uint32_t*p=NULL;CU(cudaMalloc((void**)&p,(n?n:1)*sizeof(uint32_t)));return p;}

struct MsurHeader{char magic[8];uint32_t version,n_cells,n_reps,n_patterns;uint64_t n_cpg;uint32_t sampled_per_cell,flags;uint64_t groups_offset,truth_offset,records_offset,record_bytes;};
static_assert(sizeof(MsurHeader)==72,"MSURAW2 header");
#pragma pack(push,1)
struct MsuiHeader{char magic[8];uint32_t version,flags,pattern_length,target_unit_cpgs;uint32_t n_units,n_real_memberships,n_pna_units,reserved32;uint64_t n_cpg,n_real_cpg,n_pna_cpg;uint64_t unit_offset,cpg_offset,membership_offset,file_bytes;uint64_t pattern_checksum,reserved0,reserved1,reserved2;};
struct MsuiUnit{uint64_t output_offset;uint32_t first_membership,membership_count,cpg_count,flags;};
struct MsuiMembership{uint64_t pattern_key,output_offset;uint32_t count,unit;};
struct CheckHeader{char magic[8];uint32_t version,unit,mode,rank,activation,best_step,cpg_count,input_dim;uint64_t index_checksum,run_checksum,param_floats;float best_mae;uint32_t reserved;};
struct UpfacHeader{char magic[8];uint32_t version,patterns,rank,hidden,n_input,n_active,n_cells,n_train,n_val,n_test,steps,batch;uint64_t n_cpg,seed;float val_rmse,val_mae,test_rmse,test_mae;uint64_t split_offset,prep_offset,active_offset,param_offset,file_bytes;};
#pragma pack(pop)
static_assert(sizeof(MsuiHeader)==128,"MSUIDX1 header");
static_assert(sizeof(MsuiUnit)==24,"MSUIDX1 unit");
static_assert(sizeof(MsuiMembership)==24,"MSUIDX1 membership");
static_assert(sizeof(CheckHeader)==72,"unit checkpoint header");
static_assert(sizeof(UpfacHeader)==128,"UPFAC3 header");

struct Map{int fd;size_t n;uint8_t*p;};
static Map mapread(const char*path){Map m={-1,0,NULL};m.fd=open(path,O_RDONLY);if(m.fd<0)diep("cannot open",path);struct stat s;if(fstat(m.fd,&s)||s.st_size<0)diep("cannot stat",path);m.n=(size_t)s.st_size;m.p=(uint8_t*)mmap(NULL,m.n,PROT_READ,MAP_SHARED,m.fd,0);if(m.p==MAP_FAILED)diep("cannot mmap",path);return m;}
static void unmap(Map&m){munmap(m.p,m.n);close(m.fd);}
static bool range(uint64_t o,uint64_t n,size_t z){return o<=z&&n<=(uint64_t)z-o;}
static uint64_t fnv(uint64_t h,const void*vp,size_t n){const uint8_t*p=(const uint8_t*)vp;for(size_t i=0;i<n;++i){h^=p[i];h*=UINT64_C(1099511628211);}return h;}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

struct Pcg{uint64_t state,inc;};
static uint32_t rnd(Pcg*p){uint64_t o=p->state;p->state=o*UINT64_C(6364136223846793005)+p->inc;uint32_t x=(uint32_t)(((o>>18)^o)>>27),r=(uint32_t)(o>>59);return(x>>r)|(x<<((-r)&31));}
static void seed(Pcg*p,uint64_t s){p->state=0;p->inc=UINT64_C(1442695040888963407);rnd(p);p->state+=s;rnd(p);}
__device__ static uint64_t hash64(uint64_t x){x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
__device__ static float sig(float x){if(x>=0){float z=expf(-x);return 1/(1+z);}float z=expf(x);return z/(1+z);}
__global__ static void randinit(float*x,size_t n,float scale,uint64_t s){for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){uint64_t z=hash64(s+q*UINT64_C(0x9e3779b97f4a7c15));float u=(float)((z>>40)+.5)*(1.0f/16777216.0f);x[q]=(2*u-1)*scale;}}
__global__ static void addact(float*x,const float*b,int n,int act){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x){float v=x[q]+b[q];x[q]=(act&&v<0)?0.01f*v:v;}}
__global__ static void trunk_act(float*x,const float*b,size_t n,int H){for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){float v=x[q]+b[q%H];x[q]=v>0?v:.01f*v;}}
__global__ static void trunk_residual(float*x,const float*b,const float*skip,size_t n,int H){for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){float v=x[q]+b[q%H];x[q]=skip[q]+(v>0?v:.01f*v);}}
__global__ static void actback(float*d,const float*z,int n,int act){if(!act)return;for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x)if(!(z[q]>0))d[q]*=.01f;}

struct Metric{double se,ae;unsigned long long n;};
__global__ static void factor_metric(const float*z,const float*E,const float*b,const uint32_t*id,const float*y,int B,int R,Metric*out){double se=0,ae=0;unsigned long long n=0;for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<B;q+=blockDim.x*gridDim.x){uint32_t c=id[q];float s=b[c];for(int r=0;r<R;++r)s+=E[(size_t)c*R+r]*z[r];float d=sig(s)-y[q];se+=(double)d*d;ae+=fabs((double)d);++n;}__shared__ double ss[THREADS],sa[THREADS];__shared__ unsigned long long sn[THREADS];ss[threadIdx.x]=se;sa[threadIdx.x]=ae;sn[threadIdx.x]=n;__syncthreads();for(int d=blockDim.x/2;d;d>>=1){if(threadIdx.x<d){ss[threadIdx.x]+=ss[threadIdx.x+d];sa[threadIdx.x]+=sa[threadIdx.x+d];sn[threadIdx.x]+=sn[threadIdx.x+d];}__syncthreads();}if(!threadIdx.x){atomicAdd(&out->se,ss[0]);atomicAdd(&out->ae,sa[0]);atomicAdd(&out->n,sn[0]);}}
__global__ static void direct_metric(const float*x,const float*W,const float*b,const uint32_t*id,const float*y,int B,int I,Metric*out){double se=0,ae=0;unsigned long long n=0;for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<B;q+=blockDim.x*gridDim.x){uint32_t c=id[q];float s=b[c];for(int j=0;j<I;++j)s+=W[(size_t)c*I+j]*x[j];float d=sig(s)-y[q];se+=(double)d*d;ae+=fabs((double)d);++n;}__shared__ double ss[THREADS],sa[THREADS];__shared__ unsigned long long sn[THREADS];ss[threadIdx.x]=se;sa[threadIdx.x]=ae;sn[threadIdx.x]=n;__syncthreads();for(int d=blockDim.x/2;d;d>>=1){if(threadIdx.x<d){ss[threadIdx.x]+=ss[threadIdx.x+d];sa[threadIdx.x]+=sa[threadIdx.x+d];sn[threadIdx.x]+=sn[threadIdx.x+d];}__syncthreads();}if(!threadIdx.x){atomicAdd(&out->se,ss[0]);atomicAdd(&out->ae,sa[0]);atomicAdd(&out->n,sn[0]);}}

__global__ static void factor_back(const float*z,float*E,float*Em,float*Ev,float*b,float*bm,float*bv,const uint32_t*id,const float*y,float*dz,int B,int R,float lr,float wd,float ib1,float ib2){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<B;q+=blockDim.x*gridDim.x){uint32_t c=id[q];float s=b[c];for(int r=0;r<R;++r)s+=E[(size_t)c*R+r]*z[r];float p=sig(s),dl=2*(p-y[q])*p*(1-p)/B;for(int r=0;r<R;++r){size_t k=(size_t)c*R+r;float old=E[k];atomicAdd(dz+r,dl*old);float g=dl*z[r]+wd*old,mm=.9f*Em[k]+.1f*g,vv=.999f*Ev[k]+.001f*g*g;Em[k]=mm;Ev[k]=vv;E[k]=old-lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}float old=b[c],g=dl,mm=.9f*bm[c]+.1f*g,vv=.999f*bv[c]+.001f*g*g;bm[c]=mm;bv[c]=vv;b[c]=old-lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}}
__global__ static void direct_back(const float*x,float*W,float*Wm,float*Wv,float*b,float*bm,float*bv,const uint32_t*id,const float*y,int B,int I,float lr,float wd,float ib1,float ib2){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<B;q+=blockDim.x*gridDim.x){uint32_t c=id[q];float s=b[c];for(int j=0;j<I;++j)s+=W[(size_t)c*I+j]*x[j];float p=sig(s),dl=2*(p-y[q])*p*(1-p)/B;for(int j=0;j<I;++j){size_t k=(size_t)c*I+j;float old=W[k],g=dl*x[j]+wd*old,mm=.9f*Wm[k]+.1f*g,vv=.999f*Wv[k]+.001f*g*g;Wm[k]=mm;Wv[k]=vv;W[k]=old-lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}float old=b[c],g=dl,mm=.9f*bm[c]+.1f*g,vv=.999f*bv[c]+.001f*g*g;bm[c]=mm;bv[c]=vv;b[c]=old-lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}}
__global__ static void adam_outer(float*t,float*m,float*v,const float*go,const float*in,int O,int I,float lr,float wd,float ib1,float ib2){size_t n=(size_t)O*I;for(size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=(size_t)blockDim.x*gridDim.x){int o=q/I,j=q%I;float g=go[o]*in[j]+wd*t[q],mm=.9f*m[q]+.1f*g,vv=.999f*v[q]+.001f*g*g;m[q]=mm;v[q]=vv;t[q]-=lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}}
__global__ static void adam_vec(float*t,float*m,float*v,const float*g,int n,float lr,float wd,float ib1,float ib2){for(int q=blockIdx.x*blockDim.x+threadIdx.x;q<n;q+=blockDim.x*gridDim.x){float z=g[q]+wd*t[q],mm=.9f*m[q]+.1f*z,vv=.999f*v[q]+.001f*z*z;m[q]=mm;v[q]=vv;t[q]-=lr*(mm*ib1)/(sqrtf(vv*ib2)+1e-8f);}}

struct Param{float*t,*m,*v;size_t n;};
static Param prand(size_t n,float sc,uint64_t s){Param p={df(n),df(n),df(n),n};randinit<<<blocks(n),THREADS>>>(p.t,n,sc,s);CU(cudaMemset(p.m,0,n*4));CU(cudaMemset(p.v,0,n*4));return p;}
static Param pzero(size_t n){Param p={df(n),df(n),df(n),n};CU(cudaMemset(p.t,0,n*4));CU(cudaMemset(p.m,0,n*4));CU(cudaMemset(p.v,0,n*4));return p;}
static Param pcopy(const std::vector<float>&x){Param p={df(x.size()),df(x.size()),df(x.size()),x.size()};CU(cudaMemcpy(p.t,x.data(),x.size()*4,cudaMemcpyHostToDevice));CU(cudaMemset(p.m,0,x.size()*4));CU(cudaMemset(p.v,0,x.size()*4));return p;}
static void pfree(Param&p){cudaFree(p.t);cudaFree(p.m);cudaFree(p.v);}

static bool observed(const uint32_t*a,uint32_t n,uint32_t x){return std::binary_search(a,a+n,x);}
static uint32_t targets(const MsurHeader*h,const uint8_t*base,const uint32_t*cpg,uint64_t begin,uint32_t O,uint32_t cell,uint32_t rep,uint64_t start,uint32_t want,std::vector<uint32_t>&id,std::vector<float>&y){
  size_t row=(size_t)rep*h->n_cells+cell;const uint8_t*rec=base+h->records_offset+row*h->record_bytes;const uint32_t*sel=(const uint32_t*)(rec+(size_t)h->n_patterns*8);const uint16_t*truth=(const uint16_t*)(base+h->truth_offset)+(size_t)cell*h->n_cpg;uint32_t n=0;uint64_t seen=0,q=start%O;
  while(n<want&&seen<O){uint32_t pos=cpg[begin+q];uint16_t v=truth[pos];if(v!=UINT16_MAX&&!observed(sel,h->sampled_per_cell,pos)){id[n]=(uint32_t)q;y[n]=(float)v/65534.0f;++n;}if(++q==O)q=0;++seen;}return n;
}
static void gemv(cublasHandle_t bh,const float*w,const float*x,float*y,int O,int I){const float one=1,zero=0;BL(cublasSgemv(bh,CUBLAS_OP_T,I,O,&one,w,I,x,1,&zero,y,1));}
static void trunk_gemm(cublasHandle_t bh,const float*w,const float*x,float*y,int O,int I,int rows){const float one=1,zero=0;BL(cublasSgemm(bh,CUBLAS_OP_T,CUBLAS_OP_N,O,rows,I,&one,w,I,x,I,&zero,y,O));}
static void writeall(FILE*f,const void*p,size_t n,const char*path){if(n&&fwrite(p,1,n,f)!=n)diep("write failed",path);}
static void ckpath(char*p,size_t n,const char*d,uint32_t u){if(std::snprintf(p,n,"%s/unit_%06u.upuck",d,u)>=(int)n)die("work path too long");}
static std::vector<uint8_t> unit_selection(const char*path,uint32_t n,uint32_t*count){
  std::vector<uint8_t> selected(n,path?0:1);*count=path?0:n;if(!path)return selected;FILE*f=fopen(path,"r");if(!f)diep("cannot open --pilot-units",path);uint32_t u;while(fscanf(f,"%u",&u)==1){if(u>=n)die("--pilot-units contains an out-of-range unit");if(selected[u])die("--pilot-units contains a duplicate unit");selected[u]=1;++*count;}if(!feof(f))die("--pilot-units must contain one integer unit ID per line");fclose(f);if(!*count)die("--pilot-units is empty");return selected;
}

struct EvalRow{size_t row;std::vector<uint32_t>id;std::vector<float>y;};
static std::vector<EvalRow> make_eval(const MsurHeader*h,const uint8_t*base,const uint32_t*cpg,const MsuiUnit&u,const std::vector<uint32_t>&cells,uint32_t nr,uint32_t batch,uint64_t s){
  Pcg r;seed(&r,s);std::vector<EvalRow> out;uint32_t tries=0,maxtries=std::max(32u,nr*32);while(out.size()<nr&&tries++<maxtries){uint32_t cell=cells[rnd(&r)%cells.size()],rep=rnd(&r)%h->n_reps;EvalRow e;e.row=(size_t)rep*h->n_cells+cell;e.id.resize(batch);e.y.resize(batch);uint32_t want=std::min(batch,u.cpg_count),B=targets(h,base,cpg,u.output_offset,u.cpg_count,cell,rep,rnd(&r),want,e.id,e.y);e.id.resize(B);e.y.resize(B);if(B)out.push_back(std::move(e));}return out;
}

struct Net{Param A,a,E,b;bool direct;int I,R,O;};
static void destroy(Net&n){if(!n.direct){pfree(n.A);pfree(n.a);}pfree(n.E);pfree(n.b);}
static std::vector<float> flatten(const Net&n){size_t z=(n.direct?(size_t)n.O*n.I:(size_t)n.R*n.I+n.R+(size_t)n.O*n.R)+n.O;std::vector<float>x(z);size_t q=0;auto cp=[&](const Param&p){CU(cudaMemcpy(x.data()+q,p.t,p.n*4,cudaMemcpyDeviceToHost));q+=p.n;};if(!n.direct){cp(n.A);cp(n.a);}cp(n.E);cp(n.b);return x;}
static Net make_net(bool direct,int I,int R,int O,const std::vector<float>&bias,uint64_t s){
  Net n={};n.direct=direct;n.I=I;n.R=R;n.O=O;
  if(direct){n.E=prand((size_t)O*I,std::sqrt(6.0f/(I+1)),s^11);n.b=pcopy(bias);}
  else{n.A=prand((size_t)R*I,std::sqrt(6.0f/(I+R)),s^13);n.a=pzero(R);n.E=prand((size_t)O*R,.02f,s^17);n.b=pcopy(bias);}
  return n;
}

struct Work{float*z,*dz,*y;uint32_t*id;Metric*met;};
static double eval(cublasHandle_t bh,const Net&n,Work&w,const float*dX,const std::vector<EvalRow>&rows,int act){
  Metric tot={0,0,0};for(const auto&er:rows){int B=er.id.size();CU(cudaMemcpy(w.id,er.id.data(),B*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(w.y,er.y.data(),B*4,cudaMemcpyHostToDevice));CU(cudaMemset(w.met,0,sizeof(Metric)));const float*x=dX+er.row*n.I;if(n.direct)direct_metric<<<blocks(B),THREADS>>>(x,n.E.t,n.b.t,w.id,w.y,B,n.I,w.met);else{gemv(bh,n.A.t,x,w.z,n.R,n.I);addact<<<blocks(n.R),THREADS>>>(w.z,n.a.t,n.R,act);factor_metric<<<blocks(B),THREADS>>>(w.z,n.E.t,n.b.t,w.id,w.y,B,n.R,w.met);}Metric m;CU(cudaMemcpy(&m,w.met,sizeof(m),cudaMemcpyDeviceToHost));tot.se+=m.se;tot.ae+=m.ae;tot.n+=m.n;}return tot.n?tot.ae/tot.n:NAN;
}

static bool load_checkpoint(const char*path,uint32_t ui,uint64_t sum,uint64_t runsum,int mode,int rank,int act,int O,int I,std::vector<float>&par,float&mae,uint32_t&step){
  FILE*f=fopen(path,"rb");if(!f)return false;CheckHeader h;if(fread(&h,1,sizeof(h),f)!=sizeof(h)||memcmp(h.magic,"UPUCK1",6)||h.version!=1||h.unit!=ui||h.index_checksum!=sum||h.run_checksum!=runsum||h.mode!=(uint32_t)mode||h.rank!=(uint32_t)rank||h.activation!=(uint32_t)act||h.cpg_count!=(uint32_t)O||h.input_dim!=(uint32_t)I||h.param_floats>SIZE_MAX/4){fclose(f);return false;}par.resize(h.param_floats);if(fread(par.data(),4,par.size(),f)!=par.size()||fgetc(f)!=EOF){fclose(f);par.clear();return false;}fclose(f);mae=h.best_mae;step=h.best_step;return true;
}
static void save_checkpoint(const char*path,uint32_t ui,uint64_t sum,uint64_t runsum,int mode,int rank,int act,int O,int I,const std::vector<float>&par,float mae,uint32_t step){
  char tmp[4096];if(std::snprintf(tmp,sizeof(tmp),"%s.tmp",path)>=(int)sizeof(tmp))die("checkpoint path too long");FILE*f=fopen(tmp,"wb");if(!f)diep("cannot create checkpoint",tmp);CheckHeader h={};memcpy(h.magic,"UPUCK1",6);h.version=1;h.unit=ui;h.mode=mode;h.rank=rank;h.activation=act;h.best_step=step;h.cpg_count=O;h.input_dim=I;h.index_checksum=sum;h.run_checksum=runsum;h.param_floats=par.size();h.best_mae=mae;writeall(f,&h,sizeof(h),tmp);writeall(f,par.data(),par.size()*4,tmp);if(fclose(f)||rename(tmp,path))diep("cannot finalize checkpoint",path);
}

extern "C" int ms_upunit_cuda_available(void){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;}
extern "C" int ms_upunit_train_cuda(const ms_upunit_config_t*c){
  Map data=mapread(c->data_path),idx=mapread(c->index_path);if(data.n<sizeof(MsurHeader)||idx.n<sizeof(MsuiHeader))die("truncated training input");
  const MsurHeader*h=(const MsurHeader*)data.p;const MsuiHeader*ih=(const MsuiHeader*)idx.p;
  if(memcmp(h->magic,"MSURAW2\0",8)||h->version!=2||!(h->flags&1)||!h->truth_offset)die("training requires embedded-truth MSURAW2");
  if(memcmp(ih->magic,"MSUIDX1",7)||ih->version!=1||h->n_cpg!=ih->n_cpg)die("training requires matching MSUIDX1");
  uint32_t P=c->patterns?c->patterns:h->n_patterns;if(!P||P>h->n_patterns)die("invalid pattern count");
  if(c->feature_mode>MS_UPFEATURE_BETA)die("invalid feature mode");
  int F=(c->feature_mode==MS_UPFEATURE_BETA?1:2)*(int)P,I=F,H=0;Map trunk={-1,0,NULL};const UpfacHeader*th=NULL;const float*trunk_par=NULL;uint64_t trunk_floats=0,trunk_checksum=0;
  if(c->trunk_path){
    trunk=mapread(c->trunk_path);if(trunk.n<sizeof(UpfacHeader))die("truncated UPFAC3 trunk");th=(const UpfacHeader*)trunk.p;
    if(memcmp(th->magic,"UPFAC3\0\0",8)||th->version!=3||th->patterns!=P||th->n_input!=(uint32_t)F||!th->hidden)die("incompatible UPFAC3 trunk");
    H=(int)th->hidden;I=H;trunk_floats=(uint64_t)H*F+H+(uint64_t)H*H+H;
    if(!range(th->prep_offset,(uint64_t)3*F*4,trunk.n)||!range(th->param_offset,trunk_floats*4,trunk.n))die("truncated UPFAC3 preprocessing or trunk parameters");
    trunk_par=(const float*)(trunk.p+th->param_offset);
    trunk_checksum=fnv(UINT64_C(1469598103934665603),trunk.p+th->prep_offset,(size_t)3*F*4);
    trunk_checksum=fnv(trunk_checksum,trunk_par,(size_t)trunk_floats*4);
  }
  uint64_t rows=(uint64_t)h->n_cells*h->n_reps,recend=h->records_offset+rows*h->record_bytes,truthbytes=(uint64_t)h->n_cells*h->n_cpg*2;
  if(recend>data.n||!range(h->truth_offset,truthbytes,data.n)||ih->file_bytes>idx.n||!range(ih->unit_offset,(uint64_t)ih->n_units*sizeof(MsuiUnit),idx.n)||!range(ih->cpg_offset,ih->n_cpg*4,idx.n)||!range(ih->membership_offset,(uint64_t)ih->n_real_memberships*sizeof(MsuiMembership),idx.n))die("truncated training payload");
  const MsuiUnit*units=(const MsuiUnit*)(idx.p+ih->unit_offset);const uint32_t*cpg=(const uint32_t*)(idx.p+ih->cpg_offset);const MsuiMembership*members=(const MsuiMembership*)(idx.p+ih->membership_offset);
  uint64_t isum=fnv(UINT64_C(1469598103934665603),idx.p,ih->file_bytes);
  uint64_t runsum=fnv(UINT64_C(1469598103934665603),h,sizeof(*h));
  runsum=fnv(runsum,&data.n,sizeof(data.n));runsum=fnv(runsum,&c->patterns,sizeof(c->patterns));
  runsum=fnv(runsum,&c->feature_mode,sizeof(c->feature_mode));runsum=fnv(runsum,&trunk_checksum,sizeof(trunk_checksum));
  runsum=fnv(runsum,&c->pure_bottleneck,sizeof(c->pure_bottleneck));runsum=fnv(runsum,&c->mixed_bottleneck,sizeof(c->mixed_bottleneck));
  runsum=fnv(runsum,&c->mixed_direct,sizeof(c->mixed_direct));runsum=fnv(runsum,&c->activation,sizeof(c->activation));
  runsum=fnv(runsum,&c->min_steps,sizeof(c->min_steps));runsum=fnv(runsum,&c->max_steps,sizeof(c->max_steps));
  runsum=fnv(runsum,&c->eval_every,sizeof(c->eval_every));runsum=fnv(runsum,&c->patience,sizeof(c->patience));
  runsum=fnv(runsum,&c->batch,sizeof(c->batch));runsum=fnv(runsum,&c->eval_rows,sizeof(c->eval_rows));
  runsum=fnv(runsum,&c->seed,sizeof(c->seed));runsum=fnv(runsum,&c->learning_rate,sizeof(c->learning_rate));runsum=fnv(runsum,&c->weight_decay,sizeof(c->weight_decay));
  runsum=fnv(runsum,&c->adaptive_rank,sizeof(c->adaptive_rank));runsum=fnv(runsum,&c->homogeneous_rank,sizeof(c->homogeneous_rank));
  std::vector<uint32_t>cells(h->n_cells);for(uint32_t i=0;i<h->n_cells;++i)cells[i]=i;Pcg sr;seed(&sr,c->seed);for(uint32_t q=h->n_cells;q>1;--q)std::swap(cells[q-1],cells[rnd(&sr)%q]);size_t nt=h->n_cells*70/100,nv=h->n_cells*15/100;if(!nt||!nv||nt+nv>=h->n_cells)die("too few source cells for split");std::vector<uint32_t>train(cells.begin(),cells.begin()+nt),val(cells.begin()+nt,cells.begin()+nt+nv),test(cells.begin()+nt+nv,cells.end());
  std::vector<uint8_t>is_train(h->n_cells);for(uint32_t x:train)is_train[x]=1;
  std::vector<float>mean(F),scale(F);
  if(th){
    const float*prep=(const float*)(trunk.p+th->prep_offset);memcpy(mean.data(),prep+F,(size_t)F*4);memcpy(scale.data(),prep+2*F,(size_t)F*4);
  }else{
    std::vector<double>sum(F),ss(F);std::vector<uint64_t>cnt(F);uint64_t tr=(uint64_t)train.size()*h->n_reps;
    for(uint32_t r=0;r<h->n_reps;++r)for(uint32_t cell:train){const uint8_t*rec=data.p+h->records_offset+((size_t)r*h->n_cells+cell)*h->record_bytes;const float*b=(const float*)rec;const uint32_t*n=(const uint32_t*)(rec+(size_t)h->n_patterns*4);for(uint32_t p=0;p<P;++p){uint32_t j=c->feature_mode==MS_UPFEATURE_BETA?p:2*p;if(std::isfinite(b[p])){sum[j]+=b[p];cnt[j]++;}if(c->feature_mode!=MS_UPFEATURE_BETA){double a=c->feature_mode==MS_UPFEATURE_COUNT?std::log1p((double)n[p]):(std::isfinite(b[p])?0.0:1.0);sum[j+1]+=a;cnt[j+1]++;}}}
    for(uint32_t p=0;p<P;++p){uint32_t j=c->feature_mode==MS_UPFEATURE_BETA?p:2*p;if(!cnt[j])die("one MRMP is always missing in training");mean[j]=(float)(sum[j]/cnt[j]);if(c->feature_mode!=MS_UPFEATURE_BETA)mean[j+1]=(float)(sum[j+1]/cnt[j+1]);}
    for(uint32_t r=0;r<h->n_reps;++r)for(uint32_t cell:train){const uint8_t*rec=data.p+h->records_offset+((size_t)r*h->n_cells+cell)*h->record_bytes;const float*b=(const float*)rec;const uint32_t*n=(const uint32_t*)(rec+(size_t)h->n_patterns*4);for(uint32_t p=0;p<P;++p){uint32_t j=c->feature_mode==MS_UPFEATURE_BETA?p:2*p;if(std::isfinite(b[p])){double d=b[p]-mean[j];ss[j]+=d*d;}if(c->feature_mode!=MS_UPFEATURE_BETA){double a=c->feature_mode==MS_UPFEATURE_COUNT?std::log1p((double)n[p]):(std::isfinite(b[p])?0.0:1.0),d=a-mean[j+1];ss[j+1]+=d*d;}}}
    for(int j=0;j<F;++j)scale[j]=(float)std::sqrt(std::max(ss[j]/tr,1e-12));
  }
  for(int j=0;j<F;++j)if(!std::isfinite(mean[j])||!std::isfinite(scale[j])||!(scale[j]>0))die("invalid feature preprocessing");
  std::vector<float>X((size_t)rows*F);for(size_t r=0;r<rows;++r){const uint8_t*rec=data.p+h->records_offset+r*h->record_bytes;const float*b=(const float*)rec;const uint32_t*n=(const uint32_t*)(rec+(size_t)h->n_patterns*4);float*x=X.data()+r*F;for(uint32_t p=0;p<P;++p){uint32_t j=c->feature_mode==MS_UPFEATURE_BETA?p:2*p;x[j]=std::isfinite(b[p])?(b[p]-mean[j])/scale[j]:0;if(c->feature_mode!=MS_UPFEATURE_BETA){double a=c->feature_mode==MS_UPFEATURE_COUNT?std::log1p((double)n[p]):(std::isfinite(b[p])?0.0:1.0);x[j+1]=(a-mean[j+1])/scale[j+1];}}}
  CU(cudaSetDevice(c->device));cudaDeviceProp prop;CU(cudaGetDeviceProperties(&prop,c->device));std::fprintf(stderr,"[methscope] upscale-train: GPU %s %.0f MiB; cells=%u reps=%u patterns=%u features=%s external=%d trunk=%d unit_input=%d units=%u split=%zu/%zu/%zu\n",prop.name,(double)prop.totalGlobalMem/1048576,h->n_cells,h->n_reps,P,c->feature_mode==MS_UPFEATURE_COUNT?"beta+count":c->feature_mode==MS_UPFEATURE_MISSING?"beta+missing":"beta-only",F,H,I,ih->n_units,train.size(),val.size(),test.size());
  cublasHandle_t bh;BL(cublasCreate(&bh));float*dRaw=df(X.size());CU(cudaMemcpy(dRaw,X.data(),X.size()*4,cudaMemcpyHostToDevice));X.clear();X.shrink_to_fit();float*dX=dRaw;
  if(th){float*dw1=df((size_t)H*F),*db1=df(H),*dw2=df((size_t)H*H),*db2=df(H),*dh1=df((size_t)rows*H);dX=df((size_t)rows*H);CU(cudaMemcpy(dw1,trunk_par,(size_t)H*F*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(db1,trunk_par+(size_t)H*F,(size_t)H*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(dw2,trunk_par+(size_t)H*F+H,(size_t)H*H*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(db2,trunk_par+(size_t)H*F+H+(size_t)H*H,(size_t)H*4,cudaMemcpyHostToDevice));double tt=now();trunk_gemm(bh,dw1,dRaw,dh1,H,F,(int)rows);trunk_act<<<blocks((size_t)rows*H),THREADS>>>(dh1,db1,(size_t)rows*H,H);trunk_gemm(bh,dw2,dh1,dX,H,H,(int)rows);trunk_residual<<<blocks((size_t)rows*H),THREADS>>>(dX,db2,dh1,(size_t)rows*H,H);CU(cudaDeviceSynchronize());std::fprintf(stderr,"[methscope] upscale-train: precomputed shared trunk for %llu rows in %.2fs\n",(unsigned long long)rows,now()-tt);cudaFree(dw1);cudaFree(db1);cudaFree(dw2);cudaFree(db2);cudaFree(dh1);cudaFree(dRaw);}
  uint32_t maxR=std::max(c->pure_bottleneck,c->mixed_bottleneck);
  Work w={df(maxR),df(maxR),df(c->batch),du(c->batch),NULL};CU(cudaMalloc((void**)&w.met,sizeof(Metric)));
  std::vector<uint32_t>bid(c->batch);std::vector<float>by(c->batch);const uint16_t*truth=(const uint16_t*)(data.p+h->truth_offset);
  double bias_t0=now();std::vector<float>all_bias(ih->n_cpg,0.0f);std::vector<uint16_t>all_count(ih->n_cpg,0);
  unsigned nth=std::min(8u,std::max(1u,(unsigned)std::thread::hardware_concurrency()));std::vector<std::thread>workers;
  for(unsigned ti=0;ti<nth;++ti)workers.emplace_back([&,ti](){uint64_t beg=ih->n_cpg*ti/nth,end=ih->n_cpg*(ti+1)/nth;for(uint32_t cell:train){const uint16_t*t=truth+(size_t)cell*h->n_cpg;for(uint64_t pos=beg;pos<end;++pos){uint16_t v=t[pos];if(v!=UINT16_MAX){all_bias[pos]+=(float)v/65534.0f;all_count[pos]++;}}}for(uint64_t pos=beg;pos<end;++pos){double p=all_count[pos]?all_bias[pos]/all_count[pos]:.5;p=std::max(.01,std::min(.99,p));all_bias[pos]=std::log(p/(1-p));}});
  for(auto&t:workers)t.join();all_count.clear();all_count.shrink_to_fit();std::fprintf(stderr,"[methscope] upscale-train: prepared genome-wide train-cell biases with %u CPU workers in %.1fs\n",nth,now()-bias_t0);
  /* Simple homogeneity-aware per-unit rank: an all-0/all-1 pure unit is
   * trivially predictable, so it uses homogeneous_rank; every other factor
   * unit keeps its base bottleneck. */
  std::vector<uint32_t>unit_rank(ih->n_units);
  uint64_t all1key=0;for(uint32_t i=0;i<ih->pattern_length;++i)all1key=all1key*3+1;
  for(uint32_t ui=0;ui<ih->n_units;++ui){const MsuiUnit&u=units[ui];bool pure=(u.flags&1)!=0,pna=(u.flags&2)!=0,direct=c->mixed_direct&&!pure;uint32_t base=pna?c->mixed_bottleneck:(pure?c->pure_bottleneck:c->mixed_bottleneck);if(direct){unit_rank[ui]=0;continue;}uint32_t R=base;if(c->adaptive_rank&&!pna&&pure&&u.membership_count==1&&u.first_membership<ih->n_real_memberships&&(members[u.first_membership].pattern_key==0||members[u.first_membership].pattern_key==all1key))R=c->homogeneous_rank;if(R<1)R=1;unit_rank[ui]=R;}
  double train_t0=now();uint32_t selected_count=0;std::vector<uint8_t>selected=unit_selection(c->pilot_units_path,ih->n_units,&selected_count);std::vector<float>unit_mae(ih->n_units,NAN);std::vector<uint32_t>unit_step(ih->n_units);std::vector<uint64_t>unit_n(ih->n_units);
  uint32_t trained=0,resumed=0,processed=0;
  for(uint32_t ui=0;ui<ih->n_units;++ui){if(!selected[ui])continue;const MsuiUnit&u=units[ui];bool pure=(u.flags&1)!=0,pna=(u.flags&2)!=0,direct=c->mixed_direct&&!pure;int R=direct?0:(int)unit_rank[ui],O=u.cpg_count;char path[4096];ckpath(path,sizeof(path),c->work_dir,ui);std::vector<float>best;float bestmae=NAN;uint32_t beststep=0;
    std::vector<EvalRow>ev=make_eval(h,data.p,cpg,u,val,c->eval_rows,c->batch,c->seed+101+ui);if(ev.empty())die("could not form validation rows");for(const auto&er:ev)unit_n[ui]+=er.id.size();
    if(load_checkpoint(path,ui,isum,runsum,direct?0:1,R,c->activation,O,I,best,bestmae,beststep)){++resumed;unit_mae[ui]=bestmae;unit_step[ui]=beststep;++processed;continue;}
    std::vector<float>bias(O);for(int o=0;o<O;++o)bias[o]=all_bias[cpg[u.output_offset+o]];
    Net net=make_net(direct,I,R,O,bias,c->seed^((uint64_t)ui<<32));
    double v=eval(bh,net,w,dX,ev,c->activation);bestmae=(float)v;best=flatten(net);Pcg rr;seed(&rr,c->seed^ui^UINT64_C(0xd1b54a32d192ed03));uint32_t bad=0,step=0;
    for(step=1;step<=c->max_steps;++step){uint32_t cell=train[rnd(&rr)%train.size()],rep=rnd(&rr)%h->n_reps,want=std::min(c->batch,u.cpg_count);uint32_t B=targets(h,data.p,cpg,u.output_offset,u.cpg_count,cell,rep,rnd(&rr),want,bid,by);if(!B)die("could not form unit target batch");size_t row=(size_t)rep*h->n_cells+cell;const float*x=dX+row*I;CU(cudaMemcpy(w.id,bid.data(),B*4,cudaMemcpyHostToDevice));CU(cudaMemcpy(w.y,by.data(),B*4,cudaMemcpyHostToDevice));float ib1=1/(1-std::pow(.9f,(float)step)),ib2=1/(1-std::pow(.999f,(float)step)),lr=c->learning_rate,wd=c->weight_decay;
      if(direct)direct_back<<<blocks(B),THREADS>>>(x,net.E.t,net.E.m,net.E.v,net.b.t,net.b.m,net.b.v,w.id,w.y,B,I,lr,wd,ib1,ib2);
      else{gemv(bh,net.A.t,x,w.z,R,I);addact<<<blocks(R),THREADS>>>(w.z,net.a.t,R,c->activation);CU(cudaMemset(w.dz,0,R*4));factor_back<<<blocks(B),THREADS>>>(w.z,net.E.t,net.E.m,net.E.v,net.b.t,net.b.m,net.b.v,w.id,w.y,w.dz,B,R,lr,wd,ib1,ib2);actback<<<blocks(R),THREADS>>>(w.dz,w.z,R,c->activation);adam_outer<<<blocks((size_t)R*I),THREADS>>>(net.A.t,net.A.m,net.A.v,w.dz,x,R,I,lr,wd,ib1,ib2);adam_vec<<<blocks(R),THREADS>>>(net.a.t,net.a.m,net.a.v,w.dz,R,lr,0,ib1,ib2);}
      if(step%c->eval_every==0){v=eval(bh,net,w,dX,ev,c->activation);if(v+1e-7<bestmae){bestmae=v;beststep=step;best=flatten(net);bad=0;}else if(step>=c->min_steps)++bad;if(step>=c->min_steps&&bad>=c->patience)break;}
    }
    save_checkpoint(path,ui,isum,runsum,direct?0:1,R,c->activation,O,I,best,bestmae,beststep);destroy(net);++trained;++processed;unit_mae[ui]=bestmae;unit_step[ui]=beststep;if(processed%10==0||processed==selected_count){double elapsed=now()-train_t0;std::fprintf(stderr,"[methscope] upscale-train: units %u/%u (trained=%u resumed=%u) last_val_mae=%.6f%s elapsed=%.1fs ETA=%.1fs\n",processed,selected_count,trained,resumed,bestmae,pna?" PNA":"",elapsed,elapsed/processed*(selected_count-processed));}
  }
  all_bias.clear();all_bias.shrink_to_fit();
  if(c->pilot_units_path){char pp[4096];if(std::snprintf(pp,sizeof(pp),"%s/pilot.tsv",c->work_dir)>=(int)sizeof(pp))die("pilot path too long");FILE*f=fopen(pp,"w");if(!f)diep("cannot create pilot metrics",pp);std::fprintf(f,"unit\tclass\tcpgs\tmemberships\tmode\tbottleneck_dim\tbest_step\tvalidation_n\tvalidation_mae\n");for(uint32_t ui=0;ui<ih->n_units;++ui)if(selected[ui]){const MsuiUnit&u=units[ui];bool direct=c->mixed_direct&&!(u.flags&1);std::fprintf(f,"%u\t%s\t%u\t%u\t%s\t%u\t%u\t%llu\t%.9g\n",ui,(u.flags&2)?"PNA":(u.flags&1)?"pure":"mixed",u.cpg_count,u.membership_count,direct?"direct":"factor",direct?0:unit_rank[ui],unit_step[ui],(unsigned long long)unit_n[ui],unit_mae[ui]);}if(fclose(f))diep("cannot close pilot metrics",pp);std::fprintf(stderr,"[methscope] upscale-train: wrote pilot metrics %s\n",pp);cudaFree(w.z);cudaFree(w.dz);cudaFree(w.y);cudaFree(w.id);cudaFree(w.met);cudaFree(dX);BL(cublasDestroy(bh));if(th)unmap(trunk);unmap(idx);unmap(data);return 2;}
  uint64_t metadata_bytes=sizeof(ms_updec2_header_t)+(uint64_t)F*8+(uint64_t)ih->n_units*sizeof(ms_updec2_unit_t)+ih->n_cpg*4+(uint64_t)ih->n_real_memberships*sizeof(ms_updec2_membership_t);
  std::vector<ms_updec2_unit_t>od(ih->n_units);uint64_t po=metadata_bytes+trunk_floats*4;
  for(uint32_t ui=0;ui<ih->n_units;++ui){const MsuiUnit&u=units[ui];bool direct=c->mixed_direct&&!(u.flags&1);uint32_t R=direct?0:unit_rank[ui];uint64_t nf=direct?(uint64_t)u.cpg_count*I+u.cpg_count:(uint64_t)R*I+R+(uint64_t)u.cpg_count*R+u.cpg_count;od[ui]={u.output_offset,po,nf*4,u.cpg_count,u.membership_count,(uint16_t)(direct?0:1),(uint16_t)R,u.flags};po+=nf*4;}
  ms_updec2_header_t oh={};memcpy(oh.magic,MS_UPDEC2_MAGIC,8);oh.version=3;oh.flags=MS_UPDEC2_FLAG_GENOMIC|(c->feature_mode==MS_UPFEATURE_COUNT?MS_UPDEC2_FLAG_COUNT:0)|(c->feature_mode==MS_UPFEATURE_BETA?MS_UPDEC2_FLAG_BETA_ONLY:0)|(th?MS_UPDEC2_FLAG_TRUNK:0);oh.patterns=P;oh.input_dim=F;oh.n_units=ih->n_units;oh.n_memberships=ih->n_real_memberships;oh.target_unit_cpgs=ih->target_unit_cpgs;oh.activation=c->activation;oh.n_cpg=ih->n_cpg;oh.mean_offset=sizeof(oh);oh.scale_offset=oh.mean_offset+(uint64_t)F*4;oh.unit_offset=oh.scale_offset+(uint64_t)F*4;oh.cpg_offset=oh.unit_offset+(uint64_t)ih->n_units*sizeof(ms_updec2_unit_t);oh.membership_offset=oh.cpg_offset+ih->n_cpg*4;oh.param_offset=metadata_bytes;oh.file_bytes=po;oh.index_checksum=isum;oh.reserved0=H;
  FILE*out=fopen(c->model_path,"wb");if(!out)diep("cannot create UPDEC2",c->model_path);writeall(out,&oh,sizeof(oh),c->model_path);writeall(out,mean.data(),(size_t)F*4,c->model_path);writeall(out,scale.data(),(size_t)F*4,c->model_path);writeall(out,od.data(),od.size()*sizeof(od[0]),c->model_path);writeall(out,cpg,ih->n_cpg*4,c->model_path);writeall(out,members,(size_t)ih->n_real_memberships*sizeof(*members),c->model_path);
  uint64_t psum=UINT64_C(1469598103934665603);if(th){writeall(out,trunk_par,(size_t)trunk_floats*4,c->model_path);psum=fnv(psum,trunk_par,(size_t)trunk_floats*4);}for(uint32_t ui=0;ui<ih->n_units;++ui){char path[4096];ckpath(path,sizeof(path),c->work_dir,ui);FILE*f=fopen(path,"rb");if(!f)diep("missing unit checkpoint",path);CheckHeader ch;if(fread(&ch,1,sizeof(ch),f)!=sizeof(ch))diep("truncated unit checkpoint",path);std::vector<float>p(ch.param_floats);if(fread(p.data(),4,p.size(),f)!=p.size())diep("truncated unit checkpoint",path);fclose(f);writeall(out,p.data(),p.size()*4,c->model_path);psum=fnv(psum,p.data(),p.size()*4);}oh.parameter_checksum=psum;if(fseeko(out,0,SEEK_SET)||fwrite(&oh,1,sizeof(oh),out)!=sizeof(oh)||fclose(out))diep("cannot finalize UPDEC2",c->model_path);
  std::fprintf(stderr,"[methscope] upscale-train: wrote bare UPDEC2 %s (%llu bytes), trained=%u resumed=%u\n",c->model_path,(unsigned long long)oh.file_bytes,trained,resumed);
  cudaFree(w.z);cudaFree(w.dz);cudaFree(w.y);cudaFree(w.id);cudaFree(w.met);cudaFree(dX);BL(cublasDestroy(bh));if(th)unmap(trunk);unmap(idx);unmap(data);return 0;
}
