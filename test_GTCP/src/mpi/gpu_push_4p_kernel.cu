#define GPU_KERNEL


#ifdef MULTIPLE_FILE

#include <bench_gtc.h>
#include <cutil.h>
//#include <papi.h>

extern __device__ __constant__ real temp[MAX_MPSI] __align__ (16);extern __device__ __constant__ real dtemp[MAX_MPSI] __align__ (16);
extern __device__ __constant__ real rtemi[MAX_MPSI] __align__ (16);
//extern __device__ __constant__ real pfluxpsi[MFLUX] __align__ (16);
extern __device__ gtc_global_params_t params __align__ (16);
extern __device__ __constant__ real qtinv[MAX_MPSI] __align__ (16);
extern __device__ __constant__ real delt[MAX_MPSI] __align__ (16);
extern __device__ __constant__ int igrid[MAX_MPSI] __align__ (16);
extern __device__ __constant__ int mtheta[MAX_MPSI] __align__ (16);
extern __device__ gtc_radial_decomp_t radial_decomp __align__ (16);
//extern __device__ __constant__ int igrid_in __align__ (16);
//extern __device__ __constant__ int ipsi_in __align__ (16);
//extern __device__ __constant__ int ipsi_out __align__ (16);
#if SINGLE_PRECISION
extern texture<float, 1, cudaReadModeElementType> evectorTexRef;
#else
extern texture<int2, 1, cudaReadModeElementType> evectorTexRef;
#endif

#endif

#if OPTIMIZE_ACCESS
#if USE_TEXTURE
static __inline__ __device__ real fetch_evector(int i)
{
    int2 e = tex1Dfetch(evectorTexRef,i);
    return __hiloint2double(e.y,e.x);
}
#define EVECTOR(i) (fetch_evector(i))
#else
#define EVECTOR(i) (evector[i])
#endif //USE_TEXTURE
#else
#define EVECTOR(i) (evector[i])
#endif // OPTIMIZE_ACCESS

__global__ static void
__launch_bounds__(THREAD_BLOCK/*maxThreadsPerBlock*/, 1/*minBlocksPerMultiprocessor*/)
  gpu_push_gyro_interpolation(gtc_particle_data_t *zion, gtc_aux_particle_point_t *point, gtc_field_data_t *grid, gtc_diagnosis_data_t *diagnosis, int irk, int istep, int idiag)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nblocks = gridDim.x;
  const int nthreads = blockDim.x;
  const int gid = tid + bid*nthreads;
  const int np = nblocks * nthreads;
  const int mflux = MFLUX;
  
  extern __shared__ real shared_buffer_gyro[];
  real *vdrtmp = shared_buffer_gyro;
  // data for diagnosis
  real *scalar_data_s = &vdrtmp[5];
  //    real *eflux_s = &scalar_data_s[6*nthreads];
  //    real *rmarker_s = &eflux_s[mflux*nthreads];
  real *flux_s = &scalar_data_s[7*nthreads];
  
  if (idiag==0){
    for (int i=tid; i<7*nthreads; i+=nthreads){
      scalar_data_s[i] = 0.0;
    }
    for (int i=tid; i<4*mflux*nthreads; i+=nthreads){
      //           eflux_s[i] = 0.0;
      //           rmarker_s[i] = 0.0;
      flux_s[i] = 0.0;
    } 
    __syncthreads();
  }
  
  int mi = params.mi; int mimax=params.mimax;
  int mpsi = params.mpsi;
  
  const real a = params.a;
  const real a0 = params.a0;
  const real a1 = params.a1;
  const real delr = params.delr;
  const real pi2 = 2.0*params.pi;
  const real pi2_inv = params.pi2_inv;
  const int nbound = params.nbound;
  const real gyroradius = params.gyroradius;
  const real qion =  params.qion;
  const real aion =  params.aion;
  const real tstep =params.tstep;
  const real nonlinear = params.nonlinear;
  const real paranl = params.paranl;
  
  const real* __restrict__ point_vect = point->point_vect;
  const int* __restrict__ point_index = point->point_index;
  
  real* __restrict__ scalar_data = diagnosis->scalar_data;
  real* __restrict__ flux_data = diagnosis->flux_data;
  /*
    real* __restrict__ eflux = diagnosis->eflux;
    real* __restrict__ rmarker = diagnosis->rmarker;
    real* __restrict__ dmark = diagnosis->dmark;
    real* __restrict__ dden = diagnosis->dden;
  */
  
#if !ASSUME_MZETA_EQUALS1
  const int mzeta = params.mzeta;
#endif
  
#if !USE_TEXTURE
  const real * __restrict__ evector = grid->evector;
#endif
  const real * __restrict__ pfluxpsi = grid->pfluxpsi;
  
  real * __restrict__ zion1 = zion->z0;
  real * __restrict__ zion2 = zion->z1;
  real * __restrict__ zion3 = zion->z2;
  real * __restrict__ zion4 = zion->z3;
  real * __restrict__ zion5 = zion->z4;
  real * __restrict__ zion6 = zion->z5;
  
  real * __restrict__ zion01 = zion->z00;
  real * __restrict__ zion02 = zion->z01;
  real * __restrict__ zion03 = zion->z02;
  real * __restrict__ zion04 = zion->z03;
  real * __restrict__ zion05 = zion->z04;
  real * __restrict__ zion06 = zion->z05;
  
  real zion1m,  zion2m,  zion3m,  zion4m,  zion5m, zion6m;
  real wpi1, wpi2, wpi3;
  
  real dtime;
  real sbound=1.0;
  if (nbound==0) sbound = 0.0;
  real psimax=0.5*a1*a1;
  real psimin=0.5*a0*a0;
  real cmratio=qion/aion;
  real cinv=1.0/qion;
  real vthi=gyroradius*fabs(qion)/aion;
  real d_inv=real(mflux)/(a1-a0);
  
  real zion01m, zion02m, zion03m, zion04m, zion05m;
  
  for (int m=gid; m<mi; m+=np){
    
    if(irk==1) {
      dtime=0.5*tstep;
      if(tid<mflux) 
	vdrtmp[tid] = 0.0;
      //if(istep ==1) {
#if PTX_STREAM_INTRINSICS
	TunedLoad<real,CS>::Ld(zion1m,zion1+m,0);
	TunedLoad<real,CS>::Ld(zion2m,zion2+m,0);
	TunedLoad<real,CS>::Ld(zion3m,zion3+m,0);
	TunedLoad<real,CS>::Ld(zion4m,zion4+m,0);
	TunedLoad<real,CS>::Ld(zion5m,zion5+m,0);
	
	TunedStore<real,CS>::St(zion1m,zion01+m,0);
	TunedStore<real,CS>::St(zion2m,zion02+m,0);
	TunedStore<real,CS>::St(zion3m,zion03+m,0);
	TunedStore<real,CS>::St(zion4m,zion04+m,0);
	TunedStore<real,CS>::St(zion5m,zion05+m,0);
#else
	zion01[m] = zion1[m];
	zion02[m] = zion2[m];
	zion03[m] = zion3[m];
	zion04[m] = zion4[m];
	zion05[m] = zion5[m];
#endif
	//}
    } else {
      dtime=tstep;
      if(nonlinear<0.5) {
	printf("Error! decoupling modes for "
                    "nonlinear = 0.0 not implemented\n");
	if(tid<mflux)
	  vdrtmp[tid] = 0.0;
      }  else {
	if(tid<mflux)
	  vdrtmp[tid] = pfluxpsi[tid]; 
      }
    }
    __syncthreads();
    
#if PTX_STREAM_INTRINSICS
    //if((istep != 1)||(irk!=1)) {
    if (irk!=1){
      TunedLoad<real,CS>::Ld(zion1m,zion1+m,0);
      TunedLoad<real,CS>::Ld(zion2m,zion2+m,0);
      TunedLoad<real,CS>::Ld(zion3m,zion3+m,0);
      TunedLoad<real,CS>::Ld(zion4m,zion4+m,0);
      TunedLoad<real,CS>::Ld(zion5m,zion5+m,0);
    }
    TunedLoad<real,CS>::Ld(zion6m,zion6+m,0);
#else
    zion1m = zion1[m];
    zion2m = zion2[m];
    zion3m = zion3[m];
    zion4m = zion4[m];
    zion5m = zion5[m];
    zion6m = zion6[m];
#endif
    
    wpi1 = wpi2 = wpi3 = 0.0;
    int index;
    for (int larmor=0; larmor<4; larmor++){
      index = point_index[larmor*mi+m];
#ifdef _DEBUG_GPU
      if (index>=4*EXTRA_BUFFER*mimax){
	printf("index>EXTRA_BUFFER in push\n");
	printf("index=%d mimax=%d 4*EXTRA_BUFFER*mimax=%d\n", index, mimax, 4*EXTRA_BUFFER*mimax);
	CudaAssert(index<(4*EXTRA_BUFFER*mimax));
      }
#endif
      //  if (index!=-1){
	wpi1 += point_vect[4*index];
	wpi2 += point_vect[4*index+1];
	wpi3 += point_vect[4*index+2];
	// }
    }
    wpi1 = 0.25*wpi1; wpi2 = 0.25*wpi2; wpi3 = 0.25*wpi3;
    
    if(irk ==1){
      zion01m = zion1m;
      zion02m = zion2m;
      zion03m = zion3m;
      zion04m = zion4m;
      zion05m = zion5m;
    } else {
#if PTX_STREAM_INTRINSICS
      TunedLoad<real,CS>::Ld(zion01m,zion01+m,0);
      TunedLoad<real,CS>::Ld(zion02m,zion02+m,0);
      TunedLoad<real,CS>::Ld(zion03m,zion03+m,0);
      TunedLoad<real,CS>::Ld(zion04m,zion04+m,0);
      TunedLoad<real,CS>::Ld(zion05m,zion05+m,0);
#else
      zion01m = zion01[m];
      zion02m = zion02[m];
      zion03m = zion03[m];
      zion04m = zion04[m];
      zion05m = zion05[m];
#endif
    }
    
    // primary ion marker temperature and parallel flow velocity
    real ainv=1.0/a;
    
    /* update GC position */
    //#if !ONTHEFLY_PUSHAUX
#if SQRT_PRECOMPUTED
    real r = zion1m;
#else
    real r=sqrt(2.0*zion1m);
#endif
    //#endif
    real rinv=1.0/r;
    
    const real q0 = params.q0;
    const real q1 = params.q1;
    const real q2 = params.q2;
    const real rw = params.rw;
    const real rc = params.rc;
    
    int ii=d_abs_min_int(mpsi-1,int((r-a0)*delr));
    int ip=d_abs_min_int(mflux-1,1+int((r-a0)*d_inv));
    
    real wp0=real(ii+1)-(r-a0)*delr;
    real wp1=1.0-wp0;
    real tem=wp0*temp[ii]+wp1*temp[ii+1];
    real q=q0+q1*r*ainv+q2*r*r*ainv*ainv;
    real qinv=1.0/q;
    real cost=cos(zion2m);
    real sint=sin(zion2m);
    real b=1.0/(1.0+r*cost);
    real g=1.0;
    real gp=0.0;
    real ri=0.0;
    real rip=0.0;
    real dbdp=-1.0*b*b*cost*rinv;
    real dbdt=b*b*r*sint;
    real dedb=cinv*(zion4m*zion4m*qion*b*cmratio+zion6m*zion6m);
    real deni=1.0/(g*q + ri + zion4m*(g*rip-ri*gp));
    real upara=zion4m*b*cmratio;
    real energy=0.5*aion*upara*upara+zion6m*zion6m*b;
    real rfac=rw*(r-rc);
#if PROFILE_SHAPE==0
    rfac=rfac*rfac;
    rfac=rfac*rfac*rfac;
    rfac=exp(-1*rfac);
#elif PROFILE_SHAPE==1
    rfac=tanh(rfac)*tanh(rfac);
    rfac=1.0-rfac;
#endif
    
    real kappa=1.0-sbound+sbound*rfac;
    const real kappati = params.kappati;
    const real kappan = params.kappan;
    kappa=((energy*tem-1.5)*kappati+kappan)*kappa*rinv;
    // perturbed quantities
    real dptdp=wpi1;
    real dptdt=wpi2;
    real dptdz=wpi3-wpi2*qinv;
    real epara=-1.0*wpi3*b*q*deni;
    // subtract net particle flow
    dptdt=dptdt+vdrtmp[ip];
    
    // ExB drift in radial direction for w-dot and flux diagnostics
    real vdr=q*(ri*dptdz-g*dptdt)*deni;
    real wdrive=vdr*kappa;
    real wpara=epara*(upara-dtemp[ii])*qion*tem;
    // Common subexpression elimination
    real wdrift=q*(g*dbdt*dptdp-g*dbdp*dptdt+ri*dbdp*dptdz)*deni*dedb*qion*tem;
    
    real wdot=(zion06[m]-paranl*zion5m)*(wdrive+wpara+wdrift);
    
    // self-consistent and external electric field for marker orbits
    const real flow0 = params.flow0;
    const real flow1 = params.flow1;
    const real flow2 = params.flow2;
    dptdp=dptdp*nonlinear+gyroradius*(flow0+flow1*r*ainv+flow2*r*r*ainv*ainv);
    dptdt=dptdt*nonlinear;
    dptdz=dptdz*nonlinear;
    
    // particle velocity
    real pdot = q*(-g*dedb*dbdt - g*dptdt + ri*dptdz)*deni;
    real tdot = (upara*b*(1.0-q*gp*zion4m) + q*g*(dedb*dbdp + dptdp))*deni;
    real zdot = (upara*b*q*(1.0+rip*zion4m) - q*ri*(dedb*dbdp + dptdp))*deni;
    real rdot = ((gp*zion4m-1.0)*(dedb*dbdt + paranl*dptdt)-paranl*q*(1.0+rip*zion4m)*dptdz)*deni;
    
    // update particle position
#if PTX_STREAM_INTRINSICS
#if SQRT_PRECOMPUTED
    zion1m = max(1.0e-8*psimax,0.5*zion01m*zion01m + dtime*pdot);
    zion1m = sqrt(2.0*zion1m);
#else
    zion1m = max(1.0e-8*psimax,zion01m+dtime*pdot);
#endif
    TunedStore<real,CS>::St(zion1m,zion1+m,0);
    zion2m = zion02m+dtime*tdot;
    zion3m = zion03m+dtime*zdot;
    zion4m = zion04m + dtime*rdot;
    TunedStore<real,CS>::St(zion4m,zion4+m,0);
    zion5m = zion05m + dtime*wdot;
    TunedStore<real,CS>::St(zion5m,zion5+m,0);
    real z1t = zion2m *pi2_inv+10;
    zion2m = pi2*(z1t-((int)z1t));
    TunedStore<real,CS>::St(zion2m,zion2+m,0);
    z1t = zion3m*pi2_inv+10;
    zion3m = pi2*(z1t - ((int)z1t));
    TunedStore<real,CS>::St(zion3m,zion3+m,0);
    
    if(irk==2) {
#if SQRT_PRECOMPUTED
      if((zion1m > a1)||(zion1m < a0)) {
#else
	if((zion1m > psimax)||(zion1m < psimin)) {
#endif
	TunedStore<real,CS>::St(zion01m,zion1+m,0);
	TunedStore<real,CS>::St(pi2-zion02m,zion2+m,0);
	TunedStore<real,CS>::St(zion03m,zion3+m,0);
	TunedStore<real,CS>::St(zion04m,zion4+m,0);
	TunedStore<real,CS>::St(zion05m,zion5+m,0);
        
	TunedStore<real,CS>::St(pi2-zion02m,zion02+m,0);
      } /*else {
	TunedStore<real,CS>::St(zion1m,zion01+m,0);
	TunedStore<real,CS>::St(zion2m,zion02+m,0);
	TunedStore<real,CS>::St(zion3m,zion03+m,0);
	TunedStore<real,CS>::St(zion4m,zion04+m,0);
	TunedStore<real,CS>::St(zion5m,zion05+m,0);
	}*/
    }
    
#else
#if SQRT_PRECOMPUTE
    zion1m = max(1.0e-8*psimax,0.5*zion01m*zion01m + dtime*pdot);
    zion1m = sqrt(2.0*zion1m);
#else
    zion1m = max(1.0e-8*psimax,zion01m+dtime*pdot);
#endif
    zion2m = zion02m+dtime*tdot;
    zion3m = zion03m+dtime*zdot;
    zion4[m] = zion04m+dtime*rdot;
    zion5[m] = zion05m+dtime*wdot;
    // theta and zeta normalize to [0,2*pi), modulo is slower than hand coded
    // procedure on Seaborg. However, modulo works better and is preferable.
    real z1t = zion2m *pi2_inv+10;
    zion2[m]=pi2*(z1t-((int)z1t));
    z1t = zion3m*pi2_inv+10;
    zion3[m]=pi2*(z1t - ((int)z1t));
    
    if(irk==2) {
#if SQRT_PRECOMPUTED
      if(zion1[m] > a1) {
#else
      if(zion1[m] > psimax) {
#endif
	zion1[m]=zion01m;
	zion2[m]=pi2-zion02m;
	zion3[m]=zion03m;
	zion4[m]=zion04m;
	zion5[m]=zion05m;
#if SQRT_PRECOMPUTED
        else if(zion1[m] < a0) {
#else
      } else if (zion1[m] < psimin) {
#endif
	zion1[m]=zion01m;
	zion2[m]=pi2-zion02m;
	zion3[m]=zion03m;
	zion4[m]=zion04m;
	zion5[m]=zion05m;
      }
	/*
      zion01[m] = zion1[m];
      zion02[m] = zion2[m];
      zion03[m] = zion3[m];
      zion04[m] = zion4[m];
      zion05[m] = zion5[m];
	*/
    }
    
#endif
    
    if (idiag==0){
      ip = d_abs_min_int(mflux-1, (int)((r-a0)*d_inv));
      //	ii = d_abs_min_int(mpsi, (int)((r-a0)*delr+0.5));
      
      real vdrenergy = vdr*rinv*(energy-1.5*aion*vthi*vthi*rtemi[ii])*zion05m;
      
      //        rmarker_s[ip*nthreads+tid] += zion06[m];
      //        eflux_s[ip*nthreads+tid] += vdrenergy;
      flux_s[ip*nthreads+tid] += vdrenergy; // eflux
      flux_s[mflux*nthreads+ip*nthreads+tid] += zion06[m]; //rmarker
      flux_s[2*mflux*nthreads + ip*nthreads+tid] += vdr*rinv*r; // dmark
      flux_s[3*mflux*nthreads + ip*nthreads+tid] += 1.0; //dden
      
      scalar_data_s[0*nthreads+tid] += vdrenergy; // efluxi
      scalar_data_s[1*nthreads+tid] += vdr*rinv*zion05m; // pfluxi
      scalar_data_s[2*nthreads+tid] += b*zion04m*zion05m; // dflowi
      scalar_data_s[3*nthreads+tid] += zion05m*zion05m; // entropyi
      scalar_data_s[4*nthreads+tid] +=  energy*zion05m; // particles_energy[0]
      scalar_data_s[5*nthreads+tid] +=  energy; // particles_energy[1]
      scalar_data_s[6*nthreads+tid] += zion05m;
    }

  } // end m=gid
  __syncthreads();
  
  if (idiag==0){
    
    int nTotalThreads = nthreads;
    while (nTotalThreads>1){
      int half = (nTotalThreads >> 1);
      if (tid < half){
	
	for (int i=0; i<7; i++){ 
	  scalar_data_s[i*nthreads+tid] += scalar_data_s[i*nthreads+tid+half];
	}
	
	for (int i=0; i<mflux; i++)
	  {
	    //eflux_s[i*nthreads+tid] += eflux_s[i*nthreads+tid+half];
	    //rmarker_s[i*nthreads+tid] += rmarker_s[i*nthreads+tid+half];	       
	    flux_s[i*nthreads+tid] += flux_s[i*nthreads+tid+half];
	    flux_s[mflux*nthreads+i*nthreads+tid] += flux_s[mflux*nthreads+i*nthreads+tid+half];
	    flux_s[2*mflux*nthreads+i*nthreads+tid] += flux_s[2*mflux*nthreads+i*nthreads+tid+half];
	    flux_s[3*mflux*nthreads+i*nthreads+tid] += flux_s[3*mflux*nthreads+i*nthreads+tid+half];
	  }
      }
      __syncthreads();
      nTotalThreads = (nTotalThreads >> 1);
    }
    
    if (tid==0){
      atomicDPupdate(scalar_data, scalar_data_s[0]);
      atomicDPupdate(scalar_data+2, scalar_data_s[nthreads]);
      atomicDPupdate(scalar_data+6, scalar_data_s[2*nthreads]);
      atomicDPupdate(scalar_data+8, scalar_data_s[3*nthreads]);
      atomicDPupdate(scalar_data+12, scalar_data_s[4*nthreads]);
      atomicDPupdate(scalar_data+13, scalar_data_s[5*nthreads]);
      atomicDPupdate(scalar_data+15, scalar_data_s[6*nthreads]);
    }
    /*
      if (tid<5)
      //atomicDPupdate(eflux+tid,eflux_s[tid*nthreads]);
      atomicDPupdate(eflux+tid, flux_s[tid*nthreads]);
      
      if (tid>=5&&tid<10)
      //atomicDPupdate(rmarker+tid-5,rmarker_s[(tid-5)*nthreads]);
      atomicDPupdate(rmarker+tid-5, flux_s[mflux*nthreads+(tid-5)*nthreads]);
      
      if (tid>=10&&tid<15)
      atomicDPupdate(dmark+tid-10, flux_s[2*mflux*nthreads+(tid-10)*nthreads]);
      
      if (tid>=15&&tid<20)
         atomicDPupdate(dden+tid-15, flux_s[3*mflux*nthreads+(tid-15)*nthreads]);
    */
    if (tid<5)
      //atomicDPupdate(eflux+tid,eflux_s[tid*nthreads]);
      atomicDPupdate(flux_data+tid, flux_s[tid*nthreads]);
    
    if (tid>=5&&tid<10)
      //atomicDPupdate(rmarker+tid-5,rmarker_s[(tid-5)*nthreads]);
      atomicDPupdate(flux_data+tid, flux_s[mflux*nthreads+(tid-5)*nthreads]);
    
    if (tid>=10&&tid<15)
      atomicDPupdate(flux_data+tid, flux_s[2*mflux*nthreads+(tid-10)*nthreads]);
    
    if (tid>=15&&tid<20)
      atomicDPupdate(flux_data+tid, flux_s[3*mflux*nthreads+(tid-15)*nthreads]);
  }
  
}

    __global__ static void gpu_push_point_interpolation(gtc_field_data_t* grid, gtc_aux_particle_point_t* point, int nloc_over_cluster, int mcellperthread)
{
  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int nthreadsx = blockDim.x;
  const int nthreadsy = blockDim.y;
  const int gidx=tidx+blockIdx.x*nthreadsx;
  
  my_int* __restrict__ point_index_count;
  real* __restrict__ point_vect;
  
  int mpsi = params.mpsi;
  
  const real pi2_inv = params.pi2_inv;
  const int mzeta = params.mzeta;
  
  real delz, zetamin, zetatmp;
  int mpsi_max;
  real wzt, rdum, tflr, tdumtmp, tdum;
  real wtion0tmp, wtion1tmp;
  int jtion0tmp, jtion1tmp, j00, j01;
  real wz1, wz0;
  int ij1,ij2,ij3,ij4;
  int im, ii,idx1,idx2,idx3,idx4;
 
  real wp0, wp1, wt00, wt01, wt10, wt11;
  int kk;
  real e1, e2, e3;

  int igrid_in;

  zetamin = params.zetamin;
  delz = params.delz;
  igrid_in = radial_decomp.igrid_in;
  
  point_index_count = point->point_index_count;
  point_vect = point->point_vect;
  
#if !USE_TEXTURE
  const real * __restrict__ evector = grid->evector;
#endif
  
  mpsi_max = mpsi - 1;
  
  int maxcount = 0;
  if (gidx<nloc_over_cluster)
    maxcount = (int)point_index_count[gidx];
  
  extern __shared__ int shared_buffer_sc[];
  int *offset0 = shared_buffer_sc;
  int *offset1 = &shared_buffer_sc[nthreadsx];
  real * evector_s0 = (real *)&offset1[nthreadsx];
  real * evector_s1 = &evector_s0[3*(mzeta+1)*(nthreadsx*mcellperthread+1)];
  
  // find the starting index for the array in shared memory
  if (tidy==0){
    offset0[tidx] = 1000000000;
    offset1[tidx] = 1000000000;
    
    if (gidx<nloc_over_cluster){
      if (maxcount>0){
	rdum = point_vect[4*gidx];
	tflr = point_vect[4*gidx+1];
	zetatmp = point_vect[4*gidx+2];
	
	ii = d_abs_min_int(mpsi_max, (int) rdum);
	im = ii;
	tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
	tdum = (tdumtmp - (int) tdumtmp) * delt[im];
	j00 = d_abs_min_int(mtheta[im]-1, (int) tdum);
	jtion0tmp = igrid[im] + j00;
	
#ifdef _DEBUG_GPU
	if ((jtion0tmp-igrid_in)/mcellperthread!=gidx){
	  printf("jtion0tmp=%d mcellperthread=%d gidx=%d\n", jtion0tmp, mcellperthread, gidx);
	  CudaAssert(jtion0tmp/mcellperthread==gidx);
	}
#endif
	im = ii + 1;
	tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
	tdum = (tdumtmp - (int) tdumtmp) * delt[im];
	j01 = d_abs_min_int(mtheta[im]-1, (int) tdum);
	jtion1tmp = igrid[im] + j01;
	
	offset0[tidx] = 3*(mzeta+1)*gidx*mcellperthread;
	
	if (gidx==(igrid[ii+1]-2-igrid_in)/mcellperthread||gidx==(igrid[ii+1]-3-igrid_in)/mcellperthread){
	  offset1[tidx] = 1000000000;
	}
	else{
	  offset1[tidx] = 3*((mzeta+1)*(jtion1tmp-igrid_in) - 16);
	  offset1[tidx] -= 3*(mzeta+1)*(mcellperthread-1);
	}
      }
    } 
  }
  __syncthreads();
  
  int nTotalThreads = nthreadsx;
  while (nTotalThreads>1){
    int half = (nTotalThreads >> 1);
    if (tidy==0){
      if (tidx < half){
	int temp0 = offset0[tidx+half];
	if (temp0<offset0[tidx]) offset0[tidx]=temp0;
	
	int temp1 = offset1[tidx+half];
	if (temp1<offset1[tidx]) offset1[tidx]=temp1;
      }
    }
    __syncthreads();
    nTotalThreads = (nTotalThreads >> 1);
  }
  
  if (tidy==0){
    offset0[tidx] = offset0[0];
    offset1[tidx] = offset1[0];
  }
  __syncthreads();
  
  // copy E field from global or texture to shared memory
  for (int ij=tidx; ij<nthreadsx*mcellperthread+1; ij+=nthreadsx){
    if (offset1[tidx]!=1000000000){
      int ij_off0 = 3*(mzeta+1)*ij+offset0[tidx];
      for (int ii=tidy; ii<3*(mzeta+1); ii+= nthreadsy){
	evector_s0[ij*3*(mzeta+1)+ii] = EVECTOR(ij_off0+ii);
      }
    }
    else{
      for (int ii=tidy; ii<3*(mzeta+1); ii+= nthreadsy){
	evector_s0[ij*3*(mzeta+1)+ii] = 0.0;
      }
    }
  }
  for (int ij=tidx; ij<1.5*nthreadsx*mcellperthread; ij+=nthreadsx){
    if (offset1[tidx]!=1000000000){
      int ij_off1 = 3*(mzeta+1)*ij+offset1[tidx];
      for (int ii=tidy; ii<3*(mzeta+1); ii+= nthreadsy){
	evector_s1[ij*3*(mzeta+1)+ii] = EVECTOR(ij_off1+ii);
      }
    }
    else{
      for (int ii=tidy; ii<3*(mzeta+1); ii+= nthreadsy){
	evector_s1[ij*3*(mzeta+1)+ii] = 0.0;
      }
    }
  }
  __syncthreads();
  
  // field interpolation from shared memory
  for (int m=tidy*nloc_over_cluster+gidx, iter=0; iter<maxcount; iter+=blockDim.y, m+=nloc_over_cluster*blockDim.y){
    if (iter+tidy<maxcount){
      e1 = 0.0; 
      e2 = 0.0;
      e3 = 0.0;
      
      rdum = point_vect[4*m];
      tflr = point_vect[4*m+1];
      zetatmp = point_vect[4*m+2];
      
      wzt      = (zetatmp-zetamin)*delz;
      kk       = d_abs_min_int(mzeta-1, (int) wzt);
      wz1      = wzt - (real) kk;
      wz0      = 1.0 - wz1;
      
      ii      = d_abs_min_int(mpsi_max, (int) rdum);
      wp1     = rdum - (real) ii;
      wp0     = 1.0  - wp1;
      
      im = ii;
      tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
      tdum = (tdumtmp - (int) tdumtmp) * delt[im];
      j00 = d_abs_min_int(mtheta[im]-1, (int) tdum);
      jtion0tmp = igrid[im] + j00;
      wtion0tmp = tdum - (real) j00;
      
      im = ii + 1;
      tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
      tdum = (tdumtmp - (int) tdumtmp) * delt[im];
      j01 = d_abs_min_int(mtheta[im]-1, (int) tdum);
      jtion1tmp = igrid[im] + j01;
      wtion1tmp = tdum - (real) j01;
      
#ifdef _DEBUG_GPU
      if ((jtion0tmp-igrid_in)/mcellperthread!=gidx){
	printf("jtion0tmp=%d mcellperthread=%d gidx=%d\n", jtion0tmp, mcellperthread, gidx);
	CudaAssert(jtion0tmp/mcellperthread==gidx);
      }
#endif
      wt10 = wtion0tmp;
      wt00 = 1.0 - wt10;
      
      wt11 = wtion1tmp;
      wt01 = 1.0 - wt11;
      
      ij1 = jtion0tmp - igrid_in;
      ij3 = jtion1tmp - igrid_in;
      ij2 = ij1 + 1;
      ij4 = ij3 + 1;
      
#if ASSUME_MZETA_EQUALS1
      idx1 = 6*ij1;
      //            idx2 = 6*ij2;
      idx3 = 6*ij3;
      //            idx4 = 6*ij4;
      
      idx1 = idx1 - offset0[tidx];
      
#ifdef _DEBUG_GPU
      if (idx1<0||idx1>=3*(mzeta+1)*nthreadsx*mcellperthread)
	printf("jtion0tmp=%d gidx=%d idx1=%d offset0=%d\n", jtion0tmp, gidx, idx1, offset0[tidx	\
											   ]);
      CudaAssert(idx1>=0);
      CudaAssert(idx1<3*(mzeta+1)*nthreadsx*mcellperthread);
#endif
      
      e1 = e1+wp0*wt00*(wz0*evector_s0[idx1+0]+wz1*evector_s0[idx1+3]);
      e2 = e2+wp0*wt00*(wz0*evector_s0[idx1+1]+wz1*evector_s0[idx1+4]);
      e3 = e3+wp0*wt00*(wz0*evector_s0[idx1+2]+wz1*evector_s0[idx1+5]);
      
      e1 = e1+wp0*wt10*(wz0*evector_s0[idx1+6+0]+wz1*evector_s0[idx1+6+3]);
      e2 = e2+wp0*wt10*(wz0*evector_s0[idx1+6+1]+wz1*evector_s0[idx1+6+4]);
      e3 = e3+wp0*wt10*(wz0*evector_s0[idx1+6+2]+wz1*evector_s0[idx1+6+5]);
      
      idx3 = idx3 - offset1[tidx];
      
      if (idx3<0||idx3>=3*(mzeta+1)*(1.5*nthreadsx*mcellperthread-1)){
	idx3 = idx3 + offset1[tidx];
	e1 = e1+wp1*wt01*(wz0*EVECTOR(idx3+0)+wz1*EVECTOR(idx3+3));
	e2 = e2+wp1*wt01*(wz0*EVECTOR(idx3+1)+wz1*EVECTOR(idx3+4));
	e3 = e3+wp1*wt01*(wz0*EVECTOR(idx3+2)+wz1*EVECTOR(idx3+5));
	
	e1 = e1+wp1*wt11*(wz0*EVECTOR(idx3+6+0)+wz1*EVECTOR(idx3+6+3));
	e2 = e2+wp1*wt11*(wz0*EVECTOR(idx3+6+1)+wz1*EVECTOR(idx3+6+4));
	e3 = e3+wp1*wt11*(wz0*EVECTOR(idx3+6+2)+wz1*EVECTOR(idx3+6+5));
      }
      else {
	e1 = e1+wp1*wt01*(wz0*evector_s1[idx3+0]+wz1*evector_s1[idx3+3]);
	e2 = e2+wp1*wt01*(wz0*evector_s1[idx3+1]+wz1*evector_s1[idx3+4]);
	e3 = e3+wp1*wt01*(wz0*evector_s1[idx3+2]+wz1*evector_s1[idx3+5]);
	
	e1 = e1+wp1*wt11*(wz0*evector_s1[idx3+6+0]+wz1*evector_s1[idx3+6+3]);
	e2 = e2+wp1*wt11*(wz0*evector_s1[idx3+6+1]+wz1*evector_s1[idx3+6+4]);
	e3 = e3+wp1*wt11*(wz0*evector_s1[idx3+6+2]+wz1*evector_s1[idx3+6+5]);
      }
      
      /*
      // debug
      e1 =e1+wp0*wt00*(wz0*EVECTOR(idx1+0)+wz1*EVECTOR(idx1+3));
      e2 =e2+wp0*wt00*(wz0*EVECTOR(idx1+1)+wz1*EVECTOR(idx1+4));
      e3 =e3+wp0*wt00*(wz0*EVECTOR(idx1+2)+wz1*EVECTOR(idx1+5));
      
      e1 =e1+wp0*wt10*(wz0*EVECTOR(idx1+6+0)+wz1*EVECTOR(idx1+6+3));
      e2 =e2+wp0*wt10*(wz0*EVECTOR(idx1+6+1)+wz1*EVECTOR(idx1+6+4));
      e3 =e3+wp0*wt10*(wz0*EVECTOR(idx1+6+2)+wz1*EVECTOR(idx1+6+5));
      
      e1 =e1+wp1*wt01*(wz0*EVECTOR(idx3+0)+wz1*EVECTOR(idx3+3));
      e2 =e2+wp1*wt01*(wz0*EVECTOR(idx3+1)+wz1*EVECTOR(idx3+4));
      e3 =e3+wp1*wt01*(wz0*EVECTOR(idx3+2)+wz1*EVECTOR(idx3+5));
      
      e1 =e1+wp1*wt11*(wz0*EVECTOR(idx3+6+0)+wz1*EVECTOR(idx3+6+3));
      e2 =e2+wp1*wt11*(wz0*EVECTOR(idx3+6+1)+wz1*EVECTOR(idx3+6+4));
      e3 =e3+wp1*wt11*(wz0*EVECTOR(idx3+6+2)+wz1*EVECTOR(idx3+6+5));
      */
#else
      idx1 = 3*(mzeta+1)*ij1+3*kk;
      idx2 = 3*(mzeta+1)*ij2+3*kk;
      idx3 = 3*(mzeta+1)*ij3+3*kk;
      idx4 = 3*(mzeta+1)*ij4+3*kk;
	    
      idx1 = idx1 - offset0[tidx];
      idx2 = idx2 - offset0[tidx];
      idx3 = idx3 - offset1[tidx];
      idx4 = idx4 - offset1[tidx];
      
      e1 = e1+wp0*wt00*(wz0*evector_s0[idx1+0]+wz1*evector_s0[idx1+3]);
      e2 = e2+wp0*wt00*(wz0*evector_s0[idx1+1]+wz1*evector_s0[idx1+4]);
      e3 = e3+wp0*wt00*(wz0*evector_s0[idx1+2]+wz1*evector_s0[idx1+5]);
      e1 = e1+wp0*wt10*(wz0*evector_s0[idx2+0]+wz1*evector_s0[idx2+3]);
      e2 = e2+wp0*wt10*(wz0*evector_s0[idx2+1]+wz1*evector_s0[idx2+4]);
      e3 = e3+wp0*wt10*(wz0*evector_s0[idx2+2]+wz1*evector_s0[idx2+5]);
      
      if (idx3<0||idx3>=3*(mzeta+1)*(1.5*nthreadsx*mcellperthread-1)){
	e1 = e1+wp1*wt01*(wz0*evector_s1[idx3+0]+wz1*evector_s1[idx3+3]);
	e2 = e2+wp1*wt01*(wz0*evector_s1[idx3+1]+wz1*evector_s1[idx3+4]);
	e3 = e3+wp1*wt01*(wz0*evector_s1[idx3+2]+wz1*evector_s1[idx3+5]);
	e1 = e1+wp1*wt11*(wz0*evector_s1[idx4+0]+wz1*evector_s1[idx4+3]);
	e2 = e2+wp1*wt11*(wz0*evector_s1[idx4+1]+wz1*evector_s1[idx4+4]);
	e3 = e3+wp1*wt11*(wz0*evector_s1[idx4+2]+wz1*evector_s1[idx4+5]);
      }
      else{
	idx3 = idx3 + offset1[tidx];
	idx4 = idx4 + offset1[tidx];
	e1 = e1+wp1*wt01*(wz0*EVECTOR(idx3+0)+wz1*EVECTOR(idx3+3));
	e2 = e2+wp1*wt01*(wz0*EVECTOR(idx3+1)+wz1*EVECTOR(idx3+4));
	e3 = e3+wp1*wt01*(wz0*EVECTOR(idx3+2)+wz1*EVECTOR(idx3+5));
	e1 = e1+wp1*wt11*(wz0*EVECTOR(idx4+0)+wz1*EVECTOR(idx4+3));
	e2 = e2+wp1*wt11*(wz0*EVECTOR(idx4+1)+wz1*EVECTOR(idx4+4));
	e3 = e3+wp1*wt11*(wz0*EVECTOR(idx4+2)+wz1*EVECTOR(idx4+5));
      }
#endif
      
      point_vect[4*m] = e1;
      point_vect[4*m+1] = e2;
      point_vect[4*m+2] = e3;
      
    }
  }
}

extern "C"
void call_gpu_push_4p_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int idiag){
    
  gtc_global_params_t   *h_params   = &(gtc_input->global_params);
  gtc_field_data_t *h_grid = &(gtc_input->field_data);
  gtc_field_data_t *d_grid = &(gpu_kernel_input->d_grid);
  gtc_diagnosis_data_t *h_diagnosis = &(gtc_input->diagnosis_data);
  gtc_diagnosis_data_t *d_diagnosis = &(gpu_kernel_input->d_diagnosis);   
  gtc_radial_decomp_t *h_radial_decomp = &(gtc_input->radial_decomp);
  
  int mzeta =  h_params->mzeta;    //int mgrid =  h_params->mgrid; int mpsi = h_params->mpsi;
  int mi =  h_params->mi; int mype = gtc_input->parallel_decomp.mype;	
  int nloc_over = h_radial_decomp->nloc_over;

  //  int mgrid_cluster = (mgrid-h_grid->mtheta[mpsi]+CLUSTER_SIZE-1)/CLUSTER_SIZE;    
  int nloc_over_cluster = (nloc_over+CLUSTER_SIZE-1)/CLUSTER_SIZE;
  /************** copy E field to GPU *****************/
  gpu_timer_start(gpu_kernel_input);
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_grid->evector, h_grid->evector, 3*(mzeta+1)*nloc_over*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_grid->pfluxpsi, h_grid->pfluxpsi, MFLUX*sizeof(real), cudaMemcpyHostToDevice));
  /**************** copy and reset diagnosis data****************/
  if (idiag==0){
       // notice: host-device memory copy of 64KB or less is NOT synchronous
    CUDA_SAFE_CALL(cudaMemcpy((void*)(d_diagnosis->scalar_data), (h_diagnosis->scalar_data), 16*sizeof(real),cudaMemcpyHostToDevice));
    //        CUDA_SAFE_CALL(cudaMemset(d_diagnosis->eflux, 0, 4*MFLUX*sizeof(real)));
    CUDA_SAFE_CALL(cudaMemset(d_diagnosis->flux_data, 0, 4*MFLUX*sizeof(real)));
  }
  gpu_kernel_input->gpu_timing.memtransfer_push_time +=  gpu_timer_measure(gpu_kernel_input);

  /************** interpolate grid-based E to point-based E************/
  
  int mcellperthread=CLUSTER_SIZE;
  int nthreadsx = 32; int nthreadsy=8;
  dim3 nthread_3d(nthreadsx, nthreadsy);
  int nblocks = (nloc_over_cluster + nthreadsx - 1)/nthreadsx;
  int shared_buffer = 2*nthreadsx*sizeof(int) + int(3*(mzeta+1)*mcellperthread*1.5*nthreadsx)*sizeof(real)+3*(mzeta+1)*(mcellperthread*nthreadsx+1)*sizeof(real);
  gpu_push_point_interpolation<<<nblocks, nthread_3d, shared_buffer>>>(gpu_kernel_input->ptr_d_grid, gpu_kernel_input->ptr_d_aux_point, nloc_over_cluster, mcellperthread);
  gpu_kernel_input->gpu_timing.interpolation_push_point_time += gpu_timer_measure(gpu_kernel_input);	
 
  /*************** interpolate point-based E to gyroparticle ***************/     
  int mi_per_thread = gpu_kernel_input->charge_mi_per_thread;
  int nthreads = gpu_kernel_input->nthreads/2;
  int mp = gpu_kernel_input->deviceProp.multiProcessorCount;
  int m = (mi + nthreads*mp - 1)/ (nthreads*mp);
  m = (m + mi_per_thread - 1)/mi_per_thread;
  nblocks = mp*m;
  mi_per_thread = (mi + nblocks*nthreads - 1)/ mi_per_thread;
     
  shared_buffer = 5*sizeof(real) + (7*nthreads+4*MFLUX*nthreads)*sizeof(real);
  gpu_push_gyro_interpolation<<<nblocks, nthreads, shared_buffer>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_aux_point, gpu_kernel_input->ptr_d_grid, gpu_kernel_input->ptr_d_diagnosis, gpu_kernel_input->irk, gpu_kernel_input->istep, gpu_kernel_input->idiag);
  
  gpu_kernel_input->gpu_timing.interpolation_push_gyro_time += gpu_timer_measure(gpu_kernel_input);
  
  /********************* copy diagnosis data back	to host********************/
  if (idiag==0){
    // notice: host-device memory copy of 64KB or less is NOT synchronous
    CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->scalar_data), (d_diagnosis->scalar_data), 16*sizeof(real),cudaMemcpyDeviceToHost));
    /*
      CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->eflux), (d_diagnosis->eflux), MFLUX*sizeof(real),cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->rmarker), (d_diagnosis->rmarker), MFLUX*sizeof(real),cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->dmark), (d_diagnosis->dmark), MFLUX*sizeof(real),cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->dden), (d_diagnosis->dden), MFLUX*sizeof(real),cudaMemcpyDeviceToHost));
    */
    CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->flux_data), (d_diagnosis->flux_data), 4*MFLUX*sizeof(real),cudaMemcpyDeviceToHost));
        gpu_kernel_input->gpu_timing.memtransfer_push_time +=  gpu_timer_measure(gpu_kernel_input);
	
  }
  gpu_kernel_input->gpu_timing.device_push_time += gpu_timer_measure_end(gpu_kernel_input);
  
}
