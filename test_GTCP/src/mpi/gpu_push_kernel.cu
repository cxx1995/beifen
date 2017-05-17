#define GPU_KERNEL

#ifdef MULTIPLE_FILE

#include <bench_gtc.h>
#include <cutil.h>
extern __device__ __constant__ real temp[MAX_MPSI] __align__ (16);
extern __device__ __constant__ real dtemp[MAX_MPSI] __align__ (16);
extern __device__ __constant__ real rtemi[MAX_MPSI] __align__ (16);
//extern __device__ __constant__ real pfluxpsi[MFLUX] __align__ (16);
extern __device__ gtc_global_params_t params __align__ (16);
extern __device__ __constant__ real qtinv[MAX_MPSI] __align__ (16);
extern __device__ __constant__ real delt[MAX_MPSI] __align__ (16);
extern __device__ __constant__ int igrid[MAX_MPSI] __align__ (16);
extern __device__ __constant__ int mtheta[MAX_MPSI] __align__ (16);
extern __device__ gtc_radial_decomp_t radial_decomp __align__ (16);

#if SINGLE_PRECISION
extern texture<float, 1, cudaReadModeElementType> evectorTexRef;
#else
extern texture<int2, 1, cudaReadModeElementType> evectorTexRef;
#endif

#endif

#define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!", blockIdx.x, threadIdx.x); }

#if OPTIMIZE_ACCESS
#if  	USE_TEXTURE
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
#define THREAD_BLOCK_OVERLOAD_FACTOR 4

__global__ static void 
//__launch_bounds__(THREAD_BLOCK * THREAD_BLOCK_OVERLOAD_FACTOR /*maxThreadsPerBlock*/, 1/*minBlocksPerMultiprocessor*/)
__launch_bounds__(THREAD_BLOCK /*maxThreadsPerBlock*/, 1/*minBlocksPerMultiprocessor*/)
  gpu_pushi_kernel(gtc_particle_data_t *zion, gtc_aux_particle_data_t *aux_zion, gtc_field_data_t *grid, gtc_diagnosis_data_t *diagnosis, int irk, int istep, int idiag) 
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
  real *scalar_data_s = &vdrtmp[mflux];
  real *flux_s = &scalar_data_s[7*nthreads]; 
  
  if (idiag==0){
    for (int i=tid; i<7*nthreads; i+=nthreads){
      scalar_data_s[i] = 0.0;
    }
    for (int i=tid; i<4*mflux*nthreads; i+=nthreads){
      flux_s[i] = 0.0;
    }
    __syncthreads();
  }
  
  int mi = params.mi;
  
  //if(m>=mi)
  //	return;
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

  real* __restrict__ scalar_data = diagnosis->scalar_data;
  real* __restrict__ flux_data = diagnosis->flux_data;
/*
  real* __restrict__ eflux = diagnosis->eflux;
  real* __restrict__ rmarker = diagnosis->rmarker;
  real* __restrict__ dmark = diagnosis->dmark;
  real* __restrict__ dden = diagnosis->dden;
*/
  
#if !ONTHEFLY_PUSHAUX
#if !ASSUME_MZETA_EQUALS1
  const int mzeta = params.mzeta;
#endif
  const real __restrict__ *wzion  = aux_zion->wzion;
  const real __restrict__ *wpion  = aux_zion->wpion;
  const int *__restrict__ jtion0 = aux_zion->jtion0;
  const int * __restrict__ jtion1 = aux_zion->jtion1;
  const real *__restrict__ wtion0 = aux_zion->wtion0;
  const real *__restrict__ wtion1 = aux_zion->wtion1;
#else
  real delz, a_diff, zetamin;
#if GYRO_LOCAL_COMPUTE
  real tdum_lc, rad_lc, q_lc, rho_lc, dtheta_dx;
  real radcos_lc, rad_frac_lc;
  real pgyro_lc[4], tgyro_lc[4];
  real deltar, pi, q0, q1, q2, gyrorad_const_mult;
#else
  const real*  pgyro; const real*  tgyro;
  int ipjt;
#endif
  //real *pgyro, *tgyro; int ipjt;
  int mpsi_max;
  real wzt, r_diff, rdum, rhotmp, tflr, tdumtmp, tdumtmp2, tdum, tdum2;
  real wtion0tmp, wtion1tmp;
  int iptmp, im, im2, idxpg, jt, jttmp;
  int jtion0tmp, jtion1tmp, j00, j01;
  real r, wz1, wz0;
#endif
  int ip;
  
#if !USE_TEXTURE
  const real * __restrict__ evector = grid->evector;
#endif
  const real * __restrict__ pfluxpsi = grid->pfluxpsi;

  const int igrid_in = radial_decomp.igrid_in;
  const int ipsi_valid_in = radial_decomp.ipsi_valid_in;
  const int ipsi_valid_out = radial_decomp.ipsi_valid_out;
  const int nloc_over = radial_decomp.nloc_over;
  const real rho_max = radial_decomp.rho_max;

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

#if ASSUME_MZETA_EQUALS1
  //	real levector1[12], levector2[12];
#else
#if !ONTHEFLY_PUSHAUX
  const int*  __restrict__ kzion = aux_zion->kzion;
#endif 
#endif
  real zion1m,  zion2m,  zion3m,  zion4m,  zion5m, zion6m;
  
  real dtime;
  real sbound=1.0;
  if (nbound==0) sbound=0.0;
  real psimax=0.5*a1*a1;
  real psimin=0.5*a0*a0;
  real cmratio=qion/aion;
  real cinv=1.0/qion;
  real vthi=gyroradius*fabs(qion)/aion;
  real d_inv=real(mflux)/(a1-a0);

  for (int m=gid; m<mi; m+=np){

    if(irk==1) {
      dtime=0.5*tstep;
      if(tid<mflux)
	vdrtmp[tid] = 0.0;
      //if(istep ==1) {
#if PTX_STREAM_INTRINSICS
	TunedLoad<real,CS>::Ld(zion1m,zion1+m);
	TunedLoad<real,CS>::Ld(zion2m,zion2+m);
	TunedLoad<real,CS>::Ld(zion3m,zion3+m);
	TunedLoad<real,CS>::Ld(zion4m,zion4+m);
	TunedLoad<real,CS>::Ld(zion5m,zion5+m);
	
	TunedStore<real,CS>::St(zion1m,zion01+m);
	TunedStore<real,CS>::St(zion2m,zion02+m);
	TunedStore<real,CS>::St(zion3m,zion03+m);
	TunedStore<real,CS>::St(zion4m,zion04+m);
	TunedStore<real,CS>::St(zion5m,zion05+m);
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
      } else {
	if(tid<mflux)
	  vdrtmp[tid] = pfluxpsi[tid];
      }
    }
    __syncthreads();
    real wp0, wp1, wt00, wt01, wt10, wt11;
    int kk, larmor;
    
    int ii, ij1, ij2, ij3, ij4, idx1, idx2, idx3, idx4;

#if PTX_STREAM_INTRINSICS
    //    if((istep != 1)||(irk!=1)) {
    if (irk!=1) {
      TunedLoad<real,CS>::Ld(zion1m,zion1+m);
      TunedLoad<real,CS>::Ld(zion2m,zion2+m);
      TunedLoad<real,CS>::Ld(zion3m,zion3+m);
      TunedLoad<real,CS>::Ld(zion4m,zion4+m);
      TunedLoad<real,CS>::Ld(zion5m,zion5+m);
    }
    TunedLoad<real,CS>::Ld(zion6m,zion6+m);
#else
    zion1m = zion1[m];
    zion2m = zion2[m];
    zion3m = zion3[m];
    zion4m = zion4[m];
    zion5m = zion5[m];
    zion6m = zion6[m];
#endif
    real e1=0.0;
    real e2=0.0;
    real e3=0.0;

    //    kzion has five possible values values but can be much larger
#if ASSUME_MZETA_EQUALS1
    kk = 0;
#else
#if !ONTHEFLY_PUSHAUX
    kk = kzion[m];
#else 
    printf("ASSUME_MZETA_EQUALS1!=1 && ONTHEFLY_PUSHAUX==1 in gpu_push_kernel.cu is not available\n");
#endif

#endif
    
    //a_diff = a1 - a0;
    
#if ONTHEFLY_PUSHAUX
    a_diff = a1 - a0;
    zetamin = params.zetamin;
    real smu_inv = params.smu_inv;
#if GYRO_LOCAL_COMPUTE
    q0 = params.q0; q1 = params.q1; q2 = params.q2;
    pgyro_lc[0] = pgyro_lc[1] = pgyro_lc[2] = pgyro_lc[3] = 0.0;
    tgyro_lc[0] = tgyro_lc[1] = tgyro_lc[2] = tgyro_lc[3] = 0.0;
    gyrorad_const_mult = 102.0 *
      sqrt(params.aion * params.temperature)/(fabs(params.qion)*params.b0)/params.r0;
    deltar   = a_diff/mpsi;
    pi = params.pi;
#else
    pgyro = grid->pgyro; tgyro = grid->tgyro;
#endif
    
    real psitmp   = zion1m;
    real thetatmp = zion2m; 
    real zetatmp  = zion3m;
    real rhoi     = zion6m*smu_inv;
    
    delz  = params.delz;
    mpsi_max = mpsi-1;
    
#if SQRT_PRECOMPUTED
    r        = psitmp; 
#else
    r        = sqrt(2.0*psitmp);
#endif
    iptmp    = (int) ((r-a0)*delr+0.5);
    ip       = d_abs_min_int(mpsi, iptmp);

#if GPU_DEBUG
    if (ip<ipsi_valid_in || ip>ipsi_valid_out){
      printf("ip=%d ipsi_valid_in=%d ipsi_valid_out=%d\n", ip, ipsi_valid_in, ipsi_valid_out);
      CudaAssert(ip>=ipsi_valid_in);
      CudaAssert(ip<=ipsi_valid_out);
    } 
#endif

    jttmp    = (int) (thetatmp*pi2_inv*delt[ip]+0.5); 
    jt       = d_abs_min_int(mtheta[ip], jttmp);
 
    wzt      = (zetatmp-zetamin)*delz;
    wz1 = wzt - (real) kk;
    wz0 = 1.0 - wz1;
    r_diff   = r-a0;

#if GYRO_LOCAL_COMPUTE
    rad_lc = a0 + deltar * ip;

    dtheta_dx = 1.0/rad_lc;

    tdum_lc = (2 * pi * jt)/mtheta[ip];

    rad_frac_lc = rad_lc/a;

    q_lc = q0 + q1 * rad_frac_lc  + q2 * rad_frac_lc * rad_frac_lc;

    radcos_lc = 1.0 + rad_lc * cos(tdum_lc);

    pgyro_lc[2] = pgyro_lc[3] = radcos_lc * gyrorad_const_mult * gyrorad_const_mult * dtheta_dx;
    rho_lc = sqrt(2.0 * radcos_lc) * gyrorad_const_mult;
    pgyro_lc[0] = -rho_lc;
    pgyro_lc[1] =  rho_lc;
    tgyro_lc[2] = -rho_lc * dtheta_dx;
    tgyro_lc[3] =  -tgyro_lc[2];
#else
    ipjt     = igrid[ip]+jt;
#endif

#else
    real wz1=wzion[m];
    real wz0=1.0-wz1;
#endif
    
    int ind = m;
    for(larmor=0;larmor<4;larmor++) {

#if ONTHEFLY_PUSHAUX

#if GYRO_LOCAL_COMPUTE
      rhotmp = rhoi*pgyro_lc[larmor];
      if (fabs(rhotmp)>rho_max) {
        printf("rhotmp=%e rhoi=%e rho_max=%e pgyro=%e\n", rhotmp, rhoi, rho_max, pgyro_lc[larmor]);
        printf("warning: push sub reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro_lc[larmor], rhoi);
        rhotmp = (rhotmp/fabs(rhotmp))*rho_max;
        rhoi = rhotmp/pgyro_lc[larmor];
      }
      rdum = delr * d_abs_min_real(a_diff,
				   r_diff+rhotmp);
      tflr    = thetatmp + rhoi*tgyro_lc[larmor];
#else
      idxpg   = larmor + 4*(ipjt-igrid_in);   
      rhotmp = rhoi*pgyro[idxpg];
      if (fabs(rhotmp)>rho_max) {
        printf("rhotmp=%e rhoi=%e rho_max=%e pgyro=%e\n", rhotmp, rhoi, rho_max, pgyro[idxpg]); 
	printf("warning: push sub reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro[idxpg], rhoi);
	rhotmp = (rhotmp/fabs(rhotmp))*rho_max;
	rhoi = rhotmp/pgyro[idxpg];
      }
      rdum = delr * d_abs_min_real(a_diff,
			       	   r_diff+rhotmp);
      tflr    = thetatmp + rhoi*tgyro[idxpg];
#endif

      ii      = d_abs_min_int(mpsi_max, (int) rdum);
      
      wp1     = rdum - (real) ii;
      wp0     = 1.0  - wp1;
      
      /* Inner flux surface */
      /* Outer flux surface */
      im      = ii;
      im2     = ii + 1;
      
      tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
      tdumtmp2 = pi2_inv * (tflr - zetatmp * qtinv[im2]) + 10.0;
      
      tdum = (tdumtmp - (int) tdumtmp) * delt[im];
      tdum2 = (tdumtmp2 - (int) tdumtmp2) * delt[im2];
      
      j00 = d_abs_min_int(mtheta[im]-1, (int) tdum);
      j01 = d_abs_min_int(mtheta[im2]-1, (int) tdum2);
      
      jtion0tmp = igrid[im] + j00;
      jtion1tmp = igrid[im2] + j01;
      
      wtion0tmp = tdum - (real) j00;
      wtion1tmp = tdum2 - (real) j01;
      
      ij1 = jtion0tmp - igrid_in;
      ij3 = jtion1tmp - igrid_in;

      wp0 = 1.0 - wp1;
      wt10 = wtion0tmp;
      wt11 = wtion1tmp;
      wt01 = 1.0 - wt11;
      wt00 = 1.0 - wt10;
      ij2  = ij1 + 1;
      ij4  = ij3 + 1;      
#else
      ij1 = jtion0[ind] - igrid_in;
      ij3 = jtion1[ind] - igrid_in;
      wp0 = 1.0 - wpion[ind];
      wt00 = 1.0 - wtion0[ind];
      wt01 = 1.0 - wtion1[ind];
      ij2 = ij1 + 1;
      ij4 = ij3 + 1;
      wp1  = 1.0 - wp0;
      wt10 = 1.0 - wt00;
      wt11 = 1.0 - wt01;
#endif

#if GPU_DEBUG
      if (ij1< 0 || ij1 >= nloc_over || ij3 < 0 || ij3 >= nloc_over){
        printf("ij1=%d ij3=%d nloc_over=%d jtion0tmp=%d, jtion1tmp=%d igrid_in=%d im=%d\n", ij1, ij3, nloc_over, jtion0tmp, jtion1tmp, igrid_in, im);
        CudaAssert(ij1>=0);
        CudaAssert(ij1<nloc_over-1);
        CudaAssert(ij3>=0);
        CudaAssert(ij3<nloc_over-1);
      }
#endif

#if ASSUME_MZETA_EQUALS1
      idx1 = 6*ij1;
      //        idx2 = 6*ij2;
      idx3 = 6*ij3;
      //        idx4 = 6*ij4;
      
#define idx2 (idx1+6)
#define idx4 (idx2+6)

#if _DEBUG_GPU
      if((idx1+11>params.mgrid*(mzeta+1)*3) || (idx3+11>params.mgrid*(mzeta+1)*3))
	printf("out of bound access at particle %d\n",m);
#endif

#if 0
      levector1[0] = EVECTOR(idx1+0);levector1[1] = EVECTOR(idx1+1);
      levector1[2] = EVECTOR(idx1+2);levector1[3] = EVECTOR(idx1+3);
      levector1[4] = EVECTOR(idx1+4);levector1[5] = EVECTOR(idx1+5);
      levector1[6] = EVECTOR(idx1+6);levector1[7] = EVECTOR(idx1+7);
      levector1[8] = EVECTOR(idx1+8);levector1[9] = EVECTOR(idx1+9);
      levector1[10] = EVECTOR(idx1+10);levector1[11] = EVECTOR(idx1+11);
      
      levector2[0] = EVECTOR(idx3+0);levector2[1] = EVECTOR(idx3+1);
      levector2[2] = EVECTOR(idx3+2);levector2[3] = EVECTOR(idx3+3);
      levector2[4] = EVECTOR(idx3+4);levector2[5] = EVECTOR(idx3+5);
      levector2[6] = EVECTOR(idx3+6);levector2[7] = EVECTOR(idx3+7);
      levector2[8] = EVECTOR(idx3+8);levector2[9] = EVECTOR(idx3+9);
      levector2[10] = EVECTOR(idx3+10);levector2[11] = EVECTOR(idx3+11);
      
      e1 =e1+wp0*wt00*(wz0*levector1[0]+wz1*levector1[3]);
      e2 =e2+wp0*wt00*(wz0*levector1[1]+wz1*levector1[4]);
      e3 =e3+wp0*wt00*(wz0*levector1[2]+wz1*levector1[5]);
      e1 =e1+wp0*wt10*(wz0*levector1[6+0]+wz1*levector1[6+3]);
      e2 =e2+wp0*wt10*(wz0*levector1[6+1]+wz1*levector1[6+4]);
      e3 =e3+wp0*wt10*(wz0*levector1[6+2]+wz1*levector1[6+5]);
      
      e1 =e1+wp1*wt01*(wz0*levector2[0]+wz1*levector2[3]);
      e2 =e2+wp1*wt01*(wz0*levector2[1]+wz1*levector2[4]);
      e3 =e3+wp1*wt01*(wz0*levector2[2]+wz1*levector2[5]);
      e1 =e1+wp1*wt11*(wz0*levector2[6+0]+wz1*levector2[6+3]);
      e2 =e2+wp1*wt11*(wz0*levector2[6+1]+wz1*levector2[6+4]);
      e3 =e3+wp1*wt11*(wz0*levector2[6+2]+wz1*levector2[6+5]);
#endif
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
#else
      idx1 = 3*(mzeta+1)*ij1+3*kk;
      idx2 = 3*(mzeta+1)*ij2+3*kk;
      idx3 = 3*(mzeta+1)*ij3+3*kk;
      idx4 = 3*(mzeta+1)*ij4+3*kk;
      
      e1 = e1 + wp0 * wt00 * (wz0 *  EVECTOR(idx1+0)  
			      + wz1 * EVECTOR(idx1+3));
      e2 = e2 + wp0 * wt00 * (wz0 * EVECTOR(idx1+1) 
			      + wz1 * EVECTOR(idx1+4));
      e3 = e3 + wp0 * wt00 * (wz0 * EVECTOR(idx1+2) 
			      + wz1 * EVECTOR(idx1+5));
      
      e1 = e1 + wp0 * wt10 * (wz0 * EVECTOR(idx2+0)   
			      + wz1 * EVECTOR(idx2+3));
      e2 = e2 + wp0 * wt10 * (wz0 * EVECTOR(idx2+1)
			      + wz1 * EVECTOR(idx2+4));
      e3 = e3 + wp0 * wt10 * (wz0 * EVECTOR(idx2+2)
			      + wz1 * EVECTOR(idx2+5));
      
      e1 = e1 + wp1 * wt01 * (wz0 * EVECTOR(idx3+0) 
			      + wz1 * EVECTOR(idx3+3));
      e2 = e2 + wp1 * wt01 * (wz0 * EVECTOR(idx3+1)
			      + wz1 * EVECTOR(idx3+4));
      e3 = e3 + wp1 * wt01 * (wz0 * EVECTOR(idx3+2)
			      + wz1 * EVECTOR(idx3+5));
      
      e1 = e1 + wp1 * wt11 * (wz0 * EVECTOR(idx4+0)
			      + wz1 * EVECTOR(idx4+3));
      e2 = e2 + wp1 * wt11 * (wz0 * EVECTOR(idx4+1)
			      + wz1 * EVECTOR(idx4+4));
      e3 = e3 + wp1 * wt11 * (wz0 * EVECTOR(idx4+2)
			      + wz1 * EVECTOR(idx4+5));
#endif
      
      ind += mi;
    } 
    
    real wpi1=0.25*e1;
    real wpi2=0.25*e2;
    real wpi3=0.25*e3;

    real zion01m, zion02m, zion03m, zion04m, zion05m;
    
    if(irk ==1){
      zion01m = zion1m;
      zion02m = zion2m;
      zion03m = zion3m;
      zion04m = zion4m;
      zion05m = zion5m;
    } else {
#if PTX_STREAM_INTRINSICS
      TunedLoad<real,CS>::Ld(zion01m,zion01+m);
      TunedLoad<real,CS>::Ld(zion02m,zion02+m);
      TunedLoad<real,CS>::Ld(zion03m,zion03+m);
      TunedLoad<real,CS>::Ld(zion04m,zion04+m);
      TunedLoad<real,CS>::Ld(zion05m,zion05+m);
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
    // 4 Serial
    
    /* update GC position */
#if !ONTHEFLY_PUSHAUX
#if SQRT_PRECOMPUTED
    real r = zion1m;
#else
    real r = sqrt(2.0*zion1m);
#endif
#endif

    real rinv=1.0/r;
    
    const real q0 = params.q0;
    const real q1 = params.q1;
    const real q2 = params.q2;
    const real rw = params.rw;
    const real rc = params.rc;
    ii=d_abs_min_int(mpsi-1,int((r-a0)*delr));
    ip=d_abs_min_int(mflux-1,1+int((r-a0)*d_inv));
    wp0=real(ii+1)-(r-a0)*delr;
    wp1=1.0-wp0;
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
    //real wdot=(zion06[m]-paranl*zion5m)*(wdrive+wpara+wdrift);
    real wdot = (zion06[m]-zion5m)*(wdrive+wpara+wdrift);
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
    TunedStore<real,CS>::St(zion1m,zion1+m);
    zion2m = zion02m+dtime*tdot;
    zion3m = zion03m+dtime*zdot;
    zion4m = zion04m + dtime*rdot;
    TunedStore<real,CS>::St(zion4m,zion4+m);
    zion5m = zion05m + dtime*wdot;
    TunedStore<real,CS>::St(zion5m,zion5+m);
    real z1t = zion2m *pi2_inv+10;
    zion2m = pi2*(z1t-((int)z1t));
    TunedStore<real,CS>::St(zion2m,zion2+m);
    z1t = zion3m*pi2_inv+10;
    zion3m = pi2*(z1t - ((int)z1t));
    TunedStore<real,CS>::St(zion3m,zion3+m);
    
    if(irk==2) {
#if SQRT_PRECOMPUTED
      if((zion1m > a1)||(zion1m < a0)) {
#else
      if((zion1m > psimax)||(zion1m < psimin)) {
#endif
	TunedStore<real,CS>::St(zion01m,zion1+m);
	TunedStore<real,CS>::St(pi2-zion02m,zion2+m);
	TunedStore<real,CS>::St(zion03m,zion3+m);
	TunedStore<real,CS>::St(zion04m,zion4+m);
	TunedStore<real,CS>::St(zion05m,zion5+m);
	
	TunedStore<real,CS>::St(pi2-zion02m,zion02+m);
      } /*else {
	TunedStore<real,CS>::St(zion1m,zion01+m);
	TunedStore<real,CS>::St(zion2m,zion02+m);
	TunedStore<real,CS>::St(zion3m,zion03+m);
	TunedStore<real,CS>::St(zion4m,zion04+m);
	TunedStore<real,CS>::St(zion5m,zion05+m);
	} */
    }
    
#else
#if SQRT_PRECOMPUTED
    zion1m = max(1.0e-8*psimax,0.5*zion01m*zion01m + dtime*pdot);
    zion1[m] = sqrt(2.0*zion1m);
#else
    zion1[m] = max(1.0e-8*psimax,zion01m+dtime*pdot);
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
      }	else if(zion1[m] < a0) {
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
      //      ii = d_abs_min_int(mpsi, (int)((r-a0)*delr+0.5));
      
      real vdrenergy = vdr*rinv*(energy-1.5*aion*vthi*vthi*rtemi[ii])*zion05m;
      
      //        rmarker_s[ip*nthreads+tid] += zion06[m];
      //        eflux_s[ip*nthreads+tid] += vdrenergy;
      flux_s[ip*nthreads+tid] += vdrenergy; // eflux
      flux_s[mflux*nthreads+ip*nthreads+tid] += 1.0; //rmarker
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
	    //                   eflux_s[i*nthreads+tid] += eflux_s[i*nthreads+tid+half];
	    //                   rmarker_s[i*nthreads+tid] += rmarker_s[i*nthreads+tid+half];
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

bool findGridDims(dim3 &Dg, int nblocks, int max_dim)
{
  if((nblocks > max_dim) && (nblocks%max_dim==0)) {
    Dg.x = 	max_dim;
    Dg.y = 	nblocks/max_dim;
    return true;
  }
  int guess_a = int(sqrt(nblocks));
  int	guess_b = int(sqrt(nblocks));
  int best_error = abs(guess_a * guess_b - nblocks);
  int best_guess_a = guess_a;
  int best_guess_b = guess_b;
  guess_a++;
  while(guess_a < max_dim) {
    while(guess_b*guess_a > nblocks)
      guess_b--;
    int error = abs(guess_a * guess_b - nblocks);
    if(error< best_error) {
      best_error = error;
      best_guess_a = guess_a;
      best_guess_b = guess_b;
    }		
    guess_a++;
  }
  Dg.x = best_guess_a;
  Dg.y = best_guess_b;
  if(best_error==0)
    return true;
  else {
    if(guess_a * guess_b - nblocks<0)
      Dg.x+=1;
    return false;
  }
}

extern "C"
void call_gpu_push_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int idiag)
{
  int mi = (gtc_input->global_params).mi;
  int nthreads = gpu_kernel_input->nthreads*THREAD_BLOCK_OVERLOAD_FACTOR;
  gtc_global_params_t   *h_params   = &(gtc_input->global_params);
  gtc_diagnosis_data_t *h_diagnosis = &(gtc_input->diagnosis_data);
  gtc_diagnosis_data_t *d_diagnosis = &(gpu_kernel_input->d_diagnosis);
  gtc_radial_decomp_t *h_radial_decomp = &(gtc_input->radial_decomp);

  int mzeta=  h_params->mzeta; 	int nloc_over = h_radial_decomp->nloc_over;
  int mype = gtc_input->parallel_decomp.mype;
  
  dim3 Db = dim3(nthreads,1);
  dim3 Dg;
  int nblocks = (mi + nthreads - 1) / nthreads;
  
  // We need this transfer because we are not doing field computation on GPUs
  
  gtc_field_data_t *h_grid = &(gtc_input->field_data);
  gtc_field_data_t *d_grid = &(gpu_kernel_input->d_grid);
  gpu_timer_start(gpu_kernel_input);
  
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_grid->evector, h_grid->evector, 3*(mzeta+1)*nloc_over*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_grid->pfluxpsi, h_grid->pfluxpsi, MFLUX*sizeof(real), cudaMemcpyHostToDevice));	 

  /**************** copy and reset diagnosis data****************/
  if (idiag==0){
    // notice: host-device memory copy of 64KB or less is NOT synchronous
    CUDA_SAFE_CALL(cudaMemcpy((void*)(d_diagnosis->scalar_data), (h_diagnosis->scalar_data), 16*sizeof(real),cudaMemcpyHostToDevice));
    //CUDA_SAFE_CALL(cudaMemset(d_diagnosis->eflux, 0, 4*MFLUX*sizeof(real)));
    CUDA_SAFE_CALL(cudaMemset(d_diagnosis->flux_data, 0, 4*MFLUX*sizeof(real)));
  }
  gpu_kernel_input->gpu_timing.memtransfer_push_time +=  gpu_timer_measure(gpu_kernel_input);
  
  if(nblocks < gpu_kernel_input->deviceProp.maxGridSize[0])
    Dg.x = nblocks;
  else {
    if(!findGridDims(Dg,nblocks,gpu_kernel_input->deviceProp.maxGridSize[0])) ;
    //	printf("WARNING the number of blocks is changed from %d to %d\n", nblocks, (Dg.x * Dg.y));
  }
  
  int mi_per_thread = gpu_kernel_input->charge_mi_per_thread;
  nthreads = gpu_kernel_input->nthreads/2;
  int mp = gpu_kernel_input->deviceProp.multiProcessorCount;
  int m = (mi + nthreads*mp - 1)/ (nthreads*mp);
  m = (m + mi_per_thread - 1)/mi_per_thread;
  nblocks = mp*m;
  mi_per_thread = (mi + nblocks*nthreads - 1)/ mi_per_thread;
  
  //printf("start pushi at mype=%d\n", mype);
  int shared_buffer = MFLUX*sizeof(real) + (7*nthreads+4*MFLUX*nthreads)*sizeof(real);
  gpu_pushi_kernel<<< nblocks, nthreads, shared_buffer>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_aux_zion,gpu_kernel_input->ptr_d_grid,  gpu_kernel_input->ptr_d_diagnosis, gpu_kernel_input->irk,gpu_kernel_input->istep, idiag);
  cudaError_t lasterror = cudaGetLastError();
  if(lasterror != cudaSuccess) {
    printf("Error in launching gpu_pushi_kernel routine: %s\n", cudaGetErrorString(lasterror));
  }
  //printf("end pushi at mype=%d\n", mype);
  gpu_kernel_input->gpu_timing.device_push_time += gpu_timer_measure(gpu_kernel_input);

  /********************* copy diagnosis data back to host********************/
  if (idiag==0){
    // notice: host-device memory copy of 64KB or less is NOT synchronous
    CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->scalar_data), (d_diagnosis->scalar_data), 16*sizeof(real),cudaMemcpyDeviceToHost));
    /*
      CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->eflux), (d_diagnosis->eflux), MFLUX*sizeof(real),cudaMem \
      cpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->rmarker), (d_diagnosis->rmarker), MFLUX*sizeof(real),cud \
      aMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->dmark), (d_diagnosis->dmark), MFLUX*sizeof(real),cudaMem \
      cpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->dden), (d_diagnosis->dden), MFLUX*sizeof(real),cud	\
      aMemcpyDeviceToHost));
    */
    CUDA_SAFE_CALL(cudaMemcpy((void*)(h_diagnosis->flux_data), (d_diagnosis->flux_data), 4*MFLUX*sizeof(real),cudaMemcpyDeviceToHost));
  }
  gpu_kernel_input->gpu_timing.memtransfer_push_time +=  gpu_timer_measure_end(gpu_kernel_input);
  
}

