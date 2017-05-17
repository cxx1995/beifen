#ifdef MULTIPLE_FILE
#define GPU_KERNEL

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
#endif

#define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!", blockIdx.x, threadIdx.x); }

__noinline__ __device__ int fixed_d(double d, int exponent)
{
  if (d == 0) return 0L;
  
  long long *p_d = (long long *)&d;
  long long exp_mask = 0x7FF0000000000000;
  long long significand_mask = 0x000FFFFFFFFFFFFF;
  long long positive_prefix = 0x0010000000000000;
  int  r;
  
  int  e = ((*p_d) & exp_mask ) >> 52;
  long long s;
  s = ((*p_d) & significand_mask) | positive_prefix; 
  
  int shift_amt = exponent - e;
  if (shift_amt > 0) s = s >> shift_amt;
  else s = s << (-shift_amt);
  
  r = *(((int *)&s)+1);
  if ((*p_d) < 0) r = -r;
  
  //	printf ("%08x %d %d\n", r, e, exponent);
  return r;
}

__noinline__ __device__  double fp_d (int s, int e)
{
  int negative = 0;
  double r;
  long long sl;
  
  if (s == 0) return 0.0;
  else if (s <0) { negative = 1; s=-s;}
  
  sl = ((long long)s) << 32;	
  long long one = 0x0010000000000000;
  
  //	printf ("%016lx %d\n", sl, e);
  if ( sl < one) { 
    do {
      sl = sl << 1; e--;
      //	printf ("%016lx %d\n", sl, e);
    } while(sl < one);
  }
  else if (sl > one){
    do {
      sl = sl >> 1; e++;
      //	printf ("%016lx %d\n", sl, e);
    } while(sl >= one);
    sl = sl<<1; e--;
  }
  //	printf ("%016lx %d\n", sl, e);
  long long exp = (long long) e << 52;
  //	printf ("%016lx %d\n", sl, e);
  sl = (sl & (~one)) | exp;
  
  r = *((double *)&sl);
  if (negative) return -r;
  else return r;
}
/*
  __noinline__ __device__ void atomicDPupdate(double *addr,double delta)
  {
  double  copied;
  double  updated;
  unsigned long long int *c_ptr = (unsigned long long int *)&copied, *n_ptr = (unsigned long long int *)&updated;
  do {
  copied = *addr;
  updated = copied + delta;
  } while (  atomicCAS((unsigned long long int *)addr, *c_ptr, *n_ptr) != *c_ptr);
  }
*/

__noinline__ __device__ double atomicDPupdate(double *address, double val)
{
  double old = *address, assumed;
  do { assumed = old;
    old = __longlong_as_double( atomicCAS((unsigned long long int*)address,
					  __double_as_longlong(assumed),
					  __double_as_longlong(val + assumed)));
  } while (assumed != old);
  return old;
}

__noinline__ __device__ void atomicFPupdate(float *addr,float delta)
{
  float  copied;
  float  updated;
  int *c_ptr = (int *)&copied, *n_ptr = (int *)&updated;
  
  do {
    copied = *addr;
    updated = copied + delta;
  } while (  atomicCAS((int *)addr, *c_ptr, *n_ptr) != *c_ptr);
  
}

__device__ inline int d_abs_min_int(int arg1, int arg2) {
  
  int minval, retval;
  minval = (arg1 < arg2) ? arg1 : arg2;
  retval = (minval > 0) ? minval : 0;
  return retval;
}

__device__ inline real d_abs_min_real(real arg1, real arg2) {
  
  real minval, retval;
  minval = (arg1 < arg2) ? arg1 : arg2;
  retval = (minval > 0) ? minval : 0;
  return retval;
}

__device__ inline void swap(int & a, int & b)
{
  int tmp = a;
  a = b;
  b = tmp;
}

// n particles per thread 
__global__ static void 
__launch_bounds__(THREAD_BLOCK/*maxThreadsPerBlock*/, 1/*minBlocksPerMultiprocessor*/)
  gpu_charge_multi(gtc_field_data_t *grid, gtc_particle_data_t *zion, gtc_aux_particle_data_t *aux_zion, int mi_per_thread){
  
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nblocks = gridDim.x;
  const int nthreads = blockDim.x;
  const int gid = tid + bid*nthreads;
  const int np = nblocks * nthreads;
  
  const real* __restrict__  zion0; const real* __restrict__ zion1; const real* __restrict__ zion2;
  const real* __restrict__ zion4; const real* __restrict__ zion5;
  
#if GYRO_LOCAL_COMPUTE
  real tdum_lc, rad_lc, q_lc, rho_lc, dtheta_dx;
  real radcos_lc, rad_frac_lc;
  real pgyro_lc[4], tgyro_lc[4];
  real deltar, pi, a, q0, q1, q2, gyrorad_const_mult;
#else
  const real* __restrict__ pgyro; const real* __restrict__ tgyro;
  int ipjt, idx1;
#endif

#if !ONTHEFLY_PUSHAUX
#if !ASSUME_MZETA_EQUALS1
  int*  __restrict__ kzion;
#endif
  int* __restrict__ jtion0; int* __restrict__ jtion1; 
  real*  __restrict__ wtion0; real* __restrict__ wtion1; 
  real*  __restrict__ wzion; real*  __restrict__ wpion;
#endif
  
  real *densityi;
  
#if USE_CONSTANTMEM
#else
  const int*  igrid; const real*  delt; 
  const real*  qtinv; const int*  mtheta; 
#endif
  int mi; real smu_inv; real a0;
  real a1; real delr; real delz; int mpsi;
  real pi2_inv; real zetamin; int mzeta;  
  
  /* temporary variables */
  real psitmp, thetatmp, zetatmp, weight, rhoi, rhotmp, rho_max, r, wzt, wz1, wz0;
  int iptmp, ip, jttmp, jt, kk, ii, im, im2, larmor, idx;
  real rdum, wp1, wp0, tflr;
  int j01, j00, jtion0tmp, jtion1tmp, ij1, ij2, mpsi_max;
  real tdumtmp, tdum, tdumtmp2, tdum2, wt10, wt00, wt01, wt11, wtion0tmp, wtion1tmp;
  real r_diff, a_diff;

  int igrid_in, ipsi_in, ipsi_out, nloc_over, ipsi_valid_in, ipsi_valid_out;

  mpsi = params.mpsi;
  a0 = params.a0; a1=params.a1;
  delr = params.delr;
  delz = params.delz;
  smu_inv = params.smu_inv;
  zetamin = params.zetamin; mzeta = params.mzeta;
  pi2_inv = params.pi2_inv; 
  mi = params.mi;

  igrid_in = radial_decomp.igrid_in;
  ipsi_in = radial_decomp.ipsi_in;
  ipsi_out = radial_decomp.ipsi_out;
  nloc_over = radial_decomp.nloc_over;
  ipsi_valid_in = radial_decomp.ipsi_valid_in;
  ipsi_valid_out = radial_decomp.ipsi_valid_out;
  rho_max = radial_decomp.rho_max;

#if !ONTHEFLY_PUSHAUX
#if !ASSUME_MZETA_EQUALS1
  kzion = aux_zion->kzion;
#endif
  wzion = aux_zion->wzion; 
  jtion0 = aux_zion->jtion0; 
  jtion1 = aux_zion->jtion1; 
  wtion0 = aux_zion->wtion0; 
  wtion1 = aux_zion->wtion1; 
  wpion = aux_zion->wpion; 
#endif
    
  densityi = grid->densityi;
  
  a_diff   = a1-a0;
  mpsi_max = mpsi-1;
#if GYRO_LOCAL_COMPUTE
  a = params.a;
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
  
  zion0 = zion->z0; zion1 = zion->z1; zion2 = zion->z2; zion4 = zion->z4; 
  zion5 = zion->z5;
  
  for (int m=gid; m<mi; m+=np){

    psitmp   = zion0[m];
    thetatmp = zion1[m]; 
    zetatmp  = zion2[m];
    weight   = zion4[m];
    rhoi     = zion5[m]*smu_inv;
    
#if SQRT_PRECOMPUTED
    r        = psitmp; 
#else
    r        = sqrt(2.0*psitmp);
#endif
    
    iptmp    = (int) ((r-a0)*delr+0.5);
    ip       = d_abs_min_int(mpsi, iptmp);
    
#if GPU_DEBUG
    CudaAssert(ip>=ipsi_valid_in);
    CudaAssert(ip<=ipsi_valid_out);
#endif

    jttmp    = (int) (thetatmp*pi2_inv*delt[ip]+0.5); 
    jt       = d_abs_min_int(mtheta[ip], jttmp);
    
    wzt      = (zetatmp-zetamin)*delz;
    kk       = d_abs_min_int(mzeta-1, (int) wzt);
    
#if !ONTHEFLY_PUSHAUX
#if !ASSUME_MZETA_EQUALS1 
    kzion[m] = kk;
#endif
    wzion[m] = wzt - (real) kk;
#endif

    wz1      = weight * (wzt - (real) kk);
    wz0      = weight - wz1;
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

    for (larmor = 0; larmor < 4; larmor++) {
#if !ONTHEFLY_PUSHAUX
      idx     = larmor * mi + m;
#endif

      /* particle position in r and theta */
#if GYRO_LOCAL_COMPUTE
      rhotmp = rhoi*pgyro_lc[larmor];
      if (fabs(rhotmp)>rho_max) {
	printf("warning: reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro_lc[larmor], rhoi);
	rhotmp = (rhotmp/fabs(rhotmp))*rho_max;
	rhoi = rhotmp/pgyro_lc[larmor];
      }
      rdum = delr * d_abs_min_real(a_diff,
				 r_diff+rhotmp);
      tflr    = thetatmp + rhoi*tgyro_lc[larmor];
#else
      idx1    = larmor + 4*(ipjt-igrid_in);
      rhotmp = rhoi*pgyro[idx1];
      if (fabs(rhotmp)>rho_max) {
	printf("warning: reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro[idx1\
											     ], rhotmp);
	rhotmp = (rhotmp/fabs(rhotmp))*rho_max;
	rhoi = rhotmp/pgyro[idx1];
      }
      rdum = delr * d_abs_min_real(a_diff,
				 r_diff+rhotmp);
      tflr    = thetatmp + rhoi*tgyro[idx1];
#endif
      
      ii      = d_abs_min_int(mpsi_max, (int) rdum);
#if GPU_DEBUG
      CudaAssert(ii>=ipsi_in);
      CudaAssert(ii<=ipsi_out-1);
#endif
      wp1     = rdum - (real) ii;
      wp0     = 1.0  - wp1;
            
      /* Inner flux surface */
      im = ii;
      tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
      tdum = (tdumtmp - (int) tdumtmp) * delt[im];
      j00 = d_abs_min_int(mtheta[im]-1, (int) tdum);
      jtion0tmp = igrid[im] + j00;
      wtion0tmp = tdum - (real) j00;
      
      /* Outer flux surface */
      im2 = ii + 1;
      tdumtmp2 = pi2_inv * (tflr - zetatmp * qtinv[im2]) + 10.0;
      tdum2 = (tdumtmp2 - (int) tdumtmp2) * delt[im2];
      j01 = d_abs_min_int(mtheta[im2]-1, (int) tdum2);
      jtion1tmp = igrid[im2] + j01;
      wtion1tmp = tdum2 - (real) j01;

#if GPU_DEBUG
      CudaAssert(jtion0tmp-igrid_in>=0);
      CudaAssert(jtion0tmp-igrid_in<nloc_over-1);
      CudaAssert(jtion1tmp-igrid_in>=0);
      CudaAssert(jtion1tmp-igrid_in<nloc_over-1);
#endif

#if !ONTHEFLY_PUSHAUX
      wpion [idx] = wp1;
      jtion0[idx] = jtion0tmp;
      jtion1[idx] = jtion1tmp;
      wtion0[idx] = wtion0tmp;
      wtion1[idx] = wtion1tmp;
#endif
      
#if DO_CHARGE_UPDATES            
      /* Update charge densities */
      wt10 = wp0 * wtion0tmp;
      wt00 = wp0 - wt10;
      
      wt11 = wp1 * wtion1tmp;
      wt01 = wp1 - wt11;

      ij1 = kk + (mzeta+1)*(jtion0tmp-igrid_in);
      
      atomicDPupdate(densityi+ij1,   wz0*wt00);
      atomicDPupdate(densityi+ij1+1, wz1*wt00);
      atomicDPupdate(densityi+ij1+mzeta+1, wz0*wt10);
      atomicDPupdate(densityi+ij1+mzeta+2, wz1*wt10);

      ij2 = kk + (mzeta+1)*(jtion1tmp-igrid_in);
      atomicDPupdate(densityi+ij2,   wz0*wt01);
      atomicDPupdate(densityi+ij2+1, wz1*wt01);
      atomicDPupdate(densityi+ij2+mzeta+1, wz0*wt11);
      atomicDPupdate(densityi+ij2+mzeta+2, wz1*wt11);
#endif 
    }
    }    
}

// n particles per thread 
// cooperative threading

__global__ static void 
__launch_bounds__(THREAD_BLOCK/*maxThreadsPerBlock*/, 1/*minBlocksPerMultiprocessor*/)
  gpu_charge_cooperative (gtc_field_data_t *grid, gtc_particle_data_t *zion,gtc_aux_particle_data_t *aux_zion, int mi_per_thread){

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nblocks = gridDim.x;
    const int nthreads = blockDim.x;
    const int gid = tid + bid*nthreads;
    const int np = nblocks * nthreads;

#if !FIXME
    __syncthreads();
#endif
    
    extern __shared__ int shared_buffer[];
    int *update_idx = shared_buffer;
    real *update_val = (real *)&shared_buffer[nthreads*4];
    
    //printf("inside the charge kernel 1\n");
    
    //	__shared__	int update_idx[4*THREAD_BLOCK];
    //	__shared__	double update_val[4*THREAD_BLOCK];
    const real* __restrict__ zion0; const real* __restrict__ zion1; const real* __restrict__ zion2;
    const real* __restrict__ zion4; const real*__restrict__  zion5;

#if GYRO_LOCAL_COMPUTE
    real tdum_lc, rad_lc, q_lc, rho_lc, dtheta_dx;
    real radcos_lc, rad_frac_lc;
    real pgyro_lc[4], tgyro_lc[4];
    real deltar, pi, a, q0, q1, q2, gyrorad_const_mult;
#else
    const real* __restrict__  pgyro; const real* __restrict__  tgyro;
    int ipjt, idx1;
#endif

#if !ONTHEFLY_PUSHAUX
#if !ASSUME_MZETA_EQUALS1
    int*  __restrict__ kzion;
#endif
    int* __restrict__ jtion0; int* __restrict__ jtion1; 
    real* __restrict__ wtion0; real*  __restrict__ wtion1;
    real*  __restrict__ wzion; real*  __restrict__ wpion;
#endif
    
    real * __restrict__ densityi;

#if USE_CONSTANTMEM
#else
    const int*  __restrict__ igrid; const real* __restrict__  delt; 
    const real*  __restrict__ qtinv; const int*  __restrict__ mtheta; 
#endif
    int mi; real smu_inv; real a0;
    real a1; real delr; real delz; int mpsi;
    real pi2_inv; real zetamin; int mzeta;
    
    /* temporary variables */
    real psitmp, thetatmp, zetatmp, weight, rhoi, rhotmp, rho_max, r, wzt, wz1, wz0;
    int iptmp, ip, jttmp, jt, kk, ii, im, larmor, idx;
    real rdum, wp1, wp0, tflr;
    int j01, j00, jtion0tmp, jtion1tmp, ij1, ij2, mpsi_max;
    real tdumtmp, tdum, wt10, wt00, wt01, wt11, wtion0tmp, wtion1tmp;
    real r_diff, a_diff;
    int stride;
    int last_gthreads, last_threads, last_block, total_iterations;

    int igrid_in,ipsi_in, ipsi_out;
    
    mpsi = params.mpsi;
    a0 = params.a0; a1=params.a1;
    delr = params.delr;
    delz = params.delz;
    smu_inv = params.smu_inv;
    zetamin = params.zetamin; mzeta = params.mzeta;
    pi2_inv = params.pi2_inv; 
    mi = params.mi;

    igrid_in = radial_decomp.igrid_in;
    ipsi_in = radial_decomp.ipsi_in;
    ipsi_out = radial_decomp.ipsi_out;    
    rho_max = radial_decomp.rho_max;

#if !ONTHEFLY_PUSHAUX
#if !ASSUME_MZETA_EQUALS1
    kzion = aux_zion->kzion;
#endif
    wzion = aux_zion->wzion; 
    jtion0 = aux_zion->jtion0; jtion1 = aux_zion->jtion1; 
    wtion0 = aux_zion->wtion0; wtion1 = aux_zion->wtion1; 
    wpion = aux_zion->wpion; 
#endif
    densityi = grid->densityi;
    
#if USE_CONSTANTMEM
#else
    igrid = grid->igrid; qtinv = grid->qtinv; 
    mtheta = grid->mtheta; delt = grid->delt;
#endif
    
    a_diff   = a1-a0;
    mpsi_max = mpsi-1;
#if GYRO_LOCAL_COMPUTE
    a = params.a;
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
    
    zion0 = zion->z0; zion1 = zion->z1; zion2 = zion->z2; zion4 = zion->z4; 
    zion5 = zion->z5;
 
    last_gthreads = mi % (nthreads * nblocks);
    last_threads = last_gthreads % nthreads;
    last_block = last_gthreads / nthreads;
    if (last_threads == 0) last_block--;
    
    total_iterations = (bid <= last_block) ? mi_per_thread : (mi_per_thread -1);
    stride = nthreads;
    for (int iter=0, m=gid; m<mi; iter++, m+=np) {
#if PTX_STREAM_INTRINSICS
      TunedLoad<real,CS>::Ld(psitmp,zion0+m);
      TunedLoad<real,CS>::Ld(thetatmp,zion1+m);
      TunedLoad<real,CS>::Ld(zetatmp,zion2+m);
      TunedLoad<real,CS>::Ld(weight,zion4+m);
      TunedLoad<real,CS>::Ld(rhoi,zion5+m);
      rhoi *= smu_inv;
#else
      psitmp   = zion0[m];
      thetatmp = zion1[m]; 
      zetatmp  = zion2[m];
      weight   = zion4[m];
      rhoi     = zion5[m]*smu_inv;
#endif

#if SQRT_PRECOMPUTED
      r        = psitmp; 
#else
      r        = sqrt(2.0*psitmp);
#endif
      
      iptmp    = (int) ((r-a0)*delr+0.5);
      ip       = d_abs_min_int(mpsi, iptmp);
      
      jttmp    = (int) (thetatmp*pi2_inv*delt[ip]+0.5); 
      jt       = d_abs_min_int(mtheta[ip], jttmp);
      
      wzt      = (zetatmp-zetamin)*delz;
      kk       = d_abs_min_int(mzeta-1, (int) wzt);
            
#if !ONTHEFLY_PUSHAUX
#if !ASSUME_MZETA_EQUALS1
      kzion[m] = kk;
#endif
      wzion[m] = wzt - (real) kk;
#endif
      wz1      = weight * (wzt - (real) kk);
      wz0      = weight - wz1;
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
      for (larmor = 0; larmor < 4; larmor++) {
#if !ONTHEFLY_PUSHAUX
	idx     = larmor * mi + m;
#endif

#if GYRO_LOCAL_COMPUTE
        rhotmp = rhoi*pgyro_lc[larmor];
        if (fabs(rhotmp)>rho_max) {
          printf("warning: reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro_lc[larmor], rhoi);
          rhotmp = (rhotmp/fabs(rhotmp))*rho_max; 
   	  rhoi = rhotmp/pgyro_lc[larmor];
        }
        rdum = delr * d_abs_min_real(a_diff,r_diff+rhotmp);
        tflr    = thetatmp + rhoi*tgyro_lc[larmor];
#else
        idx1    = larmor + 4*(ipjt-igrid_in);
        rhotmp = rhoi*pgyro[idx1];
        if (fabs(rhotmp)>rho_max) {
          printf("warning: reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro[idx1], rhoi);
          rhotmp = (rhotmp/fabs(rhotmp))*rho_max;
          rhoi = rhotmp/pgyro[idx1];
        }
        rdum = delr * d_abs_min_real(a_diff,r_diff+rhotmp);
        tflr    = thetatmp + rhoi*tgyro[idx1];
#endif
	
	ii      = d_abs_min_int(mpsi_max, (int) rdum);

#if GPU_DEBUG
	CudaAssert(ii>=ipsi_in);
	CudaAssert(ii<=ipsi_out-1);
#endif

	wp1     = rdum - (real) ii;
	wp0     = 1.0  - wp1;
        	
	/* Inner flux surface */
	im = ii;
	tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
	tdum = (tdumtmp - (int) tdumtmp) * delt[im];
	j00 = d_abs_min_int(mtheta[im]-1, (int) tdum);
	jtion0tmp = igrid[im] + j00;
	wtion0tmp = tdum - (real) j00;
	
	/* Outer flux surface */
	im = ii + 1;
	tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
	tdum = (tdumtmp - (int) tdumtmp) * delt[im];
	j01 = d_abs_min_int(mtheta[im]-1, (int) tdum);
	jtion1tmp = igrid[im] + j01;
	wtion1tmp = tdum - (real) j01;
	
#if !ONTHEFLY_PUSHAUX
#if PTX_STREAM_INTRINSICS
	TunedStore<int,CS>::St(jtion0tmp,jtion0+idx);
	TunedStore<int,CS>::St(jtion1tmp,jtion1+idx);
	TunedStore<real,CS>::St(wp1,wpion+idx);
	TunedStore<real,CS>::St(wtion0tmp,wtion0+idx);
	TunedStore<real,CS>::St(wtion1tmp,wtion1+idx);
#else
	wpion [idx] = wp1;
	jtion0[idx] = jtion0tmp;
	jtion1[idx] = jtion1tmp;
	wtion0[idx] = wtion0tmp;
	wtion1[idx] = wtion1tmp;
#endif
#endif
	
#if DO_CHARGE_UPDATES            
	/* Update charge densities */
	wt10 = wp0 * wtion0tmp;
	wt00 = wp0 - wt10;
	
	wt11 = wp1 * wtion1tmp;
	wt01 = wp1 - wt11;
	
	ij1 = kk + (mzeta+1)*(jtion0tmp-igrid_in);
#if _DEBUG_GPU
	int start = 1;	
	if(start&& (isnan(wz0) || isnan(wt00) || isnan(wz1) || isnan(wt10))) {
	  printf("wt10: %f, wt00: %f, wt11 %f, wt01 %f idx, ij1 %d %d\n",wt10,wt00,wt11,wt01,idx,ij1);
	  printf("wtion1tmp: %f,  wp1: %f\n",wtion0tmp,wp1);
	  printf("zions %f %f %f %f %f\n",zion0[m],zion1[m],zion2[m],zion4[m],zion5[m]);
	  start = 0;
	  
	}
#endif
	
	if ((iter == (total_iterations - 1)) && (bid == last_block) && last_threads){
	  
	  atomicDPupdate(densityi+ij1,   wz0*wt00);
	  atomicDPupdate(densityi+ij1+1, wz1*wt00);
	  atomicDPupdate(densityi+ij1+mzeta+1, wz0*wt10);
	  atomicDPupdate(densityi+ij1+mzeta+2, wz1*wt10);
	  
	  ij2 = kk + (mzeta+1)*(jtion1tmp-igrid_in);
	  atomicDPupdate(densityi+ij2,   wz0*wt01);
	  atomicDPupdate(densityi+ij2+1, wz1*wt01);
	  atomicDPupdate(densityi+ij2+mzeta+1, wz0*wt11);
	  atomicDPupdate(densityi+ij2+mzeta+2, wz1*wt11);
	  
	}
	else {
	  update_idx[4*tid]=ij1;
	  update_idx[4*tid+1]=ij1+1;
	  update_idx[4*tid+2]=ij1+mzeta+1;
	  update_idx[4*tid+3]=ij1+mzeta+2;
	  update_val[4*tid]= wz0*wt00;
	  update_val[4*tid+1]= wz1*wt00;
	  update_val[4*tid+2]= wz0*wt10;
	  update_val[4*tid+3]= wz1*wt10;
	  __syncthreads();
	  
	  atomicDPupdate(densityi+update_idx[tid], update_val[tid]);
	  atomicDPupdate(densityi+update_idx[stride+tid], update_val[stride+tid]);
	  atomicDPupdate(densityi+update_idx[2*stride+tid], update_val[2*stride+tid]);
	  atomicDPupdate(densityi+update_idx[3*stride+tid], update_val[3*stride+tid]);
	  
	  ij2 = kk + (mzeta+1)*(jtion1tmp-igrid_in);
	  update_idx[4*tid]=ij2;
	  update_idx[4*tid+1]=ij2+1;
	  update_idx[4*tid+2]=ij2+mzeta+1;
	  update_idx[4*tid+3]=ij2+mzeta+2;
	  update_val[4*tid]= wz0*wt01;
	  update_val[4*tid+1]= wz1*wt01;
	  update_val[4*tid+2]= wz0*wt11;
	  update_val[4*tid+3]= wz1*wt11;
	  __syncthreads();
	  
	  atomicDPupdate(densityi+update_idx[tid], update_val[tid]);
	  atomicDPupdate(densityi+update_idx[stride+tid], update_val[stride+tid]);
	  atomicDPupdate(densityi+update_idx[2*stride+tid], update_val[2*stride+tid]);
	  atomicDPupdate(densityi+update_idx[3*stride+tid], update_val[3*stride+tid]); 
	}
#endif
      }
      __syncthreads();
      
    }    
}


__global__ static void memreset(real *array, int size) {
  const int tidx = threadIdx.x;
  const int tid = threadIdx.y *blockDim.x + tidx;
  const int bid = blockIdx.x +gridDim.x * blockIdx.y;
  const int nthreads = blockDim.x * blockDim.y;
  const int nblocks = gridDim.x * gridDim.y;
  int step = nthreads * nblocks;
  int i;
  for(i=bid*nthreads;i<size;i+=step)
    array[i+tid] = 0;
}

extern "C"
void call_gpu_charge_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int idiag)
{
  int mi_per_thread = gpu_kernel_input->charge_mi_per_thread;
  int nthreads = gpu_kernel_input->nthreads;
  gtc_global_params_t   *h_params   = &(gtc_input->global_params);
  gtc_field_data_t *d_grid = &(gpu_kernel_input->d_grid);
  gtc_field_data_t *h_grid = &(gtc_input->field_data);
  gtc_radial_decomp_t *h_radial_decomp = &(gtc_input->radial_decomp);
    
  gpu_timer_start(gpu_kernel_input);
  int mzeta=h_params->mzeta; int nloc_over = h_radial_decomp->nloc_over;
  int mi = h_params->mi;
  int mype = gtc_input->parallel_decomp.mype;
 
  int mp = gpu_kernel_input->deviceProp.multiProcessorCount;
  int m = (mi + nthreads*mp -1) / (nthreads*mp);
  m = (m + mi_per_thread -1)/ mi_per_thread;
  int nblocks = mp * m;
  mi_per_thread = (mi + nblocks*nthreads -1)/(nblocks*nthreads);
  // 	CUDA_SAFE_CALL(cudaMemset(d_grid->densityi, 0, (mzeta+1)*nloc_over*sizeof(wreal)));
  memreset<<< mp, 512>>>(d_grid->densityi, (mzeta+1)*nloc_over);
  gpu_kernel_input->gpu_timing.memtransfer_charge_time += gpu_timer_measure(gpu_kernel_input);
  
#if _DEBUG_GPU 
  fprintf(stderr,"mi_per_thread %d, nblock: %d\n", mi_per_thread,nblocks);
  fflush(stderr);
#endif
  
 
#if COOPERATIVE_THREADING
  int	shared_buffer_sz = nthreads*4*(sizeof(int)+sizeof(real));
  gpu_charge_cooperative<<< nblocks, nthreads,shared_buffer_sz>>>(gpu_kernel_input->ptr_d_grid, gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_aux_zion, mi_per_thread);
#else
  gpu_charge_multi<<< nblocks, nthreads>>>(gpu_kernel_input->ptr_d_grid, gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_aux_zion, mi_per_thread);
#endif

  /************* overlap GPU computation with CPU I/O work *************/
  if (idiag==0) diagnosis(gtc_input);
  
  cudaError_t lasterror = cudaGetLastError();
  if(lasterror != cudaSuccess)
    printf("Error in launching gpu_charge_ routine: %s\n", cudaGetErrorString(lasterror));
  gpu_kernel_input->gpu_timing.device_charge_time += gpu_timer_measure(gpu_kernel_input);
  
  CUDA_SAFE_CALL(cudaMemcpy((void*)(h_grid->densityi), (d_grid->densityi), (mzeta+1)*nloc_over* sizeof(wreal), cudaMemcpyDeviceToHost)); 
  gpu_kernel_input->gpu_timing.memtransfer_charge_time += gpu_timer_measure_end(gpu_kernel_input);
  
#define STREAM_ANALYSIS 0
  
#if STREAM_ANALYSIS
  if(gtc_input->parallel_decomp.mype == 0) {
    int irk = gpu_kernel_input->irk;
    int istep = gpu_kernel_input->istep;
    static int irk2_save = 0;
    static int irk1_save = 0;
    gtc_aux_particle_data_t *h_aux_zion = &(gtc_input->aux_particle_data);
    gtc_aux_particle_data_t *d_aux_zion =  &(gpu_kernel_input->d_aux_zion);
    
    if((irk==2) && (istep%RADIAL_THETA_BIN_PERIOD==0)) {
      if(!irk2_save) {
	CUDA_SAFE_CALL(cudaMemcpy(h_aux_zion->jtion0,(void *)d_aux_zion->jtion0, 4*(mi)*sizeof(int) , cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_aux_zion->jtion1, (void *)d_aux_zion->jtion1, 4*(mi)*sizeof(int) , cudaMemcpyDeviceToHost));
	FILE *fd = fopen("access_stream.sorted","w+");
	fprintf(fd,"jtion0 jtion1\n");
	for(m=0;m<mi;m++)
	  fprintf(fd,"%d %d\n", h_aux_zion->jtion0[m],h_aux_zion->jtion1[m]);
	fclose(fd);
	irk2_save = 1;
      }
    } else {
      if(!irk1_save) {
	CUDA_SAFE_CALL(cudaMemcpy(h_aux_zion->jtion0,(void *)d_aux_zion->jtion0, 4*(mi)*sizeof(int) , cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(h_aux_zion->jtion1, (void *)d_aux_zion->jtion1, 4*(mi)*sizeof(int) , cudaMemcpyDeviceToHost));
	FILE *fd = fopen("access_stream.not_sorted","w+");
	fprintf(fd,"jtion0 jtion1\n");
	for(m=0;m<mi;m++)
	  fprintf(fd,"%d %d\n", h_aux_zion->jtion0[m],h_aux_zion->jtion1[m]);
	fclose(fd);
	irk1_save = 1;
      }
    }
  }
#endif
  
}


