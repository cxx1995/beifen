#include <assert.h>
//#include <thrust/reduce.h>
//#include <thrust/device_vector.h>

#define MAX_OFFSET 0x10000000

#define _GPU_DEBUG 0

// Macro definition
#define CudaAssert( X ) if ( !(X) ) { printf( "Thread %d:%d failed assert at %s:%d!", blockIdx.x, threadIdx.x); }

__device__ int16_t atomicAddShort(int16_t* address, int16_t val)
{
    unsigned int *base_address = (unsigned int *) ((char *)address - ((size_t)address & 2)); 
    unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;

    unsigned int long_old = atomicAdd(base_address, long_val);
    if((size_t)address & 2) {
        return (int16_t)(long_old >> 16);
    } else {
        unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;
        if (overflow)
            atomicSub(base_address, overflow);
        return (int16_t)(long_old & 0xffff);
    }
}

__global__ static void
gpu_charge_initialization_cooperative(gtc_field_data_t* grid, gtc_particle_data_t* zion, gtc_aux_particle_point_t* point, int nloc_over_cluster, int mcellperthread, int mype, int d_mimax, int d_extra_mimax){

    const int tid = threadIdx.x;
    const int nblocks = gridDim.x;
    const int nthreads = blockDim.x;
    const int gid = tid + blockIdx.x*nthreads;
    
    const real* __restrict__ zion0; const real* __restrict__ zion1; const real* __restrict__ zion2;
    const real* __restrict__ zion4; const real* __restrict__ zion5;
    
    my_int* __restrict__ point_index_count;
    //int* __restrict__ point_index;
    my_real* __restrict__ point_vect;
    // my_real* point_weight;

    extern __shared__ int shared_buffer[];
    int *shared_index = shared_buffer;
    my_real *shared_vect = (my_real *)&shared_index[4*nthreads];
    
#if GYRO_LOCAL_COMPUTE
    real tdum_lc, rad_lc, q_lc, rho_lc, dtheta_dx;
    real radcos_lc, rad_frac_lc;
    real pgyro_lc[4], tgyro_lc[4];
    real deltar, pi, gyrorad_const_mult;
    real a, q0, q1, q2;
#else
    const real*  pgyro; const real*  tgyro;
    int ipjt, idx1;
#endif
    
#if USE_CONSTANTMEM
#else
    const int*  igrid; const real*  delt;
    const real*  qtinv; const int*  mtheta;
#endif

    int mi;
    int mimax; real smu_inv; real a0;
    real a1; real delr; int mpsi;
    real pi2_inv;

    /* temporary variables */
    real psitmp, thetatmp, rhoi, rhotmp, rho_max, r;
    int iptmp, ip, jttmp, jt, ii, im, larmor, index, index0;
    my_real rdum, tflr, zetatmp, weight;
    int j00, jtion0tmp, mpsi_max, cellindex;
    real tdumtmp, tdum;
    real r_diff, a_diff; 
    int igrid_in, ipsi_in, ipsi_out, ipsi_valid_in, ipsi_valid_out, nloc_over;

    mpsi = params.mpsi;
    a0 = params.a0; a1=params.a1;
    delr = params.delr;
    smu_inv = params.smu_inv;
    pi2_inv = params.pi2_inv;
    mi = params.mi;
    //mimax = params.mimax;
    mimax = d_mimax;
    igrid_in = radial_decomp.igrid_in;
    nloc_over = radial_decomp.nloc_over;
    ipsi_in = radial_decomp.ipsi_in;
    ipsi_out = radial_decomp.ipsi_out;
    ipsi_valid_in = radial_decomp.ipsi_valid_in;
    ipsi_valid_out = radial_decomp.ipsi_valid_out;
    rho_max = radial_decomp.rho_max;

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

    point_index_count = point->point_index_count;
    //point_index = point->point_index;
    point_vect = point->point_vect;
   // point_weight = point->point_weight;

    int m = gid;
    if (m<mi){
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
        iptmp    = (int)((r-a0)*delr+0.5);
        ip       = d_abs_min_int(mpsi, iptmp);

#if GPU_DEBUG
	CudaAssert(ip>=ipsi_valid_in);
	CudaAssert(ip<=ipsi_valid_out);
#endif

        jttmp    = (int) (thetatmp*pi2_inv*delt[ip]+0.5);
        jt       = d_abs_min_int(mtheta[ip], jttmp);

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

        for (larmor = 0; larmor < 4; larmor+=1) {
#if GYRO_LOCAL_COMPUTE
	    rhotmp = rhoi*pgyro_lc[larmor];
	    if (fabs(rhotmp)>rho_max) {
	      printf("rhotmp=%e rhoi=%e rho_max=%e pgyro=%e\n", rhotmp, rhoi, rho_max, pgyro_lc[larmor]);
	      printf("warning: charge sub reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro_lc[larmor], rhoi);
	      rhotmp = (rhotmp/fabs(rhotmp))*rho_max;
	      rhoi = rhotmp/pgyro_lc[larmor];
	    }
	    rdum = delr * d_abs_min_real(a_diff,
					 r_diff+rhotmp);
	    tflr    = thetatmp + rhoi*tgyro_lc[larmor];
            tdum = delr*(r_diff+rhotmp);
#else
            idx1    = larmor + 4*(ipjt-igrid_in);
	    rhotmp = rhoi*pgyro[idx1];
	    if (fabs(rhotmp)>rho_max) {
	      printf("rhotmp=%e rhoi=%e rho_max=%e pgyro=%e\n", rhotmp, rhoi, rho_max, pgyro[idx1]);
	      printf("warning: charge sub reducing rhoi to %e from %e\n", (rhotmp/fabs(rhotmp))*rho_max/pgyro[idx1], rhotmp);
	      rhotmp = (rhotmp/fabs(rhotmp))*rho_max;
	      rhoi = rhotmp/pgyro[idx1];
	    }
	    rdum = delr * d_abs_min_real(a_diff,
					 r_diff+rhotmp);
	    tflr    = thetatmp + rhoi*tgyro[idx1];
            tdum = delr*(r_diff+rhotmp);
#endif
            ii      = d_abs_min_int(mpsi_max, (int) rdum);

#if GPU_DEBUG
	    CudaAssert(ii>=ipsi_in);
	    CudaAssert(ii<=ipsi_out-1);
#endif

            //if (ii>=0&&ii<=mpsi_max){
            //    ii      = d_abs_min_int(mpsi_max, (int) rdum);
	    im = ii;
	    tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
	    tdum = (tdumtmp - (int) tdumtmp) * delt[im];
	    j00 = d_abs_min_int(mtheta[im]-1, (int) tdum);
	    jtion0tmp = igrid[im] + j00;

	    cellindex = (jtion0tmp-igrid_in)/mcellperthread;

#if GPU_DEBUG
	    CudaAssert((jtion0tmp-igrid_in)>=0);
	    CudaAssert((jtion0tmp-igrid_in)<(nloc_over-1));

	    CudaAssert(cellindex>=0);
	    CudaAssert(cellindex<nloc_over_cluster);
#endif

#if SHORT_INT
	    index = atomicAddShort(point_index_count+cellindex, 1);
#else
	    index = atomicAdd(point_index_count+cellindex, 1);
#endif
	    index0 = 4*(index*nloc_over_cluster + cellindex);
	    
#if GPU_DEBUG
	    if ((index0+3)>=16*d_extra_mimax){     
	      printf("increase EXTRA_BUFFER\n");
	      printf("index0=%d index=%d, nloc_over_cluster=%d, d_extra_mimax=%d\n", index0,index, nloc_over_cluster, d_extra_mimax);
	      CudaAssert((index0+3)<16*d_extra_mimax);
	    }
#endif
            // } else {
	    //   index0 = -1;
            // }
	    
	    //point_index[larmor*mi+m] = index0;
	    
	    // store data to global memory using cooperative threading algorithm (Kemesh SC11)
	    if (blockIdx.x==(nblocks-1)){
	      //	      if (index0!=-1){
	      point_vect[index0] = rdum;
	      point_vect[index0+1] = tflr;
	      point_vect[index0+2] = zetatmp;
	      point_vect[index0+3] = weight;
	      // }
             } else {
	      shared_index[4*tid] = index0;
	      shared_index[4*tid+1] = index0;
	      shared_index[4*tid+2] = index0;
	      shared_index[4*tid+3] = index0;

	      shared_vect[4*tid] = rdum;
	      shared_vect[4*tid+1] = tflr;
	      shared_vect[4*tid+2] = zetatmp;
	      shared_vect[4*tid+3] = weight;
	      __syncthreads();
	      
	      int offset=tid%4;
	      //if (shared_index[tid]!=-1)
	      point_vect[shared_index[tid]+offset] = shared_vect[tid];
	      //if (shared_index[nthreads+tid]!=-1)
	      point_vect[shared_index[nthreads+tid]+offset] = shared_vect[nthreads+tid];
	      //if (shared_index[2*nthreads+tid]!=-1)
	      point_vect[shared_index[2*nthreads+tid]+offset] = shared_vect[2*nthreads+tid];
	      //if (shared_index[3*nthreads+tid]!=-1)
	      point_vect[shared_index[3*nthreads+tid]+offset] = shared_vect[3*nthreads+tid];
	    }
	      
        } // end larmor
    } // end m

    __syncthreads();
    
}

__global__ static void gpu_charge_interpolation(gtc_field_data_t* grid, gtc_aux_particle_point_t* point, int mype, int nloc_over_cluster, int mcellperthread, int lower, int upper){

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int gid=tid+blockIdx.x*nthreads;

    const my_int* __restrict__ point_index_count;
    const my_real* __restrict__ point_vect;
//    const my_real* __restrict__ point_weight;

    real* __restrict__ densityi;

#if USE_CONSTANTMEM
#else
    const int*  igrid; const real*  delt;
    const real*  qtinv; const int*  mtheta;
#endif
    real delz; int mpsi;
    real pi2_inv; real zetamin; int mzeta;

    /* temporary variables */
    my_real rdum, tflr, zetatmp, weight;  
    my_int maxcount;
    int kk, ii, im, im2;
    real wzt, wz1, wz0,  wp1, wp0;
    int j01, j00, jtion0tmp, jtion1tmp, ij1, ij2, mpsi_max;
    real tdumtmp, tdum, wt10, wt00, wt01, wt11, wtion0tmp, wtion1tmp;
    int igrid_in, ipsi_in, ipsi_out, nloc_over;

    mpsi = params.mpsi;
    delz = params.delz;
 
    zetamin = params.zetamin; mzeta = params.mzeta;
    pi2_inv = params.pi2_inv;
    igrid_in = radial_decomp.igrid_in;
    ipsi_in = radial_decomp.ipsi_in;
    ipsi_out = radial_decomp.ipsi_out;
    nloc_over = radial_decomp.nloc_over;

    point_index_count = point->point_index_count;
    point_vect= point->point_vect;
//    point_weight = point->point_weight;

    densityi = grid->densityi;

#if USE_CONSTANTMEM
#else
    igrid = grid->igrid; qtinv = grid->qtinv;
    mtheta = grid->mtheta; delt = grid->delt;
#endif

    mpsi_max = mpsi-1;

    maxcount = 0;
    if (gid<nloc_over_cluster)
       maxcount = point_index_count[gid];

    extern __shared__ int shared_buffer[];
    // index offset for lower and upper ring
    int *offset0 = shared_buffer;
    int *offset1 = &shared_buffer[nthreads];
    // local density array for lower and upper ring
    real * update_val0 = (real *)&offset1[nthreads];
    real * update_val0_cpy = &update_val0[lower];
    real * update_val1 = &update_val0_cpy[lower];
    //real * update_val1_cpy = &update_val1[upper];
  
    // find the starting index for array in shared memory
    offset0[tid] = MAX_OFFSET;
    offset1[tid] = MAX_OFFSET; 
    
    for (int ij=tid; ij<lower; ij+=nthreads){
       update_val0[ij] = 0.0;
       update_val0_cpy[ij] = 0.0;
    }
    for (int ij=tid; ij<upper; ij+=nthreads){
       update_val1[ij] = 0.0;
       //update_val1_cpy[ij] = 0.0;
    }
    
    if (gid<nloc_over_cluster){
        if (maxcount>0){
	   rdum = point_vect[4*gid];		
	   tflr = point_vect[4*gid+1];
	   zetatmp = point_vect[4*gid+2];

           ii = d_abs_min_int(mpsi_max, (int) rdum);
#if GPU_DEBUG
	   if (ii<ipsi_in || ii>ipsi_out-1){
	      printf("rdum=%f tflr=%f zetatmp=%f ii=%d ipsi_in=%d ipsi_out=%d\n", rdum, tflr, zetatmp, ii, ipsi_in, ipsi_out);
	      CudaAssert(ii>=ipsi_in);
	      CudaAssert(ii<=ipsi_out-1);
	   }
#endif
           im = ii;
           tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
           tdum = (tdumtmp - (int) tdumtmp) * delt[im];
           j00 = d_abs_min_int(mtheta[im]-1, (int) tdum);
           jtion0tmp = igrid[im] + j00;
         
#if GPU_DEBUG
	   if ((jtion0tmp-igrid_in)/mcellperthread!=gid)
	     printf("jtion0tmp-igrid_in/mcellperthread=%d gid=%d\n", (jtion0tmp-igrid_in)/mcellperthread, gid);
           CudaAssert((jtion0tmp-igrid_in)/mcellperthread==gid);
#endif        
           im = ii + 1;
           tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
           tdum = (tdumtmp - (int) tdumtmp) * delt[im];
           j01 = d_abs_min_int(mtheta[im]-1, (int) tdum);
           jtion1tmp = igrid[im] + j01;

           offset0[tid] = (mzeta+1)*gid*mcellperthread;

           //if ((jtion0tmp<=(igrid[ii+1]-2)&&jtion0tmp>=(igrid[ii+1]-6))){
           if (gid==(igrid[ii+1]-2-igrid_in)/mcellperthread||gid==(igrid[ii+1]-3-igrid_in)/mcellperthread){
              offset1[tid] = MAX_OFFSET;
           }
           else{
              offset1[tid] = (mzeta+1)*(jtion1tmp-igrid_in)-16;
              offset1[tid] -= (mzeta+1)*(mcellperthread-1);
#if GPU_DEBUG
	      CudaAssert(offset1[tid]>=0);
#endif
           } 
       }
    }
    __syncthreads();
    
    int nTotalThreads = nthreads;
    while (nTotalThreads>1){
      int half = (nTotalThreads >> 1);
      if (tid < half){
	int temp0 = offset0[tid+half];
	if (temp0<offset0[tid]) offset0[tid]=temp0;
	
	int temp1 = offset1[tid+half];
	if (temp1<offset1[tid]) offset1[tid]=temp1;
      }
      __syncthreads();
      nTotalThreads = (nTotalThreads >> 1);
    }
    
    offset0[tid] = offset0[0];
    offset1[tid] = offset1[0];

    __syncthreads();
    
    // charge deposition in shared memory
    
    for (int m=gid, iter=0; iter<maxcount; iter++, m+=nloc_over_cluster){
      rdum = point_vect[4*m];
      tflr = point_vect[4*m+1];
      zetatmp = point_vect[4*m+2];
      weight = point_vect[4*m+3];
      
      wzt      = (zetatmp-zetamin)*delz;
      kk       = d_abs_min_int(mzeta-1, (int) wzt);
      wz1      = weight * (wzt - (real) kk);
      wz0      = weight - wz1;

      ii      = d_abs_min_int(mpsi_max, (int) rdum);
#if GPU_DEBUG
      if (ii<ipsi_in || ii>(ipsi_out-1)){
	printf("rdum=%f tflr=%f zetatmp=%f weight=%f ii=%d ipsi_in=%d ipsi_out=%d\n",
	       rdum, tflr, zetatmp, weight, ii, ipsi_in, ipsi_out);
        CudaAssert(ii>=ipsi_in);
        CudaAssert(ii<=(ipsi_out-1));
      }
#endif
      wp1     = rdum - (real) ii;
      wp0     = 1.0  - wp1;
          
      im = ii;
      tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im]) + 10.0;
      tdum = (tdumtmp - (int) tdumtmp) * delt[im];
      j00 = d_abs_min_int(mtheta[im]-1, (int) tdum);
      jtion0tmp = igrid[im] + j00;
      wtion0tmp = tdum - (real) j00;
      
      im2 = ii + 1;
      tdumtmp = pi2_inv * (tflr - zetatmp * qtinv[im2]) + 10.0;
      tdum = (tdumtmp - (int) tdumtmp) * delt[im2];
      j01 = d_abs_min_int(mtheta[im2]-1, (int) tdum);
      jtion1tmp = igrid[im2] + j01;
      wtion1tmp = tdum - (real) j01;     
      
#if GPU_DEBUG
      if ((jtion0tmp-igrid_in)/mcellperthread!=gid)
	printf("jtion0tmp-igrid_in/mcellperthread=%d gid=%d\n", (jtion0tmp-igrid_in)/mcellperthread, gid);
      CudaAssert((jtion0tmp-igrid_in)/mcellperthread==gid);
#endif

      wt10 = wp0 * wtion0tmp;
      wt00 = wp0 - wt10;
      
      wt11 = wp1 * wtion1tmp;
      wt01 = wp1 - wt11;
      
      ij1 = kk + (mzeta+1)*(jtion0tmp-igrid_in);
      ij2 = kk + (mzeta+1)*(jtion1tmp-igrid_in);	
      
#if GPU_DEBUG
      if (ij1<0||(ij1+mzeta+2)>=(mzeta+1)*nloc_over){
        printf("ij1=%d jtion0tmp%d\n", ij1, jtion0tmp);
        CudaAssert(ij1>=0);
        CudaAssert((ij1+mzeta+2)<(mzeta+1)*nloc_over);
      }

      if (ij2<0||(ij2+mzeta+2)>=(mzeta+1)*nloc_over){
        printf("ij2=%d jtion1tmp%d\n", ij2, jtion1tmp);
        CudaAssert(ij2>=0);
        CudaAssert((ij2+mzeta+2)<(mzeta+1)*nloc_over);
      }
#endif

      ij1 = ij1 - offset0[tid];
      ij2 = ij2 - offset1[tid];
      
#if DO_CHARGE_UPDATES
  
      // copy to shared memory (no conflict)
      update_val0[ij1] += wz0*wt00;
      update_val0[ij1+1] += wz1*wt00;
      update_val0_cpy[ij1+mzeta+1] += wz0*wt10;
      update_val0_cpy[ij1+mzeta+2] += wz1*wt10;
      
      /*  
      // test 1-for lower ring: atomicAdd to shared memory
      atomicDPupdate(update_val0+ij1,   wz0*wt00);
      atomicDPupdate(update_val0+ij1+1, wz1*wt00);
      atomicDPupdate(update_val0+ij1+mzeta+1, wz0*wt10);
      atomicDPupdate(update_val0+ij1+mzeta+2, wz1*wt10);
      */
      
      /*    
      // test 2-for lower ring: atomicAdd to global memory
      ij1 = ij1 + offset0[tid];
#if GPU_DEBUG
      if (ij1<0||(ij1+mzeta+2)>=(mzeta+1)*nloc_over){
        printf("ij1=%d offset0=%d offset1=%d\n", ij1, offset0[tid], offset1[tid]);
	CudaAssert(ij1>=0);
	CudaAssert((ij1+mzeta+2)<(mzeta+1)*nloc_over);
      }
#endif
      atomicDPupdate(densityi+ij1,   wz0*wt00);
      atomicDPupdate(densityi+ij1+1, wz1*wt00);
      atomicDPupdate(densityi+ij1+mzeta+1, wz0*wt10);
      atomicDPupdate(densityi+ij1+mzeta+2, wz1*wt10);
      */

      /*
      // test 2-for upper ring: atomicAdd to global memory
      ij2 = ij2 + offset1[tid];
#if GPU_DEBUG
      if (ij2<0||(ij2+mzeta+2)>=(mzeta+1)*nloc_over){
	printf("ij2=%d offset0=%d offset1=%d\n", ij2, offset0[tid], offset1[tid]);
	CudaAssert(ij2>=0);
	CudaAssert((ij2+mzeta+2)<(mzeta+1)*nloc_over);
      }
#endif
      atomicDPupdate(densityi+ij2,   wz0*wt01);
      atomicDPupdate(densityi+ij2+1, wz1*wt01);
      atomicDPupdate(densityi+ij2+mzeta+1, wz0*wt11);
      atomicDPupdate(densityi+ij2+mzeta+2, wz1*wt11);
      */

      
      if (ij2<0||(ij2+mzeta+2)>=upper){
	ij2 = ij2 + offset1[tid];
	atomicDPupdate(densityi+ij2,   wz0*wt01);
	atomicDPupdate(densityi+ij2+1, wz1*wt01);
	atomicDPupdate(densityi+ij2+mzeta+1, wz0*wt11);
	atomicDPupdate(densityi+ij2+mzeta+2, wz1*wt11);
      }
      else{
	/*
	if (mcellperthread!=1){
	  // conflict deposition for the upper ring will result in slightly error 
	  if (gid%2==0){
	    update_val1[ij2] += wz0*wt01;
	    update_val1[ij2+1] += wz1*wt01;
	    update_val1[ij2+mzeta+1] += wz0*wt11;
	    update_val1[ij2+mzeta+2] += wz1*wt11;
	  } else {
	    update_val1_cpy[ij2] += wz0*wt01;
	    update_val1_cpy[ij2+1] += wz1*wt01;
	    update_val1_cpy[ij2+mzeta+1] += wz0*wt11;
	    update_val1_cpy[ij2+mzeta+2] += wz1*wt11;
	  }
	} else {
        */
	  atomicDPupdate(update_val1+ij2,   wz0*wt01);
	  atomicDPupdate(update_val1+ij2+1, wz1*wt01);
	  atomicDPupdate(update_val1+ij2+mzeta+1, wz0*wt11);
	  atomicDPupdate(update_val1+ij2+mzeta+2, wz1*wt11);
	  //}
      }
      
#endif
    }
    __syncthreads();

    
    // add shared memory data to global memory 
    for (int ij=tid; ij<lower; ij+=nthreads){
      update_val0[ij] += update_val0_cpy[ij];
      
      if (offset0[tid]!=MAX_OFFSET&&(offset0[tid]+ij)>=0&&(offset0[tid]+ij)<(mzeta+1)*nloc_over)
	atomicDPupdate(densityi+offset0[tid]+ij, update_val0[ij]);
    }

    for (int ij=tid; ij<upper; ij+=nthreads){
      //update_val1[ij] += update_val1_cpy[ij];
      if (offset1[tid]!=0){
	if (offset1[tid]!=MAX_OFFSET&&(offset1[tid]+ij)>=0&&(offset1[tid]+ij)<(mzeta+1)*nloc_over)
	  atomicDPupdate(densityi+offset1[tid]+ij, update_val1[ij]);
      }
    }
}

__global__ static void memreset_real(real *array, int size) {
    const int tidx = threadIdx.x;
    const int tid = threadIdx.y *blockDim.x + tidx;
    const int bid = blockIdx.x +gridDim.x * blockIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
    const int nblocks = gridDim.x * gridDim.y;
    int step = nthreads * nblocks;
    int i;
    for(i=bid*nthreads;i<size;i+=step)
      array[i+tid] = 0.0;
}

__global__ static void memreset_myint(my_int *array, int size) {
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
void call_gpu_charge_4p_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int irk, int idiag)
{
  gtc_global_params_t   *h_params   = &(gtc_input->global_params);
  gtc_field_data_t *d_grid = &(gpu_kernel_input->d_grid);
  gtc_aux_particle_point_t *d_aux_point = &(gpu_kernel_input->d_aux_point);
  gtc_field_data_t *h_grid = &(gtc_input->field_data);
  gtc_radial_decomp_t *h_radial_decomp = &(gtc_input->radial_decomp);

  int mzeta=h_params->mzeta;
  int mi = h_params->mi; 
  int nloc_over = h_radial_decomp->nloc_over;
  int nloc_over_cluster = gpu_kernel_input->d_nloc_over_cluster;
  int d_extra_mimax = gpu_kernel_input->d_extra_mimax;
  int mp = gpu_kernel_input->deviceProp.multiProcessorCount;
  
  /*************** reset charge density and point number counter*****************/
  gpu_timer_start(gpu_kernel_input);
  //CUDA_SAFE_CALL(cudaMemset(d_grid->densityi, 0, (mzeta+1)*nloc_over*sizeof(wreal)));
  memreset_real<<< mp, 512>>>(d_grid->densityi, (mzeta+1)*nloc_over);
  gpu_kernel_input->gpu_timing.memtransfer_charge_time += gpu_timer_measure(gpu_kernel_input);
  
  //CUDA_SAFE_CALL(cudaMemset(d_aux_point->point_index_count, 0, nloc_over_cluster*sizeof(my_int)));
  memreset_myint<<<mp, 512>>>(d_aux_point->point_index_count, nloc_over_cluster);
  gpu_kernel_input->gpu_timing.memreset_charge_time += gpu_timer_measure(gpu_kernel_input);
  
  /************** initialize four point data **********************/
  int mcellperthread=CLUSTER_SIZE; // for size A, mcellperthread>=4
  int nt=64;
  int nb = (mi + nt -1)/nt;

  //printf("start initialization mype=%d\n", gtc_input->parallel_decomp.mype);
  int shared_buffer= nt*4*sizeof(int)+nt*4*sizeof(my_real);
  gpu_charge_initialization_cooperative<<<nb, nt, shared_buffer>>>(gpu_kernel_input->ptr_d_grid, gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_aux_point, nloc_over_cluster, mcellperthread, gtc_input->parallel_decomp.mype, gpu_kernel_input->d_mimax, gpu_kernel_input->d_extra_mimax);

  /************* overlap GPU computation with CPU I/O work *************/
  if (idiag==0) diagnosis(gtc_input);  

  cudaError_t lasterror = cudaGetLastError();
  if(lasterror != cudaSuccess)
    printf("Error in launching gpu_charge_initialization routine: %s\n", cudaGetErrorString(lasterror));
  gpu_kernel_input->gpu_timing.initialization_charge_time += gpu_timer_measure(gpu_kernel_input);
  //gpu_kernel_input->gpu_timing.device_charge_time += gpu_timer_measure(gpu_kernel_input);
  //printf("finish initialization mype=%d\n", gtc_input->parallel_decomp.mype);

#if _GPU_DEBUG
  if (gtc_input->parallel_decomp.mype==0){
    
    gtc_aux_particle_point_t *h_aux_point = &(gtc_input->particle_point);
    CUDA_SAFE_CALL(cudaMemcpy(h_aux_point->point_index_count, (void *)d_aux_point->point_index_count, nloc_over_cluster*sizeof(my_int), cudaMemcpyDeviceToHost));
    
    //char filename[50];
    //sprintf(filename, "count%d-%d.txt",gpu_kernel_input->istep, gpu_kernel_input->irk);
    //FILE *fd = fopen(filename, "w");
    int total_m = 0;
    int max_m=0;
    for (int i=0; i<nloc_over_cluster; i++){
      if (max_m<h_aux_point->point_index_count[i])
	max_m = h_aux_point->point_index_count[i];
      total_m += h_aux_point->point_index_count[i];
      //fprintf(fd, "%d %d \n", i, h_aux_point->point_index_count[i]);
    }
    //fclose(fd);
    printf("mype=%d total_m=%d 4mi=%d diff=%d max_m=%d\n", gtc_input->parallel_decomp.mype, total_m, 4*mi, total_m-4*mi, max_m);
  }
#endif
  
  /************* interpolation ************************************/
  nt = 32;
  nb = (nloc_over_cluster + nt - 1)/nt;
  int lower = (mzeta+1)*(mcellperthread*nt+1);
  int upper = (int)((mzeta+1)*mcellperthread*nt*1.5);
  shared_buffer = 2*nt*sizeof(int) + 2*lower*sizeof(real)+ upper*sizeof(real);

  //printf("start interpolation mype=%d\n", gtc_input->parallel_decomp.mype);
  gpu_charge_interpolation<<<nb, nt, shared_buffer>>>(gpu_kernel_input->ptr_d_grid,gpu_kernel_input->ptr_d_aux_point, gtc_input->parallel_decomp.mype, nloc_over_cluster, mcellperthread, lower, upper);

  lasterror = cudaGetLastError();
  if(lasterror != cudaSuccess)
    printf("Error in launching gpu_charge_interpolation routine: %s\n", cudaGetErrorString(lasterror));

  gpu_kernel_input->gpu_timing.interpolation_charge_time += gpu_timer_measure(gpu_kernel_input);
  //gpu_kernel_input->gpu_timing.device_charge_time += gpu_timer_measure(gpu_kernel_input);
  //printf("finish interpolation mype=%d\n", gtc_input->parallel_decomp.mype);
  
  CUDA_SAFE_CALL(cudaMemcpy(h_grid->densityi, (void *)(d_grid->densityi), (mzeta+1)*nloc_over*sizeof(wreal), cudaMemcpyDeviceToHost));
  gpu_kernel_input->gpu_timing.memtransfer_charge_time += gpu_timer_measure_end(gpu_kernel_input);
  
}

extern "C"
void gpu_charge_init(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input)
{
  gtc_global_params_t   *h_params   = &(gtc_input->global_params);
  gtc_aux_particle_point_t *d_aux_point = &(gpu_kernel_input->d_aux_point);
  gtc_radial_decomp_t *h_radial_decomp = &(gtc_input->radial_decomp);
  
  int mi = h_params->mi; int nloc_over = h_radial_decomp->nloc_over; 
  int nloc_over_cluster = gpu_kernel_input->d_nloc_over_cluster;
  int mp = gpu_kernel_input->deviceProp.multiProcessorCount;
  
  /*************** reset point number counter*****************/
  gpu_timer_start(gpu_kernel_input);
  //CUDA_SAFE_CALL(cudaMemset(d_aux_point->point_index_count, 0, nloc_over_cluster*sizeof(my_int)));
  memreset_myint<<<mp, 512>>>(d_aux_point->point_index_count, nloc_over_cluster);
  gpu_kernel_input->gpu_timing.memreset_charge_time += gpu_timer_measure(gpu_kernel_input);
  
  /************** initialize four point data **********************/
  int mcellperthread=CLUSTER_SIZE; 
  int nt = 64;
  int nb = (mi + nt -1)/nt;
  
  int shared_buffer = nt*4*sizeof(int)+nt*4*sizeof(my_real);
  gpu_charge_initialization_cooperative<<<nb, nt, shared_buffer>>>(gpu_kernel_input->ptr_d_grid, gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_aux_point, nloc_over_cluster, mcellperthread, gtc_input->parallel_decomp.mype, gpu_kernel_input->d_mimax, gpu_kernel_input->d_extra_mimax);
  //gpu_kernel_input->gpu_timing.initialization_charge_time += gpu_timer_measure(gpu_kernel_input);
  gpu_kernel_input->gpu_timing.device_charge_time += gpu_timer_measure_end(gpu_kernel_input);  

#if _GPU_DEBUG
  if (gtc_input->parallel_decomp.mype==0){
    
    gtc_aux_particle_point_t *h_aux_point = &(gtc_input->particle_point);
    CUDA_SAFE_CALL(cudaMemcpy(h_aux_point->point_index_count, (void *)d_aux_point->point_index_count, nloc_over_cluster*sizeof(my_int), cudaMemcpyDeviceToHost));
    
    //char filename[50];
    //sprintf(filename, "count%d-%d.txt",gpu_kernel_input->istep, gpu_kernel_input->irk);
    //FILE *fd = fopen(filename, "w");
    int total_m = 0;
    int max_m = 0;
    for (int i=0; i<nloc_over_cluster; i++){
      if (h_aux_point->point_index_count[i]>max_m)
	max_m = h_aux_point->point_index_count[i];
      total_m += h_aux_point->point_index_count[i];
      //fprintf(fd, "%d %d \n", i, h_aux_point->point_index_count[i]);
    }
    //fclose(fd);
    
    printf("total_m=%d 4mi=%d diff=%d\n", total_m, 4*mi, total_m-4*mi);
  }
#endif

}


