#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mpi.h"
#include "bench_gtc.h"
#include <math.h>
#if _OPENMP
#include <omp.h>
#endif

//#define _DEBUG_GPU 1
#define PROFILE_CYCLE 0

#if PROFILE_CYCLE 

inline unsigned long read_cycle()
{
#if defined(__ia64)
    /* for Intel Itanium 64 bit processor */
    unsigned long x;
    __asm__ __volatile__("mov &#37;0=ar.itc" : "=r"(x) :: "memory");
    while (__builtin_expect ((int) x == -1, 0))
        __asm__ __volatile__("mov %0=ar.itc" : "=r"(x) :: "memory");
    return x;
#else
    /* for all others */
    unsigned long long x;
    unsigned int low,high;
    __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high));
    x = high;
    x = (x<<32)+low;
    return x;
#endif
}

#endif


int gpu_chargei(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input) 
{
    gtc_global_params_t     *params;
    gtc_field_data_t        *field_data;
    gtc_particle_decomp_t   *parallel_decomp;
    gtc_radial_decomp_t     *radial_decomp;
    
    int mpsi, mzeta; 
    int ntoroidal;
    int nbound;
    real deltar;
    real a1, a0;

    const int*  __restrict__ itran;

    const int*  __restrict__ igrid;
    const int*  __restrict__ mtheta;

    wreal *densityi;

    wreal *recvr; real *zonali; real *markeri; real *pmarki;
    int *recvr_index;
    real *adum, *adum2;

    real a_diff;

#if USE_MPI
    wreal *sendl, *dnitmp;
    int icount, idest, isource, isendtag, irecvtag; 
    MPI_Status istatus;
#endif
    int nloc_over, ipsi_in, ipsi_out, ipsi_nover_in, ipsi_nover_out, igrid_in,
      ipsi_valid_in, ipsi_valid_out;
    /*******/

    params            = &(gtc_input->global_params);
    field_data        = &(gtc_input->field_data);
    parallel_decomp   = &(gtc_input->parallel_decomp);
    radial_decomp     = &(gtc_input->radial_decomp);

    //mgrid = params->mgrid;
    mzeta = params->mzeta; mpsi = params->mpsi;
    nbound = params->nbound;
    a1 = params->a1; a0 = params->a0;
       
    densityi = field_data->densityi;
    igrid = field_data->igrid; 
    mtheta = field_data->mtheta;
    itran = field_data->itran; 
    recvr = field_data->recvr; 
    recvr_index = field_data->recvr_index;

#if USE_MPI
    sendl = field_data->sendl;
    dnitmp = field_data->dnitmp;
#endif

    zonali = field_data->zonali; markeri = field_data->markeri;
    pmarki = field_data->pmarki; 
    adum = field_data->adum; adum2 = field_data->adum2;

    ntoroidal = parallel_decomp->ntoroidal;

    nloc_over = radial_decomp->nloc_over;
    ipsi_in   = radial_decomp->ipsi_in;
    ipsi_out  = radial_decomp->ipsi_out;
    ipsi_valid_in = radial_decomp->ipsi_valid_in;
    ipsi_valid_out = radial_decomp->ipsi_valid_out;
    ipsi_nover_in = radial_decomp->ipsi_nover_in;
    ipsi_nover_out = radial_decomp->ipsi_nover_out;
    igrid_in  = radial_decomp->igrid_in;
    /********/

    a_diff   = a1-a0;
    deltar   = a_diff/mpsi;
    
#if !FOURPOINT
    call_gpu_charge_kernel(gtc_input,gpu_kernel_input, idiag);
    
#if GPU_DEBUG
    if (parallel_decomp->mype==0){
      char filename[50];
      sprintf(filename, "charge_old%d-%d.txt",gpu_kernel_input->istep, gpu_kernel_input->irk);
      FILE *fp;
      fp = fopen(filename, "w");
      for (int j=0; j<nloc_over; j++){
	if (j%100==0){
	  fprintf(fp, "j=%d, den=%e\n", j, densityi[(mzeta+1)*j]);
	}
      }
      fclose(fp);
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Abort(MPI_COMM_WORLD, 1); 
#endif
    
#else
    call_gpu_charge_4p_kernel(gtc_input,gpu_kernel_input,istep, idiag);
    
#if GPU_DEBUG			
    if (parallel_decomp->mype==0){
      char filename[50];
      sprintf(filename, "charge_new%d-%d.txt",gpu_kernel_input->istep, gpu_kernel_input->irk);
      FILE *fp;
      fp = fopen(filename, "w");
      for (int j=0; j<nloc_over; j++){
	if (j%100==0){
	  fprintf(fp, "j=%d, den=%e\n", j, densityi[(mzeta+1)*j]);
	}
      }
      fclose(fp);
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    //MPI_Abort(MPI_COMM_WORLD, 1);
#endif

#endif
    /*
#if GPU_DEBUG
    if (parallel_decomp->mype==0) {
      char filename[50];
      sprintf(filename, "den_gpu_out0.txt");
      FILE *fp;
      fp = fopen(filename, "w");
      for (int j=0; j<nloc_over; j++){
        fprintf(fp, "j=%d, den=%e\n", j+igrid_in, densityi[(mzeta+1)*j+1]);
      }
      fclose(fp);
    }
      
    if (parallel_decomp->mype==1) {
      char filename[50];
      sprintf(filename, "den_gpu_out1.txt");
      FILE *fp;
      fp = fopen(filename, "w");
      for (int j=0; j<nloc_over; j++){
        fprintf(fp, "j=%d, den=%e\n", j+igrid_in, densityi[(mzeta+1)*j+1]);
      }
      fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    */

#if USE_MPI
    if (parallel_decomp->npartdom > 1) {
#if FINE_TIMER
      real start_t = MPI_Wtime();
#endif
      /* sum radial particles */
      if (radial_decomp->nproc_radial_partd > 1){
#pragma omp parallel for
        for (int ij=0; ij<nloc_over; ij++) {
          for (int kk=0; kk<mzeta+1; kk++) {
            dnitmp[ij*(mzeta+1)+kk] = densityi[ij*(mzeta+1)+kk];
            densityi[ij*(mzeta+1)+kk] = 0.0;
          }
        }
        MPI_Allreduce(dnitmp, densityi, nloc_over*(mzeta+1),
                      MPI_MYWREAL, MPI_SUM,
                      parallel_decomp->radial_partd_comm);
      }
      /* sum ghost cell*/
      sum_plane(gtc_input);
#if FINE_TIMER
      real end_t = MPI_Wtime();
      charge_t_comm += (end_t - start_t);
#endif
    }
#endif

    /* Poloidal end cell, discard ghost cell j = 0 */
#pragma omp parallel for if(ipsi_out-ipsi_in+1>=parallel_decomp->nthreads)
    for (int i=ipsi_in; i<ipsi_out+1; i++) {
      int col_start = igrid[i] + mtheta[i] - igrid_in;
      for (int j=0; j<mzeta+1; j++) {
	densityi[col_start*(mzeta+1) + j]
	  += densityi[(igrid[i]-igrid_in)*(mzeta+1)+j];
      }
    }

#if USE_MPI
#pragma omp parallel for
    for (int j=0; j<nloc_over; j++) {
        sendl[j] = densityi[(mzeta+1)*j];
        recvr[j] = 0.0;
    }

    icount = nloc_over;
    idest = parallel_decomp->left_pe;
    isource = parallel_decomp->right_pe;
    isendtag = parallel_decomp->myrank_toroidal;
    irecvtag = isource;
    
    MPI_Sendrecv(sendl, icount, MPI_MYWREAL, idest, isendtag,
            recvr, icount, MPI_MYWREAL,
            isource, irecvtag, parallel_decomp->toroidal_comm, &istatus);
#else
#pragma omp parallel for
    for (int j=0; j<nloc_over; j++) {
        recvr[j] = densityi[(mzeta+1)*j];
    }
#endif
    
#pragma omp parallel for
    for (int j=0; j<nloc_over; j++){
      densityi[mzeta+(mzeta+1)*j] += recvr[recvr_index[j]];
    }
    
    /* Zero out charge in radial boundary cell */
    for (int i=0; i<nbound; i++) {
      int col_start, col_end;
      if (i>=ipsi_in&&i<=ipsi_out){
	col_start = igrid[i]-igrid_in;
	col_end = igrid[i]+mtheta[i]-igrid_in;
	for (int j=col_start+1; j<col_end+1; j++) {
	  for (int k=0; k<mzeta+1; k++) {
	    densityi[k+j*(mzeta+1)] *= (real) i/(real) nbound;
	  }
	}
      }
      if ((mpsi-i)<=ipsi_out&&(mpsi-i)>=ipsi_in){
	col_start = igrid[mpsi-i]-igrid_in;
	col_end   = igrid[mpsi-i]+mtheta[mpsi-i]-igrid_in;
	for (int j=col_start+1; j<col_end+1; j++) {
	  for (int k=0; k<mzeta+1; k++) {
	    densityi[k+j*(mzeta+1)] *= (real) i/(real) nbound;
	  }
	}
      }
    }

    /* Flux surface average and normalization */
#pragma omp parallel for
    for (int i=0; i<mpsi+1; i++) {
      zonali[i] = 0.0;
    }

    //#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i=ipsi_in; i<ipsi_out+1; i++) {
      for (int j=1; j<mtheta[i]+1; j++) {
	for (int k=1; k<mzeta+1; k++) {
	  int ij = igrid[i] + j - igrid_in;
	  if (i>=ipsi_nover_in&&i<=ipsi_nover_out)
	    zonali[i] += 0.25*densityi[k+ij*(mzeta+1)];
	  densityi[k+ij*(mzeta+1)] =
	    0.25*densityi[k+ij*(mzeta+1)]*markeri[k-1+ij*mzeta];
	}
      }
    }

 #if USE_MPI
    
#if 0
    adum = (real *) _mm_malloc((mpsi+1)*sizeof(real), IDEAL_ALIGNMENT);
    adum2 = (real *) _mm_malloc((mpsi+1)*sizeof(real), IDEAL_ALIGNMENT);
    assert(adum != NULL); assert(adum2 != NULL);
#endif
#if FINE_TIMER
    real start_t = MPI_Wtime();
#endif
    MPI_Allreduce(zonali, adum2, mpsi+1, MPI_MYREAL, MPI_SUM, 
            parallel_decomp->toroidal_comm);
    MPI_Allreduce(adum2, adum, mpsi+1, MPI_MYREAL, MPI_SUM,
		  parallel_decomp->partd_comm);
#if FINE_TIMER
    real end_t = MPI_Wtime();
    charge_t_comm3 += (end_t - start_t);
#endif
#pragma omp parallel for
    for (int i=0; i<mpsi+1; i++) {
        zonali[i] = adum[i]*pmarki[i];
    }
    
#if 0
    _mm_free(adum);
    _mm_free(adum2);
#endif  
  
#else
#pragma omp parallel for
    for (int i=0; i<mpsi+1; i++) {
        zonali[i] = zonali[i]*pmarki[i];
    }
#endif

    /* densityi subtracted (0,0) mode */
    //#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i=ipsi_in;i<ipsi_out+1; i++) {
      for (int j=1; j<mtheta[i]+1; j++) {
	for (int k=1; k<mzeta+1; k++) {
	  int ij = igrid[i] + j - igrid_in;
	  densityi[k+ij*(mzeta+1)] -= zonali[i];
	}
      }
      /* poloidal BC condition */
      int col_start = igrid[i] - igrid_in;
      int col_end   = igrid[i] + mtheta[i] - igrid_in;
      for (int j=1; j<mzeta+1; j++) {
	densityi[j+col_start*(mzeta+1)] = densityi[j+col_end*(mzeta+1)];
      }
    }

    /*  
#if CPU_DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    if (parallel_decomp->mype==0) {
      char filename[50];
      sprintf(filename, "den_gpu_out0.txt");
      FILE *fp;
      fp = fopen(filename, "w");
      for (int j=0; j<nloc_over; j++){
        fprintf(fp, "j=%d, den=%14.13e\n", j+igrid_in, densityi[(mzeta+1)*j+1]);
      }
      fclose(fp);
    }
    if (parallel_decomp->mype==1) {
      char filename[50];
      sprintf(filename, "den_gpu_out1.txt");
      FILE *fp;
      fp = fopen(filename, "w");
      for (int j=0; j<nloc_over; j++){
        fprintf(fp, "j=%d, den=%14.13e\n", j+igrid_in, densityi[(mzeta+1)*j+1]);
      }
      fclose(fp);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);     
#endif 
*/
    /* enforce charge conservation for zonal flow mode */
    real rdum = 0.0;
    real tdum = 0.0;
#pragma omp parallel for reduction(+:rdum, tdum)
    for (int i=1; i<mpsi; i++) {
        real r = a0+deltar*(real) i;
        rdum += r;
        tdum += r*zonali[i];
    }
    tdum = tdum/rdum;
    
#pragma omp parallel for
    for (int i=1; i<mpsi; i++) {
        zonali[i] -= tdum;
    }
    return 0; 
}

int gpu_pushi(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input) 
{
  gtc_global_params_t     *params;
  gtc_field_data_t *field_data;
  gtc_diagnosis_data_t *diagnosis;

  real *pfluxpsi;
  real *eflux, *rmarker, *dmark, *dden;
  real *flux_data;
  
  params = &(gtc_input->global_params);
  field_data = &(gtc_input->field_data);
  diagnosis = &(gtc_input->diagnosis_data);

  pfluxpsi = field_data->pfluxpsi;
  eflux = diagnosis->eflux;
  rmarker = diagnosis->rmarker;
  dmark = diagnosis->dmark;
  dden = diagnosis->dden;
  flux_data = diagnosis->flux_data;
 
  gpu_kernel_input->irk = irk;
  gpu_kernel_input->istep = istep;
  gpu_kernel_input->idiag = idiag;

  //#if !FOURPOINT
  call_gpu_push_kernel(gtc_input,gpu_kernel_input, idiag);
  //#else
  //  call_gpu_push_4p_kernel(gtc_input, gpu_kernel_input, idiag);
  //#endif
  
  if (idiag==0) {
    const int mflux=MFLUX;
    int ndiag = params->ndiag;
    real dmarktmp0[MFLUX];
    real dmarktmp1[MFLUX];
    real ddentmp0[MFLUX];
    real ddentmp1[MFLUX];
    
    real tdum;
    for (int i=0; i<MFLUX; i++)
      {
	eflux[i] = flux_data[i];
        rmarker[i] = flux_data[MFLUX+i];
	dmark[i] = flux_data[2*MFLUX+i];
        dden[i] = flux_data[3*MFLUX+i];
	
        dmarktmp0[i] = flux_data[2*MFLUX+i];
        ddentmp0[i] = flux_data[3*MFLUX+i];
	
	dmarktmp1[i] = 0.0;
	ddentmp1[i] = 0.0;
      }
    MPI_Allreduce(&dmarktmp0, &dmarktmp1, mflux, MPI_MYREAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ddentmp0, &ddentmp1, mflux, MPI_MYREAL, MPI_SUM, MPI_COMM_WORLD);
    for (int i=0; i<MFLUX; i++)
      {
	dmarktmp0[i] = dmarktmp1[i]/gtc_max(1.0, ddentmp1[i]);
      }
    tdum=0.01*(real)ndiag;
    for (int i=0; i<MFLUX; i++)
      {
	pfluxpsi[i] = (1.0-tdum)*pfluxpsi[i]+tdum*dmarktmp0[i];
      }
  }
  
  return 0;
}

int gpu_shifti_toroidal(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input) 
{
#if PROFILE_CYCLE 
  
#define MEASUREMENT_PHASES 200
  unsigned long last_phase=0, start=read_cycle(), end;
  unsigned long cycle_measurements[MEASUREMENT_PHASES];
#endif
  
#if USE_MPI
  gtc_global_params_t     *params;
  gtc_particle_data_t     *particle_data;
  gtc_aux_particle_data_t *aux_particle_data;
  gtc_particle_decomp_t   *parallel_decomp;
  gtc_radial_decomp_t     *radial_decomp;

  int  mimax;
  int ntoroidal;
  real zetamax, zetamin;
  
  int mi_end;
  
  int *kzi;
  
  int *mshift;
  
  int nparam;
  real pi, pi2, pi_inv;
  
  int m0, iteration;
  int max_threads;
    
  /*******/
  
  params            = &(gtc_input->global_params);
  particle_data     = &(gtc_input->particle_data);
  aux_particle_data = &(gtc_input->aux_particle_data);
  parallel_decomp   = &(gtc_input->parallel_decomp);
  radial_decomp     = &(gtc_input->radial_decomp);

  mimax = params->mimax;
  pi = params->pi;
  zetamax = params->zetamax; zetamin = params->zetamin;
  
  kzi = aux_particle_data->kzi;
  
  ntoroidal = parallel_decomp->ntoroidal;
  
  nparam = 12;
  pi_inv = 1.0/pi;
  pi2 = 2.0*pi;
  
  /********/
  
  if (ntoroidal == 1)
    return 0;
  
  m0 = 0;
  mi_end = 0;
  iteration = 0;
  
  max_threads = 32;
  mshift = (int *) _mm_malloc(max_threads * 8 * sizeof(int),
			      IDEAL_ALIGNMENT);
  assert(mshift != NULL);
  unsigned int tops[3];
  real *sends[3];
  sends[0] = sends[1] = sends[2] = NULL;
  int mimax_shift = gpu_kernel_input->d_max_shift_mi;
  real *zion = (real *) malloc(sizeof(real)*nparam*mimax_shift);
  assert(zion != NULL);

  int mi = 0;

  while (iteration <= ntoroidal) {
    real *sendright, *sendleft;
    real *recvleft, *recvright;
    int msend, mrecv, msendright, msendleft, pos;
    int mrecvleft, mrecvright;
    int isendtag, irecvtag;
    MPI_Status istatus1, istatus2;
    
    if(iteration == 0) {
#if PROFILE_CYCLE 
      end = read_cycle();
      cycle_measurements[last_phase++] = end - start;
      start = end;
#endif
      call_gpu_shifti_extract_kernel(gtc_input,gpu_kernel_input, tops, sends, 0);

      msend = tops[1] + tops[2];
      msendright = tops[2];
      msendleft = tops[1];
      sendright = sends[2];
      sendleft = sends[1];

#if _DEBUG_GPU
      int misclassified_sendleft = 0;
      int misclassified_sendright = 0;
      int misclassified_send = 0;
      int i;
      real zetaright, zetaleft;
      real z2val;
      
      for (i=0; i<tops[1]; i++) {
	//z2val = sendleft[i*nparam+2];
	z2val = sendleft[2*tops[1]+i];
	if(abs(z2val)>pi2*3)
	  fprintf(stderr,"strange value %d\n",i);
	zetaright = gtc_min(pi2, z2val) - zetamax;
	zetaleft  = z2val - zetamin;
	if (zetaright * zetaleft > 0.0) {
	  zetaright = zetaright*0.5*pi_inv; 
	  zetaright = zetaright-floor(zetaright);             
	  if (zetaright < 0.5) {
	    misclassified_sendright++;
	  } else {
	    /* Just oK*/
	  }
	} else
	  misclassified_send++;
      }
      for (i=0; i<tops[2]; i++) {
	//z2val = sendright[i*nparam+2];
	z2val = sendright[2*tops[2]+i];
	zetaright = gtc_min(pi2, z2val) - zetamax;
	zetaleft  = z2val - zetamin;
	if (zetaright * zetaleft > 0.0) {
	  zetaright = zetaright*0.5*pi_inv;
	  zetaright = zetaright-floor(zetaright);             
	  if (zetaright < 0.5) {
	    /* Just oK*/
	  } else {
	    misclassified_sendleft++;
	  }
	} else
	  misclassified_send++;
      }
      fprintf(stderr,"Toroidal Send left %d, misclassfied left %d Send right %d, misclassified right %d total misclassified: %d\n",msendleft, misclassified_sendleft, msendright, misclassified_sendright,misclassified_send);
#endif // _DEBUG_GPU
      
      iteration++;

    }else {
#if PROFILE_CYCLE 
      end = read_cycle();
      cycle_measurements[last_phase++] = end - start;
      start = end;
#endif
      
      msend = msendright = msendleft = 0;
      
#pragma omp parallel 
      {    
	int mlstack_max, mrstack_max; 
	int *lstack, *rstack, *stack_tmp;
	int lpos, rpos;
	
	int i;  
	int tid, nthreads;
	real zetaright, zetaleft;
	real z2val;
	
#ifdef _OPENMP
	tid = omp_get_thread_num();
	nthreads = omp_get_num_threads();
#else
	tid = 0;
	nthreads = 1;
#endif
	assert(nthreads == parallel_decomp->nthreads);
	assert(nthreads <= max_threads);

	mshift[8*tid] = mshift[8*tid+1] 
	  = mshift[8*tid+2] = mshift[8*tid+3] 
	  = mshift[8*tid+4] = 0;

	mlstack_max = mrstack_max = 2048;
	lstack = (int *) malloc(mlstack_max * sizeof(int));
	rstack = (int *) malloc(mrstack_max * sizeof(int));
	assert(lstack != NULL); assert(rstack != NULL);

#pragma omp barrier
	
#pragma omp for 
	for (i=m0; i<mi_end; i++) {
	  //z2val = zion[i*nparam+2];
	  z2val = zion[2*mimax_shift+i];
	  if (z2val == HOLEVAL) {
	    continue;
	  }
	  zetaright = gtc_min(pi2, z2val) - zetamax;
	  zetaleft  = z2val - zetamin;
	  if (zetaright * zetaleft > 0.0) {

	    zetaright = zetaright*0.5*pi_inv;
	    zetaright = zetaright-floor(zetaright);             

	    mshift[8*tid]++;

	    /* particle to move right */
	    if (zetaright < 0.5) {

	      rpos = mshift[8*tid+1]++;
	      if (rpos == mrstack_max) {
		/* double the size */
		stack_tmp = (int *) malloc(2*mrstack_max*sizeof(int));
		assert(stack_tmp != NULL);
		memcpy(stack_tmp, rstack, mrstack_max*sizeof(int));
		free(rstack);
		rstack = stack_tmp;
		mrstack_max = 2 * mrstack_max;
	      }
	      rstack[rpos] = i;
	      /* particle to move left */        
	    } else {
	      
	      lpos = mshift[8*tid+2]++;
	      if (lpos == mlstack_max) {
		/* double the size */
		stack_tmp = (int *)
		  malloc(2*mlstack_max*sizeof(int));
		assert(stack_tmp != NULL);
		memcpy(stack_tmp, lstack, mlstack_max*sizeof(int));
		free(lstack);
		lstack = stack_tmp;
		mlstack_max = 2 * mlstack_max;
	      }
	      lstack[lpos] = i;
	    }

	  }
	}
	
        /* Merge partial arrays */
	if (tid == 0) {
	  mshift[8*0+3] = 0;
	  mshift[8*0+4] = 0;
	  for (i=1; i<nthreads; i++) {
	    mshift[8*i+3] = mshift[8*(i-1)+3] + mshift[8*(i-1)+1];
	    mshift[8*i+4] = mshift[8*(i-1)+4] + mshift[8*(i-1)+2];
	  }
	  msendright  = mshift[8*(nthreads-1)+3] + mshift[8*(nthreads-1)+1];
	  msendleft   = mshift[8*(nthreads-1)+4] + mshift[8*(nthreads-1)+2];
	  msend = msendright + msendleft;
	  if (msend > mi) {
	    fprintf(stderr, "Error! mype %d, msend %d, left %d, right"
		    " %d, mi %d\n", 
		    parallel_decomp->mype, msend, msendleft, msendright, mi);
	    exit(1);
	  }
	} 

#pragma omp barrier

	memcpy(kzi+mshift[8*tid+3], rstack, mshift[8*tid+1]*sizeof(int));
	memcpy(kzi+msendright+mshift[8*tid+4], lstack,
	       mshift[8*tid+2]*sizeof(int));
	free(lstack);
	free(rstack);
      } //omp parallel      
      
      iteration++;

      if (iteration > 1) {
	mrecv = 0;
	MPI_Allreduce(&msend, &mrecv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	if (mrecv == 0) {
	  break;
	}
	if (parallel_decomp->mype == 0) {
	  fprintf(stderr, "toroidal iteration %d, mrecv %d\n", iteration,
		  mrecv);
	}
      }

      if (msendleft + msendright > parallel_decomp->sendbuf_size) {
	fprintf(stderr, "Error! PE %d, msendleft %d, msendright %d, "
		"sendbuf_size %d\n", parallel_decomp->mype, 
		msendleft, msendright,
		parallel_decomp->sendbuf_size);
	exit(1);
      } else {
	sendleft = parallel_decomp->sendbuf;
	sendright = parallel_decomp->sendbuf + nparam*msendleft;
      }      
      
#if 0
      /* Allocate space for sendright and sendleft */
      sendright = (real *) malloc(nparam*msendright*sizeof(real));
      assert(sendright != NULL);
      
      sendleft = (real *) malloc(nparam*msendleft*sizeof(real));
      assert(sendleft != NULL);
#endif

      int i,j;
      /* pack particle data */
      /*
#pragma omp parallel for private(pos)
      for (i=0; i<msendright; i++) {
	pos = kzi[i];
	for(j=0;j<nparam;j++)
	  sendright[nparam*i+j] = zion[pos*nparam + j];
	zion[pos*nparam +2] = HOLEVAL;
	zion[pos*nparam +8] = HOLEVAL;
      }
      
#pragma omp parallel for private(pos)
      for (i=0; i<msendleft; i++) {
	pos = kzi[i+msendright];
	for(j=0;j<nparam;j++)
	  sendleft[nparam*i+j] = zion[pos*nparam + j];
	zion[pos*nparam + 2] = HOLEVAL;
	zion[pos*nparam + 8] = HOLEVAL;
      }
      */
      /* pack particle data */
#pragma omp parallel for private (pos,j,i)
      for (j=0; j<nparam; j++){
	for (i=0; i<msendright; i++){
	  pos = kzi[i];
	  sendright[j*msendright+i] = zion[j*mimax_shift+pos];
	  if (j==2){
	    zion[2*mimax_shift+pos] = HOLEVAL;
	  }
	}
      }

#pragma omp parallel for private (pos,j,i)
      for (j=0; j<nparam; j++){
        for (i=0; i<msendleft;i++){
          pos =kzi[i+msendright];
          sendleft[j*msendleft+i] = zion[j*mimax_shift+pos];
          if (j==2){
            zion[2*mimax_shift+pos] = HOLEVAL;
	  }
        }
      }


    } // else second iteration forward
#if PROFILE_CYCLE 
    end = read_cycle();
    cycle_measurements[last_phase++] = end - start;
    start = end;
#endif
    
    /* send # of particles to move right, get # to recv from left */
    isendtag = parallel_decomp->myrank_toroidal;
    irecvtag = parallel_decomp->left_pe;
    mrecvleft = 0;
    MPI_Sendrecv(&msendright, 1, MPI_INT, parallel_decomp->right_pe, 
		 isendtag, &mrecvleft, 1, MPI_INT, parallel_decomp->left_pe, 
		 irecvtag, parallel_decomp->toroidal_comm, &istatus1);
    
    /* send # of particles to move left, get # to recv from right */
    irecvtag = parallel_decomp->right_pe;
    mrecvright = 0;
    MPI_Sendrecv(&msendleft, 1, MPI_INT, parallel_decomp->left_pe, 
		 isendtag, &mrecvright, 1, MPI_INT, parallel_decomp->right_pe, 
		 irecvtag, parallel_decomp->toroidal_comm, &istatus2);
    
    /* allocate mem for incoming particle data */
    if (mrecvleft + mrecvright > parallel_decomp->recvbuf_size) {
      fprintf(stderr, "Error! PE %d, mrecvleft %d, mrecvright %d, "
	      "recvbuf_size %d\n", parallel_decomp->mype, 
	      mrecvleft, mrecvright,
	      parallel_decomp->recvbuf_size);
      exit(1);
    }
#if  _TRACK_SHIFTED_PARTICLES 
    else {
      fprintf(stderr, "PE %d, mrecvleft %d, mrecvright %d, "
	      "sendleft %d, sendright %d\n", parallel_decomp->mype, 
	      mrecvleft, mrecvright,
	      msendleft, msendright);
      
    }
#endif
    recvleft = parallel_decomp->recvbuf;
    recvright = parallel_decomp->recvbuf + nparam*mrecvleft;
    
    /* send particles to right neighbor, recv from left */
    irecvtag = parallel_decomp->left_pe;
#if FINE_TIMER
    real start_t = MPI_Wtime();
#endif
    MPI_Sendrecv(sendright, msendright*nparam, MPI_MYREAL, 
		 parallel_decomp->right_pe,
		 isendtag, recvleft, mrecvleft*nparam, MPI_MYREAL, 
		 parallel_decomp->left_pe,
		 irecvtag, parallel_decomp->toroidal_comm, &istatus1);
    
    /* send particles to left neighbor, recv from right */
    irecvtag = parallel_decomp->right_pe;
    MPI_Sendrecv(sendleft, msendleft*nparam, MPI_MYREAL, 
		 parallel_decomp->left_pe, 
		 isendtag, recvright, mrecvright*nparam, MPI_MYREAL, 
		 parallel_decomp->right_pe,
		 irecvtag, parallel_decomp->toroidal_comm, &istatus2);
#if FINE_TIMER
    real end_t = MPI_Wtime();
    shift_t_comm2 += (end_t - start_t);
#endif
    /* copy received data to particle arrays */
    assert(mi_end + mrecvleft + mrecvright < mimax);

#if PROFILE_CYCLE 
    end = read_cycle();
    cycle_measurements[last_phase++] = end - start;
    start = end;
#endif

#if 0
    free(sendleft);
    free(sendright);
#endif    
    
    //zion = (real *)realloc(zion, sizeof(real)*nparam*(mi_end + mrecvleft + mrecvright));
    //memcpy(zion+mi_end*nparam,recvleft, sizeof(real)* nparam* (mrecvleft+mrecvright));
    /*
    int offset = mi_end*nparam;
#pragma omp parallel for 
    for (int i=0; i<mrecvleft+mrecvright; i++){
      for (int j=0; j<nparam; j++){
	zion[offset+nparam*i+j] = recvleft[nparam*i+j];
      }
    }
    */
    
    assert(mimax_shift>(mi_end+mrecvleft+mrecvright));

#pragma omp parallel for
    for (int j=0; j<nparam; j++){
      for (int i=0; i<mrecvleft; i++){
	zion[j*mimax_shift + mi_end + i] = recvleft[j*mrecvleft+i];
      }

      for (int i=0; i<mrecvright; i++){
	zion[j*mimax_shift + mi_end + mrecvleft + i] = recvright[j*mrecvright+i];
      }
    }
    
    /* update m0, m_end, and other global counts */
    m0 = mi_end;
    mi += mrecvleft + mrecvright;
    mi_end = mi;

#if  TRACK_SHIFTED_PARTICLES 
    if (parallel_decomp->mype == 0)
      fprintf(stderr, "[%d %d], send %d %d "
	      "recv %d %d mi %d\n",
	      istep, irk,
	      msendleft, msendright, 
	      mrecvleft, mrecvright, params->mi - params->holecount);
#endif
    
#if PROFILE_CYCLE 
    end = read_cycle();
    cycle_measurements[last_phase++] = end - start;
    start = end;
#endif    
    
  } /* while section */

#if PROFILE_CYCLE 
  end = read_cycle();
  cycle_measurements[last_phase++] = end - start;
  start = end;
#endif
  
  /*Remove remaining holes*/
  int dest = 0, holes = 0,i,j;
  for(i = 0;i<mi_end;i++) {
    //real z2val = zion[i*nparam+2];
    real z2val = zion[2*mimax_shift+i];
    if (z2val == HOLEVAL) {
      holes ++;
      continue;
    }
 
    if(dest!=i){
      for(j=0;j<nparam;j++)
	//zion[nparam*dest+j] = zion[nparam*i + j];
	zion[j*mimax_shift+dest] = zion[j*mimax_shift+i];
    }

    dest++;
  }
  mi_end-= holes;
  
#if PROFILE_CYCLE 
  end = read_cycle();
  cycle_measurements[last_phase++] = end - start;
  start = end;
#endif

  call_gpu_shifti_append_kernel (gtc_input,gpu_kernel_input, mi_end, zion);

  //if(zion)
  free(zion);

  if (iteration == (ntoroidal+1)) {
    fprintf(stderr, "Error! Endless particle sorting loop at PE %d\n",
	    parallel_decomp->mype);
  }

  _mm_free(mshift);

#endif
  
  
#if PROFILE_CYCLE 
  end = read_cycle();
  cycle_measurements[last_phase++] = end - start;
  start = end;
  if(istep %5 ==0) {
    printf("printing timing for step %d\n", istep);
    for(i=0;i<last_phase;i++) 
      printf("phase %d cycle: %lld\n", i,cycle_measurements[i]);
  }
#endif
  
  return 0;
}


int gpu_shifti_radial(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input)
{
#if PROFILE_CYCLE
  
#define MEASUREMENT_PHASES 200
  unsigned long last_phase=0, start=read_cycle(), end;
  unsigned long cycle_measurements[MEASUREMENT_PHASES];
#endif
  
#if USE_MPI
  gtc_global_params_t     *params;
  gtc_particle_data_t     *particle_data;
  gtc_aux_particle_data_t *aux_particle_data;
  gtc_particle_decomp_t   *parallel_decomp;
  gtc_radial_decomp_t     *radial_decomp;

  int  mimax;
  int nradial_dom, myrank_radiald;
  real a_nover_in, a_nover_out;

  int mi_end;

  int *kzi;

  int *mshift;

  int nparam;
  real pi, pi2, pi_inv;

  int m0, iteration;
  int max_threads;

  params            = &(gtc_input->global_params);
  particle_data     = &(gtc_input->particle_data);
  aux_particle_data = &(gtc_input->aux_particle_data);
  parallel_decomp   = &(gtc_input->parallel_decomp);
  radial_decomp     = &(gtc_input->radial_decomp);

  mimax = params->mimax;
  pi = params->pi;

  nradial_dom = radial_decomp->nradial_dom;
  myrank_radiald = radial_decomp->myrank_radiald;

  a_nover_in = radial_decomp->a_nover_in;
  a_nover_out = radial_decomp->a_nover_out;

  kzi = aux_particle_data->kzi;

  nparam = 12;
  pi_inv = 1.0/pi;
  pi2 = 2.0*pi;

  /********/

  if (nradial_dom == 1)
    return 0;

  m0 = 0;
  mi_end = 0;
  iteration = 0;
  
  max_threads = 32;
  mshift = (int *) _mm_malloc(max_threads * 8 * sizeof(int),
                              IDEAL_ALIGNMENT);
  assert(mshift != NULL);
  unsigned int tops[3];
  real *sends[3];
  sends[0] = sends[1] = sends[2] = NULL;
  int mimax_shift = gpu_kernel_input->d_max_shift_mi;
  real *zion = (real *) malloc(sizeof(real)*nparam*mimax_shift);
  assert(zion != NULL);

  int mi = 0;
  
  while (iteration <= nradial_dom) {
    real *sendright, *sendleft;
    real *recvleft, *recvright;
    int msend, mrecv, msendright, msendleft, pos;
    int mrecvleft, mrecvright;
    int isendtag, irecvtag;
    MPI_Status istatus1, istatus2;

    if(iteration == 0) {
#if PROFILE_CYCLE
      end = read_cycle();
      cycle_measurements[last_phase++] = end - start;
      start = end;
#endif
      call_gpu_shifti_extract_kernel(gtc_input,gpu_kernel_input, tops, sends, 1);
      msend = tops[1] + tops[2];
      msendright = tops[2];
      msendleft = tops[1];
      sendright = sends[2];
      sendleft = sends[1];
      
#if _DEBUG_GPU
      int misclassified_sendleft = 0;
      int misclassified_sendright = 0;
      int misclassified_send = 0;
      int i;
      //real zetaright, zetaleft;
      //real z0val, zion0m, r;
      real radialright, radailleft;
      real z2val, psitmp, r;

      for (i=0; i<tops[1]; i++) {
	//z2val = sendleft[i*nparam+2];
	z2val = sendleft[2*tops[1]+i];
	if (z2val == HOLEVAL){
	  continue;
	}
        //psitmp = sendleft[i*nparam+0];
	psitmp = sendleft[0*tops[1]+i];
#if SQRT_PRECOMPUTED
	r = psitmp;
#else
	r = sqrt(2.0*psitmp);
#endif
	if (r<a_nover_in) {
	  /* Just oK*/

        } else
          misclassified_sendleft++;
      }

      for (i=0; i<tops[2]; i++) {
	//z2val =sendright[i*nparam+2];
	z2val = sendright[2*tops[2]+i];
	if (z2val == HOLEVAL){
          continue;
	}

	//psitmp = sendright[i*nparam+0];
	psitmp = sendright[0*tops[2]+i];
#if SQRT_PRECOMPUTED
	r = psitmp;
#else   
        r = sqrt(2.0*psitmp);
#endif
	if (r>a_nover_out){
	  /*Just ok */
	 
        } else
          misclassified_sendright++;
      }
      fprintf(stderr,"Radial PE=%d send left %d, misclassfied left %d send right %d, misclassified right %d total misclassified: %d\n",
              radial_decomp->myrank_radiald, msendleft, misclassified_sendleft, msendright, misclassified_sendright,misclassified_send);
#endif // _DEBUG_GPU        
      
      iteration++;
    }else {
#if PROFILE_CYCLE
      end = read_cycle();
      cycle_measurements[last_phase++] = end - start;
      start = end;
#endif
      
      msend = msendright = msendleft = 0;
      
#pragma omp parallel
      {
        int mlstack_max, mrstack_max;
        int *lstack, *rstack, *stack_tmp;
        int lpos, rpos;
	
        int i;
        int tid, nthreads;
        //real zetaright, zetaleft;
        //real z2val, zion0m, r;
	real radialright, radailleft;
	real z2val, psitmp, r;

#ifdef _OPENMP
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#else
        tid = 0;
        nthreads = 1;
#endif
        assert(nthreads == parallel_decomp->nthreads);
        assert(nthreads <= max_threads);
        mshift[8*tid] = mshift[8*tid+1]
          = mshift[8*tid+2] = mshift[8*tid+3]
          = mshift[8*tid+4] = 0;
        mlstack_max = mrstack_max = 2048;
        lstack = (int *) malloc(mlstack_max * sizeof(int));
        rstack = (int *) malloc(mrstack_max * sizeof(int));
        assert(lstack != NULL); assert(rstack != NULL);
#pragma omp barrier

#pragma omp for
        for (i=m0; i<mi_end; i++) {
          //z2val = zion[i*nparam+2];
	  z2val = zion[2*mimax_shift+i];

          if (z2val == HOLEVAL) {
            continue;
          }

          //psitmp = zion[i*nparam+0];
	  psitmp = zion[0*mimax_shift+i];
	  
#if SQRT_PRECOMPUTED
          r = psitmp;
#else
          r = sqrt(2.0*psitmp);
#endif
          if ((r<a_nover_in && myrank_radiald > 0) ||
              (r>a_nover_out && myrank_radiald < (nradial_dom-1))) {

            mshift[8*tid]++;
            /* particle to move right */
            if (r > a_nover_out) {
              rpos = mshift[8*tid+1]++;

              if (rpos == mrstack_max) {
                /* double the size */
                stack_tmp = (int *) malloc(2*mrstack_max*sizeof(int));
		assert(stack_tmp != NULL);
                memcpy(stack_tmp, rstack, mrstack_max*sizeof(int));
                free(rstack);
                rstack = stack_tmp;
                mrstack_max = 2 * mrstack_max;
              }
              rstack[rpos] = i;
              /* particle to move left */
            } else {
	      
              lpos = mshift[8*tid+2]++;
              if (lpos == mlstack_max) {
                /* double the size */
                stack_tmp = (int *)
                  malloc(2*mlstack_max*sizeof(int));
		assert(stack_tmp != NULL);
                memcpy(stack_tmp, lstack, mlstack_max*sizeof(int));
                free(lstack);
                lstack = stack_tmp;
                mlstack_max = 2 * mlstack_max;
              }
              lstack[lpos] = i;
            }
          }
        }
	
        /* Merge partial arrays */
        if (tid == 0) {
          mshift[8*0+3] = 0;
          mshift[8*0+4] = 0;
          for (i=1; i<nthreads; i++) {
            mshift[8*i+3] = mshift[8*(i-1)+3] + mshift[8*(i-1)+1];
            mshift[8*i+4] = mshift[8*(i-1)+4] + mshift[8*(i-1)+2];
          }
          msendright  = mshift[8*(nthreads-1)+3] + mshift[8*(nthreads-1)+1];
          msendleft   = mshift[8*(nthreads-1)+4] + mshift[8*(nthreads-1)+2];
          msend = msendright + msendleft;
	  if (msend > mi) {  
            fprintf(stderr, "Error! mype %d, msend %d, left %d, right"
                    " %d, mi %d\n",
                    parallel_decomp->mype, msend, msendleft, msendright, mi);
            exit(1);                                                            
	  }
        }

#pragma omp barrier

        memcpy(kzi+mshift[8*tid+3], rstack, mshift[8*tid+1]*sizeof(int));
        memcpy(kzi+msendright+mshift[8*tid+4], lstack,
               mshift[8*tid+2]*sizeof(int));
        free(lstack);
        free(rstack);
      } //omp parallel 

      iteration++;
      if (iteration > 1) {
        mrecv = 0;
        MPI_Allreduce(&msend, &mrecv, 1, MPI_INT, MPI_SUM, parallel_decomp->partd_comm);
        if (mrecv == 0) {
          break;
        }
        if (parallel_decomp->mype == 0) {
          fprintf(stderr, "radial iteration %d, mrecv %d\n", iteration,
                  mrecv);
	}
      }
       
      if (msendleft + msendright > parallel_decomp->sendbuf_size) {
	fprintf(stderr, "Error! PE %d, msendleft %d, msendright %d, "
		"sendbuf_size %d\n", parallel_decomp->mype, 
		msendleft, msendright,
		parallel_decomp->sendbuf_size);
	exit(1);
      } else {
	sendleft = parallel_decomp->sendbuf;
	sendright = parallel_decomp->sendbuf + nparam*msendleft;
      }
      

#if 0
      /* Allocate space for sendright and sendleft */
      sendright = (real *) malloc(nparam*msendright*sizeof(real));
      assert(sendright != NULL);

      sendleft = (real *) malloc(nparam*msendleft*sizeof(real));
      assert(sendleft != NULL);
#endif
      int i,j;
      /* pack particle data */
      /*
#pragma omp parallel for private(pos)
      for (i=0; i<msendright; i++) {
        pos = kzi[i];
        for(j=0;j<nparam;j++)
          sendright[nparam*i+j] = zion[pos*nparam + j];
        zion[pos*nparam +2] = HOLEVAL;
	zion[pos*nparam +8] = HOLEVAL;
      }

#pragma omp parallel for private(pos)
      for (i=0; i<msendleft; i++) {
        pos = kzi[i+msendright];
        for(j=0;j<nparam;j++)
          sendleft[nparam*i+j] = zion[pos*nparam + j];
	zion[pos*nparam+2] = HOLEVAL;
	zion[pos*nparam+8] = HOLEVAL;
      }
      */

      /* pack particle data */
#pragma omp parallel for private (pos,j,i)
      for (j=0; j<nparam; j++){
        for (i=0; i<msendright; i++){
          pos = kzi[i];
          sendright[j*msendright+i] = zion[j*mimax_shift+pos];
          if (j==2){
            zion[2*mimax_shift+pos] = HOLEVAL;
          }
        }
      }

    } // else second iteration forward 

#if PROFILE_CYCLE
    end = read_cycle();
    cycle_measurements[last_phase++] = end - start;
    start = end;
#endif

    /* send # of particles to move right, get # to recv from left */
    isendtag = parallel_decomp->myrank_partd;
    irecvtag = radial_decomp->left_radial_pe;
    mrecvleft = 0;
    MPI_Sendrecv(&msendright, 1, MPI_INT, radial_decomp->right_radial_pe,
                 isendtag, &mrecvleft, 1, MPI_INT, radial_decomp->left_radial_pe,
                 irecvtag, parallel_decomp->partd_comm, &istatus1);

    /* send # of particles to move left, get # to recv from right */
    irecvtag = radial_decomp->right_radial_pe;
    mrecvright = 0;
    MPI_Sendrecv(&msendleft, 1, MPI_INT, radial_decomp->left_radial_pe,
                 isendtag, &mrecvright, 1, MPI_INT, radial_decomp->right_radial_pe,
                 irecvtag, parallel_decomp->partd_comm, &istatus2);

    /* allocate mem for incoming particle data */
    if (mrecvleft + mrecvright > parallel_decomp->recvbuf_size) {
      fprintf(stderr, "Error! PE %d, mrecvleft %d, mrecvright %d, "
              "recvbuf_size %d\n", parallel_decomp->mype,
              mrecvleft, mrecvright,
              parallel_decomp->recvbuf_size);
      exit(1);
    }
#if  _TRACK_SHIFTED_PARTICLES
    else {
      fprintf(stderr, "PE %d, mrecvleft %d, mrecvright %d, "
              "sendleft %d, sendright %d\n", parallel_decomp->mype,
              mrecvleft, mrecvright,
              msendleft, msendright);

    }
#endif
    recvleft = parallel_decomp->recvbuf;
    recvright = parallel_decomp->recvbuf + nparam*mrecvleft;

    /* send particles to right neighbor, recv from left */
    irecvtag = radial_decomp->left_radial_pe;
    MPI_Sendrecv(sendright, msendright*nparam, MPI_MYREAL,
                 radial_decomp->right_radial_pe,
                 isendtag, recvleft, mrecvleft*nparam, MPI_MYREAL,
                 radial_decomp->left_radial_pe,
                 irecvtag, parallel_decomp->partd_comm, &istatus1);

    /* send particles to left neighbor, recv from right */
    irecvtag = radial_decomp->right_radial_pe;
    MPI_Sendrecv(sendleft, msendleft*nparam, MPI_MYREAL,
                 radial_decomp->left_radial_pe,
                 isendtag, recvright, mrecvright*nparam, MPI_MYREAL,
                 radial_decomp->right_radial_pe,
                 irecvtag, parallel_decomp->partd_comm, &istatus2);

    /* copy received data to particle arrays */
#if PROFILE_CYCLE
    end = read_cycle();
    cycle_measurements[last_phase++] = end - start;
    start = end;
#endif

#if 0
    free(sendleft);
    free(sendright);
#endif  

    //zion = (real*) realloc(zion, sizeof(real)*nparam*(mi_end + mrecvleft + mrecvright));
    //memcpy(zion+mi_end*nparam,recvleft, sizeof(real)* nparam* (mrecvleft+mrecvright));
    /*
    int offset = mi_end*nparam;
#pragma omp parallel for
    for (int i=0; i<mrecvleft+mrecvright; i++){
      for (int j=0; j<nparam; j++){
        zion[offset+nparam*i+j] = recvleft[nparam*i+j];
      }
    }
    */

    assert(mimax_shift>(mi_end+mrecvleft+mrecvright));
#pragma omp parallel for
    for (int j=0; j<nparam; j++){
      for (int i=0; i<mrecvleft; i++){
        zion[j*mimax_shift + mi_end + i] = recvleft[j*mrecvleft+i];
      }

      for (int i=0; i<mrecvright; i++){
        zion[j*mimax_shift + mi_end + mrecvleft + i] = recvright[j*mrecvright+i];
      }
    }

    /* update m0, m_end, and other global counts */
    m0 = mi_end;
    mi += mrecvleft + mrecvright;
    mi_end = mi;

#if  _TRACK_SHIFTED_PARTICLES
    if (parallel_decomp->mype == 0)
      fprintf(stderr, "[%d %d], send %d %d "
              "recv %d %d mi %d\n",
              istep, irk,
              msendleft, msendright,
              mrecvleft, mrecvright, params->mi - params->holecount);
#endif

#if PROFILE_CYCLE
    end = read_cycle();
    cycle_measurements[last_phase++] = end - start;
    start = end;
#endif

  } /* while section */

#if PROFILE_CYCLE
  end = read_cycle();
  cycle_measurements[last_phase++] = end - start;
  start = end;
#endif

  /*Remove remaining holes*/
  int dest = 0, holes = 0,i,j;
  for(i = 0;i<mi_end;i++) {
    real z2val = zion[i*nparam+2];
    if (z2val == HOLEVAL) {
      holes ++;
      continue;
    }

    if(dest!=i)
      for(j=0;j<nparam;j++)
	//zion[nparam*dest+j] = zion[nparam*i + j];
	zion[j*mimax_shift+dest] = zion[j*mimax_shift+i];
    dest++;
  }

  mi_end-= holes;

#if PROFILE_CYCLE
  end = read_cycle();
  cycle_measurements[last_phase++] = end - start;
  start = end;
#endif
  
  call_gpu_shifti_append_kernel (gtc_input,gpu_kernel_input, mi_end,zion);

  //if(zion)
  free(zion);

  if (iteration == (nradial_dom+1)) {
    fprintf(stderr, "Error! Endless particle sorting loop at PE %d\n",
	    parallel_decomp->mype);
  }

  _mm_free(mshift);
#endif
  
#if PROFILE_CYCLE
  end = read_cycle();
  cycle_measurements[last_phase++] = end - start;
  start = end;
  if(istep %5 ==0) {
    printf("printing timing for step %d\n", istep);
    for(i=0;i<last_phase;i++)
	printf("phase %d cycle: %lld\n", i,cycle_measurements[i]);
  }
#endif
  
  return 0;
}



#if	PARTICLE_BINNING
int gpu_bin_particles(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int shift_direction) 
{  
  gpu_kernel_input->irk = irk;
  gpu_kernel_input->istep = istep;
#if !SYNERGESTIC_SORT_SHIFT
  if((irk==2) && (istep%RADIAL_THETA_BIN_PERIOD==0))
#endif
    call_gpu_bin_particles_kernel(gtc_input,gpu_kernel_input,shift_direction);
  return 0;
}
#else

#if SYNERGESTIC_SORT_SHIFT
int gpu_bin_particles(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int shift_direction)

{
  gpu_kernel_input->irk = irk;
  gpu_kernel_input->istep = istep;
  call_gpu_bin_particles_kernel(gtc_input,gpu_kernel_input,shift_direction);
  return 0;
}
#endif

#endif


