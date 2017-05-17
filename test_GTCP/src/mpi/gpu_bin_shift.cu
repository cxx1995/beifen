#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <stddef.h>
#include <assert.h>
#include <limits.h>
//using namespace thrust;
#define NO_SHIFT	0
#define SHIFT_LEFT	1
#define SHIFT_RIGHT	2
__device__ int d_start_shift[3];

 static inline __device__ void init_shift_bounds(int mi)
{
		d_start_shift[0] = mi;
		d_start_shift[SHIFT_LEFT] = mi;
		d_start_shift[SHIFT_RIGHT] = mi;
}

#define SHIFT_OFFSET 0x10000000

__global__ static void calculate_sort_keypair(gtc_particle_data_t *zion, int *seq, int *psi_theta)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int m = tid + bid*nthreads;

    real * z0 = zion->z0;
    real * z1 = zion->z1;
    
    int mi = params.mi;
    real a0 = params.a0;
    real delr = params.delr;
    int mpsi = params.mpsi;
    real r;

    if (m < mi){
      if(m==0) 
	init_shift_bounds(mi);

      seq[m]=m;
      real pi2_inv = params.pi2_inv;
      real psi     = z0[m];
      real theta   = z1[m];
      real zetamin = params.zetamin;
      real zetamax = params.zetamax;
      
      real pi2 = 2*params.pi;
      real * zion2 = zion->z2;
      real zetatmp = zion2[m];
      real zetaright=min(pi2,zetatmp)-zetamax;
      real zetaleft = zetatmp - zetamin;
      int shift = zetaright*zetaleft>0;
      zetaright = zetaright*pi2_inv;
      zetaright = zetaright-floor(zetaright);
      int right = zetaright<0.5;
      
#if SQRT_PRECOMPUTED
      r        = psi;
#else
      r        = sqrt(2.0*psi);
#endif
      
      int iptmp    = (int) ((r-a0)*delr+0.5);
      int ip = d_abs_min_int(mpsi, iptmp);
#if THETA_BIN_ORG
      //int jttmp    = (int) (theta*pi2_inv*delt[ip]+0.5);
      //psi_theta[m] = d_abs_min_int(mtheta[ip], jttmp) + shift *(1+right) * SHIFT_OFFSET;
      psi_theta[m] = ip + shift * (1+ right)* SHIFT_OFFSET;
#else
      psi_theta[m] = theta*100.0 + ip*700+ shift *(1+right) * SHIFT_OFFSET;
#endif
    }
}

__global__ static void calculate_sort_keypair_radial(gtc_particle_data_t *zion, int *seq, int *psi_theta)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int m = tid + bid*nthreads;
    real * z0 = zion->z0;
    real * z1 = zion->z1;
    
    const real a_nover_in = radial_decomp.a_nover_in;
    const real a_nover_out = radial_decomp.a_nover_out;
    const int myrank_radiald = radial_decomp.myrank_radiald;
    const int nradial_dom = radial_decomp.nradial_dom;

    int mi = params.mi;
    real a0 = params.a0;
    real delr = params.delr;
    int mpsi = params.mpsi;
    real r;
    
    if (m < mi){
      if(m==0)
	init_shift_bounds(mi);
      
      seq[m]=m;
      real pi2_inv = params.pi2_inv;
      real psi     = z0[m];
      real theta   = z1[m];
      //real zetamin = params.zetamin;
      //real zetamax = params.zetamax;
      
      real pi2 = 2*params.pi;
      //real * zion2 = zion->z2;
      //real zetatmp = zion2[m];

#if SQRT_PRECOMPUTED
      r        = psi;
#else
      r        = sqrt(2.0*psi);
#endif

      //      real zetaright=min(pi2,zetatmp)-zetamax;
      //      real zetaleft = zetatmp - zetamin;
      //      int shift = zetaright*zetaleft>0;
      //      zetaright = zetaright*pi2_inv;
      //      zetaright = zetaright-floor(zetaright);
      //      int right = zetaright<0.5;

      int shift = ((r<a_nover_in && myrank_radiald > 0)||(r>a_nover_out && myrank_radiald < (nradial_dom-1)));
      int right = (r>a_nover_out && myrank_radiald < (nradial_dom-1));
      
      int iptmp    = (int) ((r-a0)*delr+0.5);
      int ip = d_abs_min_int(mpsi, iptmp);
#if THETA_BIN_ORG
      //int jttmp    = (int) (theta*pi2_inv*delt[ip]+0.5);
      //psi_theta[m] = d_abs_min_int(mtheta[ip], jttmp) + shift *(1+right) * SHIFT_OFFSET;
      psi_theta[m] = ip + shift*(1 + right)* SHIFT_OFFSET;
#else
      psi_theta[m] = theta*100.0 + ip*700+ shift *(1+right) * SHIFT_OFFSET;
#endif
 }
}

__global__ static void calculate_sort_keypair_shift(gtc_particle_data_t *zion, int *seq, int *psi_theta)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int m = tid + bid*nthreads;
    int mi = params.mi;

    if (m < mi){
      if(m==0) 
	init_shift_bounds(mi);
      seq[m]=m;
      real pi2_inv = params.pi2_inv;
      real zetamin = params.zetamin;
      real zetamax = params.zetamax;
      real pi2 = 2.0*params.pi;
      real *zion2 = zion->z2;
      real zetatmp = zion2[m];
      real zetaright=min(pi2,zetatmp)-zetamax;
      real zetaleft = zetatmp - zetamin;
      int shift = (zetaright*zetaleft)>0.0;
      zetaright = zetaright*pi2_inv;
      zetaright = zetaright-floor(zetaright);
      int right = zetaright<0.5;
      //		psi_theta[m] = shift << right;
      psi_theta[m] = shift? (right?2:1):0;

    }
}

__global__ static void calculate_sort_keypair_radial_shift(gtc_particle_data_t *zion, int *seq, int *psi_theta)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int m = tid + bid*nthreads;
    
    const real a_nover_in = radial_decomp.a_nover_in;
    const real a_nover_out = radial_decomp.a_nover_out;
    const int myrank_radiald = radial_decomp.myrank_radiald;
    const int nradial_dom = radial_decomp.nradial_dom;
    
    int mi = params.mi;
    
    if (m < mi){
      if(m==0)
	init_shift_bounds(mi);
      seq[m]=m;
      real pi2_inv = params.pi2_inv;
      real zetamin = params.zetamin;
      real zetamax = params.zetamax;
      real pi2 = 2*params.pi;
      
      real * zion0 = zion->z0;
      real z0 = zion0[m];
      
#if SQRT_PRECOMPUTED
      real r = z0;
#else
      real r = sqrt(2.0*z0);
#endif
      
      //                int shift = zetaright*zetaleft>0;
      //                zetaright = zetaright*pi2_inv;
      //                zetaright = zetaright-floor(zetaright);
      //                int right = zetaright<0.5;
      
      int shift = ((r<a_nover_in && myrank_radiald > 0)||(r>a_nover_out && myrank_radiald < (nradial_dom-1)));
      int right = (r>a_nover_out && myrank_radiald < (nradial_dom-1));
      
      psi_theta[m] = shift? (right?2:1):0;
      
    }
}

__global__ static void permute_particles_zion_ph1(gtc_particle_data_t *d_zion, gtc_particle_data_t *d_auxs_zion, int *permutation, int irk) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int gid = tid + bid*nthreads;
    int mi = params.mi;

    if (gid < mi){
      int pos = permutation[gid];
      
#if PTX_STREAM_INTRINSICS
      real z0, z1, z2, z3, z4, z00, z01, z02, z03, z04;
      z0 = TunedTexLoad<real,CA>::Ld(d_zion->z0+pos);
      z1 = TunedTexLoad<real,CA>::Ld(d_zion->z1+pos);
      z2 = TunedTexLoad<real,CA>::Ld(d_zion->z2+pos);
      z3 = TunedTexLoad<real,CA>::Ld(d_zion->z3+pos);
      z4 = TunedTexLoad<real,CA>::Ld(d_zion->z4+pos);

      TunedStore<real,CS>::St(z0,d_auxs_zion->z0+gid);
      TunedStore<real,CS>::St(z1,d_auxs_zion->z1+gid);
      TunedStore<real,CS>::St(z2,d_auxs_zion->z2+gid);
      TunedStore<real,CS>::St(z3,d_auxs_zion->z3+gid);
      TunedStore<real,CS>::St(z4,d_auxs_zion->z4+gid);

      if (irk==1) {
         z00 = TunedTexLoad<real,CA>::Ld(d_zion->z00+pos);
         z01 = TunedTexLoad<real,CA>::Ld(d_zion->z01+pos);
         z02 = TunedTexLoad<real,CA>::Ld(d_zion->z02+pos);
         z03 = TunedTexLoad<real,CA>::Ld(d_zion->z03+pos);
         z04 = TunedTexLoad<real,CA>::Ld(d_zion->z04+pos);

         TunedStore<real,CS>::St(z00,d_auxs_zion->z00+gid);
         TunedStore<real,CS>::St(z01,d_auxs_zion->z01+gid);
         TunedStore<real,CS>::St(z02,d_auxs_zion->z02+gid);
         TunedStore<real,CS>::St(z03,d_auxs_zion->z03+gid);
         TunedStore<real,CS>::St(z04,d_auxs_zion->z04+gid);
      }
#else
      (d_auxs_zion->z0)[gid] = (d_zion->z0)[pos];
      (d_auxs_zion->z1)[gid] = (d_zion->z1)[pos];
      (d_auxs_zion->z2)[gid] = (d_zion->z2)[pos];
      (d_auxs_zion->z3)[gid] = (d_zion->z3)[pos];
      (d_auxs_zion->z4)[gid] = (d_zion->z4)[pos];

      if (irk==1){
         (d_auxs_zion->z00)[gid] = (d_zion->z00)[pos];
         (d_auxs_zion->z01)[gid] = (d_zion->z01)[pos];
         (d_auxs_zion->z02)[gid] = (d_zion->z02)[pos];
         (d_auxs_zion->z03)[gid] = (d_zion->z03)[pos];
         (d_auxs_zion->z04)[gid] = (d_zion->z04)[pos];
      }
#endif
    }
}

__global__ static void permute_particles_zion_ph2(gtc_particle_data_t *d_zion, gtc_particle_data_t *d_auxs_zion, int *permutation,int *keys) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int gid = tid + bid*nthreads;
    int mi = params.mi;
    
    if (gid < mi){
      int pos = permutation[gid];
      int key_val = keys[gid];
      int prev_key_val = gid>0?keys[gid-1]:keys[0];

      if(key_val >= SHIFT_LEFT * SHIFT_OFFSET && key_val < SHIFT_RIGHT * SHIFT_OFFSET &&  prev_key_val < SHIFT_LEFT*SHIFT_OFFSET) 
	d_start_shift[SHIFT_LEFT] = gid;	
      if(key_val >= SHIFT_RIGHT*SHIFT_OFFSET &&  prev_key_val < SHIFT_RIGHT*SHIFT_OFFSET)
	d_start_shift[SHIFT_RIGHT] = gid;

#if PTX_STREAM_INTRINSICS
      real z5, z05;
      z5 = TunedTexLoad<real,CA>::Ld(d_zion->z5+pos);
      z05 = TunedTexLoad<real,CA>::Ld(d_zion->z05+pos);
      TunedStore<real,CS>::St(z5,d_auxs_zion->z5+gid);
      TunedStore<real,CS>::St(z05,d_auxs_zion->z05+gid);
#else
      (d_auxs_zion->z5)[gid] = (d_zion->z5)[pos];
      (d_auxs_zion->z05)[gid] = (d_zion->z05)[pos];
#endif
    }
}

__global__ static void permute_particles_zion_shift_ph2(gtc_particle_data_t *d_zion, gtc_particle_data_t *d_auxs_zion, int *permutation,int *keys) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int gid = tid + bid*nthreads;
    int mi = params.mi;
    
    if (gid < mi){
      int  pos = permutation[gid];
      int key_val = keys[gid];
      int prev_key_val = gid>0?keys[gid-1]:keys[0];
      if(key_val == SHIFT_LEFT &&  prev_key_val == NO_SHIFT) 
	d_start_shift[SHIFT_LEFT] = gid;
      
      if((key_val == SHIFT_RIGHT &&  prev_key_val == SHIFT_LEFT)||
	 (key_val == SHIFT_RIGHT &&  prev_key_val == NO_SHIFT))
	d_start_shift[SHIFT_RIGHT] = gid;
      
#if PTX_STREAM_INTRINSICS
      real z5, z05;
      z5 = TunedTexLoad<real,CA>::Ld(d_zion->z5+pos);
      z05 = TunedTexLoad<real,CA>::Ld(d_zion->z05+pos);
      TunedStore<real,CS>::St(z5,d_auxs_zion->z5+gid);
      TunedStore<real,CS>::St(z05,d_auxs_zion->z05+gid);
#else
      (d_auxs_zion->z5)[gid] = (d_zion->z5)[pos];
      (d_auxs_zion->z05)[gid] = (d_zion->z05)[pos];
#endif
    }
}

__global__ static void update_particle_zion_ph1(gtc_particle_data_t *d_zion, gtc_particle_data_t *d_auxs_zion, int irk) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int gid = tid + bid*nthreads;
    int mi = params.mi;

    if (gid < mi){
      (d_zion->z0)[gid] = (d_auxs_zion->z0)[gid];
      (d_zion->z1)[gid] = (d_auxs_zion->z1)[gid];
      (d_zion->z2)[gid] = (d_auxs_zion->z2)[gid];
      (d_zion->z3)[gid] = (d_auxs_zion->z3)[gid];
      (d_zion->z4)[gid] = (d_auxs_zion->z4)[gid];

      if (irk==1){
         (d_zion->z00)[gid] = (d_auxs_zion->z00)[gid];
         (d_zion->z01)[gid] = (d_auxs_zion->z01)[gid];
         (d_zion->z02)[gid] = (d_auxs_zion->z02)[gid];
         (d_zion->z03)[gid] = (d_auxs_zion->z03)[gid];
         (d_zion->z04)[gid] = (d_auxs_zion->z04)[gid];
      }
    }
}

__global__ static void update_particle_zion_ph2(gtc_particle_data_t *d_zion, gtc_particle_data_t *d_auxs_zion) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int nthreads = blockDim.x;
    const int gid = tid + bid*nthreads;
    int mi = params.mi;

    if (gid < mi) {
      (d_zion->z5)[gid] = (d_auxs_zion->z5)[gid];
      (d_zion->z05)[gid] = (d_auxs_zion->z05)[gid];
    }
}


void call_gpu_bin_particles_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int shift_direction)
{
     gpu_timer_start(gpu_kernel_input);
     int mi = gtc_input->global_params.mi;
     int nt = 512;
     int nb = (mi + nt - 1)/nt;
     int irk = gpu_kernel_input->irk;
     int istep = gpu_kernel_input->istep;

     gtc_sort_particle_t	*d_sort =  &(gpu_kernel_input->d_sort);
#if SYNERGESTIC_SORT_SHIFT
     
#if PARTICLE_BINNING
     int shift_only = (irk!=2) || (istep%RADIAL_THETA_BIN_PERIOD!=0) || shift_direction==1;
#else
     int shift_only = 1;
#endif
     if(shift_only)
       if (shift_direction == 0) 
	 calculate_sort_keypair_shift<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, d_sort->d_value, d_sort->d_sort_key);
       else if (shift_direction == 1)
	 calculate_sort_keypair_radial_shift<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, d_sort->d_value, d_sort->d_sort_key);
       else
	 printf("other shift_direction options are not available\n");
     else {
#endif 
       if (shift_direction == 0)
	 calculate_sort_keypair<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, d_sort->d_value, d_sort->d_sort_key);
       else if (shift_direction == 1)
	 calculate_sort_keypair_radial<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, d_sort->d_value, d_sort->d_sort_key);
       else
	 printf("other shift_direction options are not available\n");
#if SYNERGESTIC_SORT_SHIFT
     }
#endif
     thrust::device_ptr<int32_t> d_key(d_sort->d_sort_key);
     thrust::device_ptr<int32_t> d_value(d_sort->d_value);
     thrust::stable_sort_by_key(d_key, d_key + mi, d_value);
     gpu_kernel_input->gpu_timing.device_particle_sort_time += gpu_timer_measure(gpu_kernel_input);
     
#if SYNERGESTIC_SORT_SHIFT
     if(shift_only)
       permute_particles_zion_shift_ph2<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_auxs_zion, d_sort->d_value,d_sort->d_sort_key);
     else
#endif
       permute_particles_zion_ph2<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_auxs_zion, d_sort->d_value,d_sort->d_sort_key);
     update_particle_zion_ph2<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_auxs_zion);
     
     permute_particles_zion_ph1<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_auxs_zion, d_sort->d_value, irk);
     update_particle_zion_ph1<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_auxs_zion, irk);
     
     gpu_kernel_input->gpu_timing.device_particle_bin_time += gpu_timer_measure_end(gpu_kernel_input);     
}

static int h_start_shift[3];
#define NPARAMS 12

extern "C"
void call_gpu_shifti_extract_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, unsigned int tops[3], real *sends[3], int shift_direction)
{
     gpu_timer_start(gpu_kernel_input);
     
     CUDA_SAFE_CALL(cudaMemcpyFromSymbol(h_start_shift, d_start_shift, sizeof(int)*3,0,cudaMemcpyDeviceToHost));
     int shift_left_size, shift_right_size,total_shift;
     int start_shift_left;
     int mi = gtc_input->global_params.mi;
     
     assert(mi == h_start_shift[0]);
     if(h_start_shift[SHIFT_LEFT] == mi)
       h_start_shift[SHIFT_LEFT] = h_start_shift[SHIFT_RIGHT];
     
     if(h_start_shift[SHIFT_LEFT]< mi) {
       shift_left_size = h_start_shift[SHIFT_RIGHT]  - h_start_shift[SHIFT_LEFT];
       start_shift_left =  h_start_shift[SHIFT_LEFT];
     } else
       shift_left_size = 0;	
     
     if (h_start_shift[SHIFT_RIGHT] < mi) {
       shift_right_size = mi - h_start_shift[SHIFT_RIGHT];
     } else 
       shift_right_size = 0;
     
     tops[SHIFT_LEFT] = shift_left_size;
     tops[SHIFT_RIGHT] = shift_right_size;
     total_shift = shift_left_size + shift_right_size;

     if(total_shift == 0) {
       gpu_kernel_input->gpu_timing.memtransfer_shift_time += gpu_timer_measure_end(gpu_kernel_input);
       return;
     }
   
     gtc_particle_decomp_t   *parallel_decomp =&(gtc_input->parallel_decomp);
     
     if (2*total_shift >= parallel_decomp->sendbuf_size) {
       fprintf(stderr, "Error! GPU PE %d, shift_left_size %d, shift_right_size %d, "
	       "sendbuf_size %d\n", parallel_decomp->mype, 
	       shift_left_size, shift_right_size,
	       parallel_decomp->sendbuf_size);
       exit(1);
     } else {
       sends[SHIFT_LEFT] = parallel_decomp->sendbuf;
       sends[SHIFT_RIGHT] = parallel_decomp->sendbuf + NPARAMS*shift_left_size;
       sends[0] = parallel_decomp->sendbuf + NPARAMS*total_shift;
     } 
     
     cudaStream_t 	stream;
     CUDA_SAFE_CALL(cudaStreamCreate(&stream));
     int d_mimax = gpu_kernel_input->d_mimax;
     int i,j,k;

     /*
     for(i=0;i<NPARAMS;i++)
       CUDA_SAFE_CALL(cudaMemcpyAsync((void*)(sends[0]+i*total_shift), gpu_kernel_input->d_zion.z0+i*d_mimax+start_shift_left, total_shift * sizeof(real), cudaMemcpyDeviceToHost,stream));
     
     gtc_input->global_params.mi -= total_shift;
     CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
     
     for(j=0;j<NPARAMS;j++){
       for(i=0;i<shift_left_size;i++)
	 sends[SHIFT_LEFT][i*NPARAMS+j] = sends[0][j*total_shift+i];
       for(k=0;k<shift_right_size;i++,k++)
	 sends[SHIFT_RIGHT][k*NPARAMS+j] = sends[0][j*total_shift+i];
     }
     */

     for (i=0; i<NPARAMS; i++){
          CUDA_SAFE_CALL(cudaMemcpyAsync((void*)(sends[SHIFT_LEFT]+i*shift_left_size), gpu_kernel_input->d_zion.z0+i*d_mimax+start_shift_left, shift_left_size * sizeof(real), cudaMemcpyDeviceToHost,stream));
          CUDA_SAFE_CALL(cudaMemcpyAsync((void*)(sends[SHIFT_RIGHT]+i*shift_right_size), gpu_kernel_input->d_zion.z0+i*d_mimax+start_shift_left+shift_left_size, shift_right_size * sizeof(real), cudaMemcpyDeviceToHost,stream));
     }
                    
     gtc_input->global_params.mi -= total_shift;	
     CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
     CUDA_SAFE_CALL(cudaStreamDestroy(stream));

     gpu_kernel_input->gpu_timing.memtransfer_shift_time += gpu_timer_measure_end(gpu_kernel_input);

}

extern "C"
void call_gpu_shifti_append_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int mi_append, real *particle_data)
{
     if(mi_append==0)
       return;
     gpu_timer_start(gpu_kernel_input);
     cudaStream_t 	stream;
     CUDA_SAFE_CALL(cudaStreamCreate(&stream));
     int d_mimax = gpu_kernel_input->d_mimax;
     int mimax_shift = gpu_kernel_input->d_max_shift_mi;
     int i, j;  
   
/*
     real * transposed = (real*) malloc(sizeof(real)*NPARAMS*mi_append);
     
     for(i=0;i<mi_append;i++)
       for(j=0;j<NPARAMS;j++)
	 transposed[j*mi_append+i] =particle_data[i*NPARAMS+j];
 
     for(i=0;i<NPARAMS;i++) 
       CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_kernel_input->d_zion.z0+i*d_mimax+h_start_shift[SHIFT_LEFT],transposed+i*mi_append, mi_append*sizeof(real),cudaMemcpyHostToDevice,stream));

*/

     for(i=0;i<NPARAMS;i++)                           
       CUDA_SAFE_CALL(cudaMemcpyAsync(gpu_kernel_input->d_zion.z0+i*d_mimax+h_start_shift[SHIFT_LEFT],particle_data+i*mimax_shift,
     mi_append* sizeof(real),cudaMemcpyHostToDevice,stream));

     CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
     CUDA_SAFE_CALL(cudaStreamDestroy(stream));
     
     gpu_kernel_input->gpu_timing.memtransfer_shift_time += gpu_timer_measure(gpu_kernel_input);

     gtc_input->global_params.mi +=  mi_append;

     //free(transposed);
     
     CUDA_SAFE_CALL(cudaMemcpyToSymbol(params, &gtc_input->global_params.mi, sizeof(int),offsetof(gtc_global_params_t,mi) ,cudaMemcpyHostToDevice));
     
     gpu_kernel_input->gpu_timing.device_shift_time += gpu_timer_measure_end(gpu_kernel_input);

}





