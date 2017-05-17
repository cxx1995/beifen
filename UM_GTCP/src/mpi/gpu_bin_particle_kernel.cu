#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <cstdlib>

//using namespace thrust;

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
      seq[m]=m;
      
      real psi     = z0[m];
      real theta   = z1[m];
      
#if SQRT_PRECOMPUTED
      r        = psi;
#else
      r        = sqrt(2.0*psi);
#endif
      
      int iptmp    = (int) ((r-a0)*delr+0.5);
      int ip = d_abs_min_int(mpsi, iptmp);
#if THETA_BIN_ORG
      //        int jttmp    = (int) (theta*pi2_inv*delt[ip]+0.5);
      //        psi_theta[m] = d_abs_min_int(mtheta[ip], jttmp);
      psi_theta[m] = ip;
#else
      psi_theta[m] = theta*100.0 + ip*700;
#endif
    }
}

__global__ static void pre_permute_particles_zion(gtc_particle_data_t *d_zion, real *aux_zion05, int *permutation) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int gid = tid + bid*nthreads;
  int mi = params.mi;
  if (gid < mi){
    int pos = permutation[gid];
    aux_zion05[gid] = (d_zion->z05)[pos];
  }
  
}

__global__ static void post_permute_particles_zion(gtc_particle_data_t *d_zion, real* aux_zion05) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int gid = tid + bid*nthreads;
  int mi = params.mi;
  if (gid < mi){
    (d_zion->z05)[gid]=aux_zion05[gid];
  }
  
}

__global__ static void permute_particles_zion(gtc_particle_data_t *d_zion, int *permutation) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int gid = tid + bid*nthreads;
  int mi = params.mi;
  
  if (gid < mi){
    int pos = permutation[gid];
    (d_zion->z00)[gid] = (d_zion->z0)[pos];
    (d_zion->z01)[gid] = (d_zion->z1)[pos];
    (d_zion->z02)[gid] = (d_zion->z2)[pos];
    (d_zion->z03)[gid] = (d_zion->z3)[pos];
    (d_zion->z04)[gid] = (d_zion->z4)[pos];
    (d_zion->z05)[gid] = (d_zion->z5)[pos];
  }
}


__global__ static void update_particle_zion(gtc_particle_data_t *d_zion) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nthreads = blockDim.x;
  const int gid = tid + bid*nthreads;
  int mi = params.mi;
  
  if (gid < mi){
    (d_zion->z0)[gid] = (d_zion->z00)[gid];
    (d_zion->z1)[gid] = (d_zion->z01)[gid];
    (d_zion->z2)[gid] = (d_zion->z02)[gid];
    (d_zion->z3)[gid] = (d_zion->z03)[gid];
    (d_zion->z4)[gid] = (d_zion->z04)[gid];
    (d_zion->z5)[gid] = (d_zion->z05)[gid];
  }
    
}

/* Assumption is IRK==2 so zion0 and zion1 should be identical */
/* zion0-zion4 are identical with zion00-zion04, but zion5 is different from zion05, (bwang June 12) */
void call_gpu_bin_particles_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int shift_direction)
{        
  gpu_timer_start(gpu_kernel_input);
  int mi	= gtc_input->global_params.mi;
  int nt = 512;
  int nb = (mi + nt - 1)/nt;
  gtc_sort_particle_t	*d_sort =  &(gpu_kernel_input->d_sort);
  calculate_sort_keypair<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, d_sort->d_value, d_sort->d_sort_key);
  thrust::device_ptr<int> d_key(d_sort->d_sort_key);
  thrust::device_ptr<int> d_value(d_sort->d_value);
  thrust::sort_by_key(d_key, d_key + mi, d_value);
  gpu_kernel_input->gpu_timing.device_particle_sort_time += gpu_timer_measure(gpu_kernel_input);
  
  pre_permute_particles_zion<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, d_sort->d_aux_zion05, d_sort->d_value);
  gpu_kernel_input->gpu_timing.device_particle_sort_time += gpu_timer_measure(gpu_kernel_input);
  permute_particles_zion<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, d_sort->d_value);
  gpu_kernel_input->gpu_timing.device_particle_sort_time += gpu_timer_measure(gpu_kernel_input);
  update_particle_zion<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion);
  gpu_kernel_input->gpu_timing.device_particle_sort_time += gpu_timer_measure(gpu_kernel_input);
  post_permute_particles_zion<<<nb, nt>>>(gpu_kernel_input->ptr_d_zion, d_sort->d_aux_zion05);

#if _DEBUG_GPU
  thrust::host_vector<int> h_key(mi);
  thrust::host_vector<int> h_value(mi);
  thrust::copy(d_key, d_key+mi, h_key.begin());
  thrust::copy(d_value, d_value+mi, h_value.begin());
  
  for(int i=0;i<100;i++)
    printf("key %d: value %d\n",h_key[i],h_value[i]);
#endif

  gpu_kernel_input->gpu_timing.device_particle_bin_time += gpu_timer_measure_end(gpu_kernel_input);
	
}



