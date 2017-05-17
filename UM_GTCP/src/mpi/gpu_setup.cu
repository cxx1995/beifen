#define GPU_KERNEL
#include "bench_gtc.h"
#include <cutil.h>

__device__ __managed__ gtc_global_params_t params __align__ (16);
//__device__ __constant__ real temp[MAX_MPSI] __align__ (16);
__device__ __managed__ real temp[MAX_MPSI] __align__ (16);
__device__ __constant__ real dtemp[MAX_MPSI] __align__ (16);
//__device__ __constant__ real rtemi[MAX_MPSI] __align__ (16);
__device__ __managed__ real rtemi[MAX_MPSI] __align__ (16);
//__device__ __constant__ real pfluxpsi[MFLUX+1] __align__ (16);
//__device__ __constant__ real qtinv[MAX_MPSI] __align__ (16);
__device__ __managed__ real qtinv[MAX_MPSI] __align__ (16);

//__device__ __constant__ real delt[MAX_MPSI] __align__ (16);
__device__ __managed__ real delt[MAX_MPSI] __align__ (16);
//__device__ __constant__ int igrid[MAX_MPSI] __align__ (16);
__device__ __managed__ int igrid[MAX_MPSI] __align__ (16);
//__device__ __constant__ int mtheta[MAX_MPSI] __align__ (16);
__device__ __managed__ int mtheta[MAX_MPSI] __align__ (16);
//__device__ __constant__ int max_shift_mi __align__ (16);
__device__ __managed__ int max_shift_mi __align__ (16);
__device__ __managed__ gtc_radial_decomp_t radial_decomp __align__ (16);

#if CUDA_TIMER
void gpu_timer_start(gpu_kernel_args_t* gpu_kernel_input) 
{   	
  CUDA_SAFE_CALL(cudaEventCreate(&gpu_kernel_input->gpu_timing.start));
  CUDA_SAFE_CALL(cudaEventCreate(&gpu_kernel_input->gpu_timing.stop));
  CUDA_SAFE_CALL(cudaEventRecord(gpu_kernel_input->gpu_timing.start,0));
}

float gpu_timer_measure(gpu_kernel_args_t* gpu_kernel_input) 
{
  float elapsedTime;
  CUDA_SAFE_CALL( cudaEventRecord( gpu_kernel_input->gpu_timing.stop, 0 ) );
  CUDA_SAFE_CALL( cudaEventSynchronize(gpu_kernel_input->gpu_timing.stop) );
  CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,gpu_kernel_input->gpu_timing.start,gpu_kernel_input->gpu_timing.stop));
  
  cudaEvent_t temp = gpu_kernel_input->gpu_timing.start;
  gpu_kernel_input->gpu_timing.start = gpu_kernel_input->gpu_timing.stop;
  gpu_kernel_input->gpu_timing.stop = temp;
  return elapsedTime/1000; 
}

float gpu_timer_measure_end(gpu_kernel_args_t* gpu_kernel_input) 
{
  float elapsedTime;
  CUDA_SAFE_CALL( cudaEventRecord( gpu_kernel_input->gpu_timing.stop, 0 ) );
  CUDA_SAFE_CALL( cudaEventSynchronize(gpu_kernel_input->gpu_timing.stop) );
  CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,gpu_kernel_input->gpu_timing.start,gpu_kernel_input->gpu_timing.stop));
  CUDA_SAFE_CALL(cudaEventDestroy(gpu_kernel_input->gpu_timing.start)); 
  CUDA_SAFE_CALL(cudaEventDestroy(gpu_kernel_input->gpu_timing.stop));
  return elapsedTime/1000; 
}
#else

void gpu_timer_start(gpu_kernel_args_t* gpu_kernel_input) 
{
  
}

float gpu_timer_measure(gpu_kernel_args_t* gpu_kernel_input) 
{
  return 0;
}

float gpu_timer_measure_end(gpu_kernel_args_t* gpu_kernel_input) 
{
  return 0;
}

#endif

#if !USE_SM35_TEXTURE
#if SINGLE_PRECISION
texture<float, 1, cudaReadModeElementType> evectorTexRef;
#else
texture<int2, 1, cudaReadModeElementType> evectorTexRef;
#endif
#endif

static void allocate_device_data(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_args)
{
  gtc_global_params_t *h_params   = &(gtc_input->global_params);
  gtc_radial_decomp_t *h_radial_decomp = &(gtc_input->radial_decomp);

  int d_mimax = gpu_kernel_args->d_mimax;
  //int nloc_over = (h_radial_decomp->nloc_over/GPU_ALIGNEMENT + 1)*GPU_ALIGNEMENT;
  int nloc_over = h_radial_decomp->nloc_over;
  int d_extra_mimax  = gpu_kernel_args->d_extra_mimax;
  int nloc_over_cluster = gpu_kernel_args->d_nloc_over_cluster;

  int mzeta=  h_params->mzeta;
  
  gtc_field_data_t        *d_grid = &(gpu_kernel_args->d_grid);
  gtc_particle_data_t     *d_zion  = &(gpu_kernel_args->d_zion);
  gtc_diagnosis_data_t    *d_diagnosis = &(gpu_kernel_args->d_diagnosis);
  gtc_field_data_t *h_grid = &(gtc_input->field_data);
  
  // allocate particle array //
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpu_kernel_args->ptr_d_zion), sizeof(gtc_particle_data_t)));
CUDA_SAFE_CALL(cudaMallocManaged(&(gpu_kernel_args->ptr_d_zion), sizeof(gtc_particle_data_t)));
//CUDA_SAFE_CALL(cudaMalloc((void**)&(d_zion->z0), 12*d_mimax*sizeof(real)));
CUDA_SAFE_CALL(cudaMallocManaged(&(d_zion->z0), 12*d_mimax*sizeof(real)));

  d_zion->z1 = d_zion->z0 + d_mimax;
  d_zion->z2 = d_zion->z0 + 2*d_mimax;
  d_zion->z3 = d_zion->z0 + 3*d_mimax;
  d_zion->z4 = d_zion->z0 + 4*d_mimax;
  d_zion->z5 = d_zion->z0 + 5*d_mimax;
  d_zion->z00 = d_zion->z0 + 6*d_mimax;
  d_zion->z01 = d_zion->z0 + 7*d_mimax;
  d_zion->z02 = d_zion->z0 + 8*d_mimax;
  d_zion->z03 = d_zion->z0 + 9*d_mimax;
  d_zion->z04 = d_zion->z0 + 10*d_mimax;
  d_zion->z05 = d_zion->z0 + 11*d_mimax;

#if SYNERGESTIC_SORT_SHIFT 
  // allocate auxillary particle array //
  gtc_particle_data_t *d_auxs_zion = &(gpu_kernel_args->d_auxs_zion);
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpu_kernel_args->ptr_d_auxs_zion), sizeof(gtc_particle_data_t)));
CUDA_SAFE_CALL(cudaMallocManaged(&(gpu_kernel_args->ptr_d_auxs_zion), sizeof(gtc_particle_data_t)));


  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_auxs_zion->z0), 12* (d_mimax)*sizeof(real)));
  d_auxs_zion->z1 = d_auxs_zion->z0 + d_mimax;
  d_auxs_zion->z2 = d_auxs_zion->z0 + 2*d_mimax;
  d_auxs_zion->z3 = d_auxs_zion->z0 + 3*d_mimax;
  d_auxs_zion->z4 = d_auxs_zion->z0 + 4*d_mimax;
  d_auxs_zion->z5 = d_auxs_zion->z0 + 5*d_mimax;
  d_auxs_zion->z00 = d_auxs_zion->z0 + 6*d_mimax;
  d_auxs_zion->z01 = d_auxs_zion->z0 + 7*d_mimax;
  d_auxs_zion->z02 = d_auxs_zion->z0 + 8*d_mimax;
  d_auxs_zion->z03 = d_auxs_zion->z0 + 9*d_mimax;
  d_auxs_zion->z04 = d_auxs_zion->z0 + 10*d_mimax;
  d_auxs_zion->z05 = d_auxs_zion->z0 + 11*d_mimax;
#endif
  
  // allocate intermediate particle data //
#if FOURPOINT
  gtc_aux_particle_point_t *d_aux_point = &(gpu_kernel_args->d_aux_point);
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpu_kernel_args->ptr_d_aux_point), sizeof(gtc_aux_particle_point_t)));
CUDA_SAFE_CALL(cudaMallocManaged(&(gpu_kernel_args->ptr_d_aux_point), sizeof(gtc_aux_particle_point_t)));

  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_aux_point->point_index), (4*d_mimax)*sizeof(int)));

//modified by cxx 2017/5/8 11:15
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_aux_point->point_index_count), nloc_over_cluster*sizeof(my_int)));
CUDA_SAFE_CALL(cudaMallocManaged(&(d_aux_point->point_index_count), nloc_over_cluster*sizeof(my_int)));


  // reserve extra space such that the buffer attached to each bucket in bucket sort is large enough
  // the required extra buffer is reduced as we increase mcellperthread(each thread works on several cells)
  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_aux_point->point_vect), 16*d_extra_mimax*sizeof(my_real)));
  //       d_aux_point->point_weight = d_aux_point->point_vect + 12*d_extra_mimax;

#if PARTICLE_BINNING || SYNERGESTIC_SORT_SHIFT
  gtc_sort_particle_t *d_sort =  &(gpu_kernel_args->d_sort);
  d_sort->d_sort_key = d_aux_point->point_index;
  d_sort->d_value = d_sort->d_sort_key + d_mimax;
#if !SYNERGESTIC_SORT_SHIFT
  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_sort->d_aux_zion05), d_mimax*sizeof(real)));
#endif
#endif
  
#endif
  
  // allocate aux partilce data //
#if !ONTHEFLY_PUSHAUX
  gtc_aux_particle_data_t *d_aux_zion =  &(gpu_kernel_args->d_aux_zion);
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpu_kernel_args->ptr_d_aux_zion), sizeof(gtc_aux_particle_data_t)));
CUDA_SAFE_CALL(cudaMallocManaged(&(gpu_kernel_args->ptr_d_aux_zion), sizeof(gtc_aux_particle_data_t)));

  
  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_aux_zion->kzion), (d_mimax)*sizeof(int)));
//modified by cxx 2017/5/8 11:28
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_aux_zion->jtion0), (8*d_mimax)*sizeof(int)));
CUDA_SAFE_CALL(cudaMallocManaged(&(d_aux_zion->jtion0), (8*d_mimax)*sizeof(int)));


  d_aux_zion->jtion1 = d_aux_zion->jtion0 + 4*(d_mimax);

//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_aux_zion->wzion), (13*d_mimax)*sizeof(real))); 
CUDA_SAFE_CALL(cudaMallocManaged(&(d_aux_zion->wzion), (13*d_mimax)*sizeof(real))); 

  d_aux_zion->wpion = d_aux_zion->wzion + d_mimax;
  d_aux_zion->wtion0 = d_aux_zion->wzion + 5*d_mimax;
  d_aux_zion->wtion1 = d_aux_zion->wzion + 9*d_mimax;
#if PARTICLE_BINNING || SYNERGESTIC_SORT_SHIFT
  gtc_sort_particle_t	*d_sort =  &(gpu_kernel_args->d_sort);
  d_sort->d_sort_key = d_aux_zion->jtion0;
  d_sort->d_value = d_aux_zion->jtion1; 
  d_sort->d_aux_zion05 = d_aux_zion->wzion;
#endif

#endif
  
  // allocate sort data //
#if PARTICLE_BINNING || SYNERGESTIC_SORT_SHIFT
#if !FOURPOINT
#if ONTHEFLY_PUSHAUX
  gtc_sort_particle_t *d_sort =  &(gpu_kernel_args->d_sort);
  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_sort->d_sort_key), 2*d_mimax*sizeof(int)));
  d_sort->d_value = d_sort->d_sort_key + d_mimax;
  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_sort->d_aux_zion05), d_mimax*sizeof(real)));
#endif 
#endif
#endif
  
  // allocate diagnosis data //
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpu_kernel_args->ptr_d_diagnosis), sizeof(gtc_diagnosis_data_t)));
CUDA_SAFE_CALL(cudaMallocManaged(&(gpu_kernel_args->ptr_d_diagnosis), sizeof(gtc_diagnosis_data_t)));


//modified by cxx 2017/5/8 
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_diagnosis->scalar_data), 16*sizeof(real)));
CUDA_SAFE_CALL(cudaMallocManaged(&(d_diagnosis->scalar_data), 16*sizeof(real)));


//modified by cxx 2017/5/8 
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_diagnosis->flux_data), MFLUX*4*sizeof(real)));
CUDA_SAFE_CALL(cudaMallocManaged(&(d_diagnosis->flux_data), MFLUX*4*sizeof(real)));


  // allocate shifti data //
#if !SYNERGESTIC_SORT_SHIFT
//modified by cxx 2017/5/8 14:09
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpu_kernel_args->ptr_d_zion_shift),12*gpu_kernel_args->d_max_shift_mi*sizeof(real)));
CUDA_SAFE_CALL(cudaMallocManaged(&(gpu_kernel_args->ptr_d_zion_shift),12*gpu_kernel_args->d_max_shift_mi*sizeof(real)));

#endif

  // allocate grid related data //
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(gpu_kernel_args->ptr_d_grid), sizeof(gtc_field_data_t)));
CUDA_SAFE_CALL(cudaMallocManaged(&(gpu_kernel_args->ptr_d_grid), sizeof(gtc_field_data_t)));
#if !GYRO_LOCAL_COMPUTE
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_grid->pgyro), 2*4*nloc_over*sizeof(real)));
CUDA_SAFE_CALL(cudaMallocManaged(&(d_grid->pgyro), 2*4*nloc_over*sizeof(real)));

  d_grid->tgyro = d_grid->pgyro + 4*nloc_over;
#endif

//modified by cxx 2017/5/7
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_grid->evector), nloc_over*3*(mzeta+1)*sizeof(real)));
CUDA_SAFE_CALL(cudaMallocManaged(&(d_grid->evector), nloc_over*3*(mzeta+1)*sizeof(real)));



  //  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_grid->densityi), (mzeta+1)*nloc_over*sizeof(wreal)));
  // These pointer do not co-exist so one commen buffer should be OK.



  d_grid->densityi = d_grid->evector;

//modified by cxx 2017/5/8 11:38
//  CUDA_SAFE_CALL(cudaMalloc((void**)&(d_grid->pfluxpsi), MFLUX*sizeof(real)));
CUDA_SAFE_CALL(cudaMallocManaged(&(d_grid->pfluxpsi), MFLUX*sizeof(real)));
}

extern "C"
void free_gtc_data_on_device(gpu_kernel_args_t* gpu_kernel_input)
{
  gtc_particle_data_t     *d_zion  = &(gpu_kernel_input->d_zion);
  gtc_field_data_t *d_grid = &(gpu_kernel_input->d_grid);
  gtc_diagnosis_data_t *d_diagnosis = &(gpu_kernel_input->d_diagnosis);

  CUDA_SAFE_CALL(cudaFree(gpu_kernel_input->ptr_d_zion));
  CUDA_SAFE_CALL(cudaFree(d_zion->z0));

#if FOURPOINT
  gtc_aux_particle_point_t *d_aux_point = &(gpu_kernel_input->d_aux_point);
  CUDA_SAFE_CALL(cudaFree(gpu_kernel_input->ptr_d_aux_point));
  CUDA_SAFE_CALL(cudaFree(d_aux_point->point_index));
  CUDA_SAFE_CALL(cudaFree(d_aux_point->point_index_count));
  CUDA_SAFE_CALL(cudaFree(d_aux_point->point_vect));
#if PARTICLE_BINNING || SYNERGESTIC_SORT_SHIFT
#if !SYNERGESTIC_SORT_SHIFT
  gtc_sort_particle_t *d_sort =  &(gpu_kernel_input->d_sort);
  CUDA_SAFE_CALL(cudaFree(d_sort->d_aux_zion05));
#endif
#endif
#endif
  
#if SYNERGESTIC_SORT_SHIFT
  gtc_particle_data_t *d_auxs_zion = &(gpu_kernel_input->d_auxs_zion);
  CUDA_SAFE_CALL(cudaFree(gpu_kernel_input->ptr_d_auxs_zion));
  CUDA_SAFE_CALL(cudaFree(d_auxs_zion->z0));
#endif

#if !ONTHEFLY_PUSHAUX
  gtc_aux_particle_data_t *d_aux_zion =  &(gpu_kernel_input->d_aux_zion);
  CUDA_SAFE_CALL(cudaFree(gpu_kernel_input->ptr_d_aux_zion));
  CUDA_SAFE_CALL(cudaFree(d_aux_zion->kzion));
  CUDA_SAFE_CALL(cudaFree(d_aux_zion->jtion0));
  CUDA_SAFE_CALL(cudaFree(d_aux_zion->wzion));
#endif
  
#if PARTICLE_BINNING || SYNERGESTIC_SORT_SHIFT
#if !FOURPOINT
#if ONTHEFLY_PUSHAUX
  gtc_sort_particle_t	*d_sort =  &(gpu_kernel_input->d_sort);
  CUDA_SAFE_CALL(cudaFree(d_sort->d_sort_key));
  CUDA_SAFE_CALL(cudaFree(d_sort->d_aux_zion05));
#endif
#endif
#endif
  
  CUDA_SAFE_CALL(cudaFree(gpu_kernel_input->ptr_d_diagnosis));
  CUDA_SAFE_CALL(cudaFree(d_diagnosis->scalar_data));
  CUDA_SAFE_CALL(cudaFree(d_diagnosis->flux_data));
  
#if !SYNERGESTIC_SORT_SHIFT
  CUDA_SAFE_CALL(cudaFree(gpu_kernel_input->ptr_d_zion_shift));
#endif

  CUDA_SAFE_CALL(cudaFree(gpu_kernel_input->ptr_d_grid));
#if (!GYRO_LOCAL_COMPUTE)
  CUDA_SAFE_CALL(cudaFree(d_grid->pgyro));
#endif
  
  CUDA_SAFE_CALL(cudaFree(d_grid->evector));
  // CUDA_SAFE_CALL(cudaFree(d_grid->densityi));
  CUDA_SAFE_CALL(cudaFree(d_grid->pfluxpsi));
}

#include <ptx_custom.cu>
#include <gpu_charge_kernel.cu>
#if FOURPOINT
#include <gpu_charge_4p_kernel.cu>
#endif

#if SYNERGESTIC_SORT_SHIFT
        #include <gpu_bin_shift.cu>
#else
#if	PARTICLE_BINNING
        #include <gpu_bin_particle_kernel.cu>
#endif
#include <gpu_shift_kernel.cu>
#endif

//#if !FOURPOINT
#include <gpu_push_kernel.cu>
//#else
//#include <gpu_push_4p_kernel.cu>
//#endif

extern "C"
void cpy_gtc_data_to_device(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_args) 
{
  gtc_global_params_t   *h_params = &(gtc_input->global_params);
  gtc_field_data_t      *h_grid = &(gtc_input->field_data);
  gtc_field_data_t      *d_grid = &(gpu_kernel_args->d_grid);
  gtc_particle_data_t   *d_zion = &(gpu_kernel_args->d_zion);
  gtc_particle_data_t   *h_zion = &(gtc_input->particle_data);
  gtc_radial_decomp_t   *h_radial_decomp = &(gtc_input->radial_decomp);

  /* load the values that were previously calculated in the setup function */
  int mpsi = h_params->mpsi; 	
  int mi = h_params->mi;
  int nloc_over = h_radial_decomp->nloc_over;
  int d_mimax = mi;
  
  /* 2. copy zion and zion0 */
//modified by cxx 2017/5/8 
//  CUDA_SAFE_CALL(cudaMemcpy((void *)gpu_kernel_args->ptr_d_zion, (void *)d_zion, sizeof(gtc_particle_data_t) , cudaMemcpyHostToDevice));
memcpy(gpu_kernel_args->ptr_d_zion,d_zion, sizeof(gtc_particle_data_t));

/*  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z0, h_zion->z0, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z1, h_zion->z1, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z2, h_zion->z2, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z3, h_zion->z3, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z4, h_zion->z4, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z5, h_zion->z5, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z00, h_zion->z0, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z01, h_zion->z1, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z02, h_zion->z2, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z03, h_zion->z3, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z04, h_zion->z4, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_zion->z05, h_zion->z05, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
*/

  memcpy(d_zion->z0, h_zion->z0, (d_mimax)*sizeof(real));
  memcpy(d_zion->z1, h_zion->z1, (d_mimax)*sizeof(real));
  memcpy(d_zion->z2, h_zion->z2, (d_mimax)*sizeof(real));
  memcpy(d_zion->z3, h_zion->z3, (d_mimax)*sizeof(real));
  memcpy(d_zion->z4, h_zion->z4, (d_mimax)*sizeof(real));
  memcpy(d_zion->z5, h_zion->z5, (d_mimax)*sizeof(real));
  memcpy(d_zion->z00, h_zion->z0, (d_mimax)*sizeof(real));
  memcpy(d_zion->z01, h_zion->z1, (d_mimax)*sizeof(real));
  memcpy(d_zion->z02, h_zion->z2, (d_mimax)*sizeof(real));
  memcpy(d_zion->z03, h_zion->z3, (d_mimax)*sizeof(real));
  memcpy(d_zion->z04, h_zion->z4, (d_mimax)*sizeof(real));
  memcpy(d_zion->z05, h_zion->z05, (d_mimax)*sizeof(real));

#if SYNERGESTIC_SORT_SHIFT
  gtc_particle_data_t *d_auxs_zion = &(gpu_kernel_args->d_auxs_zion);
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMemcpy((void *)gpu_kernel_args->ptr_d_auxs_zion, (void *)d_auxs_zion, sizeof(gtc_particle_data_t), cudaMemcpyHostToDevice));
memcpy(gpu_kernel_args->ptr_d_auxs_zion, d_auxs_zion, sizeof(gtc_particle_data_t));

#endif

#if FOURPOINT
  gtc_aux_particle_point_t *d_aux_point = &(gpu_kernel_args->d_aux_point);
//modifide by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMemcpy((void*)gpu_kernel_args->ptr_d_aux_point, (void *)d_aux_point, sizeof(gtc_aux_particle_point_t), cudaMemcpyHostToDevice));
memcpy(gpu_kernel_args->ptr_d_aux_point, d_aux_point, sizeof(gtc_aux_particle_point_t));
#endif

  /* 4. copy aux_zion */
#if !ONTHEFLY_PUSHAUX
  gtc_aux_particle_data_t *h_aux_zion = &(gtc_input->aux_particle_data);
  gtc_aux_particle_data_t *d_aux_zion =  &(gpu_kernel_args->d_aux_zion);

//modified by cxx 2017/5/8  
//  CUDA_SAFE_CALL(cudaMemcpy((void *)d_aux_zion->jtion0, h_aux_zion->jtion0, 4*(d_mimax)*sizeof(int) , cudaMemcpyHostToDevice));
memcpy(d_aux_zion->jtion0, h_aux_zion->jtion0, 4*(d_mimax)*sizeof(int));
//  CUDA_SAFE_CALL(cudaMemcpy((void *)d_aux_zion->jtion1, h_aux_zion->jtion1, 4*(d_mimax)*sizeof(int) , cudaMemcpyHostToDevice));
memcpy(d_aux_zion->jtion1, h_aux_zion->jtion1, 4*(d_mimax)*sizeof(int));

//modified by cxx 2017/5/8
/*  CUDA_SAFE_CALL(cudaMemcpy((void *)d_aux_zion->wzion, h_aux_zion->wzion, (d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_aux_zion->wpion, h_aux_zion->wpion, (4*d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_aux_zion->wtion0, h_aux_zion->wtion0, (4*d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy((void *)d_aux_zion->wtion1, h_aux_zion->wtion1, (4*d_mimax)*sizeof(real) , cudaMemcpyHostToDevice));
*/
memcpy(d_aux_zion->wzion, h_aux_zion->wzion, (d_mimax)*sizeof(real));
memcpy(d_aux_zion->wpion, h_aux_zion->wpion, (4*d_mimax)*sizeof(real));
memcpy(d_aux_zion->wtion0, h_aux_zion->wtion0, (4*d_mimax)*sizeof(real));
memcpy(d_aux_zion->wtion1, h_aux_zion->wtion1, (4*d_mimax)*sizeof(real));

//modified by cxx 2017/5/8  
//  CUDA_SAFE_CALL(cudaMemcpy((void *)gpu_kernel_args->ptr_d_aux_zion, (void *)d_aux_zion, sizeof(gtc_aux_particle_data_t) , cudaMemcpyHostToDevice));
memcpy(gpu_kernel_args->ptr_d_aux_zion, d_aux_zion, sizeof(gtc_aux_particle_data_t));
#endif

  gtc_diagnosis_data_t *d_diagnosis = &(gpu_kernel_args->d_diagnosis);
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMemcpy((void*)gpu_kernel_args->ptr_d_diagnosis, (void *)d_diagnosis, sizeof(gtc_diagnosis_data_t), cudaMemcpyHostToDevice));
memcpy(gpu_kernel_args->ptr_d_diagnosis, d_diagnosis, sizeof(gtc_diagnosis_data_t));  
  /* 1. copy params */
  int tmp = h_params->mimax;
  h_params->mimax = gpu_kernel_args->d_mimax;
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMemcpyToSymbol(params,h_params, sizeof(gtc_global_params_t),0,cudaMemcpyHostToDevice));
//printf("h_params: ")
params = *h_params;

  h_params->mimax = tmp;
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMemcpyToSymbol(radial_decomp, h_radial_decomp, sizeof(gtc_radial_decomp_t), 0, cudaMemcpyHostToDevice));
radial_decomp = *h_radial_decomp;

  
  real vthi = h_params->gyroradius*fabs(h_params->qion)/h_params->aion;
  /* primary ion marker temperature and parallel flow velocity */
  int i;
  for (i=0; i<mpsi+1; i++) {
    h_grid->temp[i]  = 1.0;
    h_grid->dtemp[i] = 0.0;
    h_grid->temp[i]  = 1.0/(h_grid->temp[i] * h_grid->rtemi[i] * h_params->aion * vthi * vthi); 
    /*inverse local temperature */
  }
  
  // pfluxpsi is updated every idiag==0. With diagnosis, pfluxpsi is put in GPU global memory (BW Sep 12,2012) 
  // CUDA_SAFE_CALL(cudaMemcpyToSymbol("pfluxpsi", h_grid->pfluxpsi, (MFLUX)*sizeof(real),0, cudaMemcpyHostToDevice));
//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMemcpyToSymbol(temp, (h_grid->temp), (mpsi+1)*sizeof(real),0,cudaMemcpyHostToDevice));
memcpy(temp, (h_grid->temp), (mpsi+1)*sizeof(real));

//modified by cxx 2017/5/8
//  CUDA_SAFE_CALL(cudaMemcpyToSymbol(rtemi, h_grid->rtemi, (mpsi+1)*sizeof(real),0,cudaMemcpyHostToDevice));
memcpy(rtemi, h_grid->rtemi, (mpsi+1)*sizeof(real));

/*  CUDA_SAFE_CALL(cudaMemcpyToSymbol(igrid, (h_grid->igrid), (mpsi+1)*sizeof(int),0,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(delt, (h_grid->delt), (mpsi+1)*sizeof(real),0,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(qtinv, (h_grid->qtinv), (mpsi+1)*sizeof(real),0,cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(mtheta, (h_grid->mtheta), (mpsi+1)*sizeof(int),0,cudaMemcpyHostToDevice));
*/
memcpy(igrid, (h_grid->igrid), (mpsi+1)*sizeof(int));
memcpy(delt, (h_grid->delt), (mpsi+1)*sizeof(real));
memcpy(qtinv, (h_grid->qtinv), (mpsi+1)*sizeof(real));
memcpy(mtheta, (h_grid->mtheta), (mpsi+1)*sizeof(int));


//  CUDA_SAFE_CALL(cudaMemcpyToSymbol(max_shift_mi, &(gpu_kernel_args->d_max_shift_mi), sizeof(int),0,cudaMemcpyHostToDevice));
max_shift_mi = gpu_kernel_args->d_max_shift_mi;


//modified by cxx 2017/5/8
/*  CUDA_SAFE_CALL(cudaMemcpy((void *)gpu_kernel_args->ptr_d_grid,d_grid, sizeof(gtc_field_data_t), cudaMemcpyHostToDevice));
#if !GYRO_LOCAL_COMPUTE
  CUDA_SAFE_CALL(cudaMemcpy((void*)d_grid->pgyro, h_grid->pgyro, 4*nloc_over*sizeof(real), cudaMemcpyHostToDevice)); 
  CUDA_SAFE_CALL(cudaMemcpy((void*)d_grid->tgyro, h_grid->tgyro, 4*nloc_over*sizeof(real), cudaMemcpyHostToDevice)); 
*/
memcpy(gpu_kernel_args->ptr_d_grid,d_grid, sizeof(gtc_field_data_t));
#if !GYRO_LOCAL_COMPUTE
  memcpy(d_grid->pgyro, h_grid->pgyro, 4*nloc_over*sizeof(real)); 
  memcpy(d_grid->tgyro, h_grid->tgyro, 4*nloc_over*sizeof(real)); 
#endif

#if  USE_TEXTURE
#if !USE_SM35_TEXTURE
  int mzeta=  h_params->mzeta;
  CUDA_SAFE_CALL(cudaBindTexture(0, evectorTexRef, (void *)(d_grid->evector),  nloc_over*3*(mzeta+1)*sizeof(real)));
#endif
#endif
//  CUDA_SAFE_CALL(cudaMemcpy((void*)d_grid->pfluxpsi, h_grid->pfluxpsi, MFLUX*sizeof(real), cudaMemcpyHostToDevice));
memcpy(d_grid->pfluxpsi, h_grid->pfluxpsi, MFLUX*sizeof(real));
  
}

extern "C"
void gpu_atexit(void)
{
	/*this will cause the program to flush */
//	CUDA_SAFE_CALL(cudaThreadSynchronize ());

}

extern "C"
void gpu_setup(gtc_bench_data_t* gtc_input, gpu_kernel_args_t* gpu_kernel_args) 
{
  gtc_global_params_t   *h_params   = &(gtc_input->global_params);
  gpu_kernel_args->d_mimax = h_params->mi + 100*int(ceil(sqrt(h_params->mi)));
  gpu_kernel_args->d_mimax  = (gpu_kernel_args->d_mimax/GPU_ALIGNEMENT + 1) * GPU_ALIGNEMENT;

  gpu_kernel_args->d_max_shift_mi  = gpu_kernel_args->d_mimax / (100/MAXIMUM_SHIFT_PERCENT);
  gpu_kernel_args->d_nloc_over_cluster = (gtc_input->radial_decomp.nloc_over+CLUSTER_SIZE-1)/CLUSTER_SIZE;
  gpu_kernel_args->d_nloc_over_cluster = (gpu_kernel_args->d_nloc_over_cluster/GPU_ALIGNEMENT + 1) * GPU_ALIGNEMENT;
  gpu_kernel_args->d_extra_mimax = (int)(EXTRA_BUFFER*gtc_input->radial_decomp.nloc_over*gtc_input->global_params.micell);
  gpu_kernel_args->d_extra_mimax  = (gpu_kernel_args->d_extra_mimax/GPU_ALIGNEMENT + 1) * GPU_ALIGNEMENT;

  int device_count = 0;
  int pe = gtc_input->parallel_decomp.mype; 

  CUDA_SAFE_CALL(cudaGetDeviceCount(&device_count));
  int gpu_device = pe%device_count;
  CUDA_SAFE_CALL(cudaSetDevice(gpu_device));
  CUDA_SAFE_CALL(cudaGetDevice(&gpu_device));
  cudaGetDeviceProperties(&gpu_kernel_args->deviceProp, gpu_device);
  int nblocks=1;
  if (pe==0) printf("multiProcessorCount=%d\n", gpu_kernel_args->deviceProp.multiProcessorCount);
  while(nblocks < gpu_kernel_args->deviceProp.multiProcessorCount)
    nblocks *= 2;
  /* Note we give a 4.5x number of multiprocessors compared with the 6 blocks per multiprocessor that is typically suggested */
  gpu_kernel_args->nblocks = 2*nblocks;
  
  if(pe == 0) {
    char *name[] = {"no","yes"};
    fprintf(stderr, "PE %d: Running on cuda device %d, total devices %d, Warpsize: %d, Min block count: %d\n",pe,gpu_device,device_count, gpu_kernel_args->deviceProp.warpSize, gpu_kernel_args->nblocks);
    fprintf(stderr, "GPU Run Configuration\n========================\n"
	    "Prefer L1: %s\n"
	    "Use of PTX intrinsics: %s\n"
	    "Particle binning: %s (period %d)\n"
	    "On the fly aux computation: %s\n"
	    "Gyro local compute: %s\n"
	    "Use cooperative threading for charge deposition: %s\n"
            "Use four-point algorithm: %s\n"
	    "Use synergestic_sort_shift: %s\n=======================\n",
	    name[PREFER_L1], name[PTX_STREAM_INTRINSICS], name[PARTICLE_BINNING],RADIAL_THETA_BIN_PERIOD,name[ONTHEFLY_PUSHAUX],name[GYRO_LOCAL_COMPUTE],name[COOPERATIVE_THREADING], name[FOURPOINT], name[SYNERGESTIC_SORT_SHIFT]);
  }

  gpu_kernel_args->nthreads = THREAD_BLOCK;
  gpu_kernel_args->charge_mi_per_thread = 64;
  allocate_device_data(gtc_input,gpu_kernel_args);
  memset(&gpu_kernel_args->gpu_timing,0,sizeof(gpu_timing_t));

#if !SYNERGESTIC_SORT_SHIFT
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_shifti_extract_kernel,cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_shifti_radial_extract_kernel,cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_shifti_append_kernel,cudaFuncCachePreferShared));
#endif

#if SHARED_8BYTE_BANK
  CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#endif

#if FOURPOINT
  //        CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_push_gyro_interpolation, cudaFuncCachePreferL1));
#endif
  
#if	PREFER_L1
#if !FOURPOINT
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_charge_cooperative, cudaFuncCachePreferL1));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_pushi_kernel, cudaFuncCachePreferL1));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_charge_multi, cudaFuncCachePreferL1));
#else
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_charge_initialization_cooperative, cudaFuncCachePreferL1));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_charge_interpolation, cudaFuncCachePreferL1));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_pushi_kernel, cudaFuncCachePreferL1));
  //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_push_point_interpolation, cudaFuncCachePreferL1));
  //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_push_gyro_interpolation, cudaFuncCachePreferL1));
#endif
#else
#if !FOURPOINT
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_pushi_kernel, cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_charge_cooperative, cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_charge_multi, cudaFuncCachePreferShared));
#else
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_charge_initialization_cooperative, cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_charge_interpolation, cudaFuncCachePreferShared));
  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_pushi_kernel, cudaFuncCachePreferShared));
  //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_push_point_interpolation, cudaFuncCachePreferShared));
  //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(gpu_push_gyro_interpolation, cudaFuncCachePreferShared));
#endif
#endif
  atexit (gpu_atexit); 
}

void print_gpu_timing(gpu_kernel_args_t* gpu_kernel_args)
{
  fprintf(stderr, "\nGPU PCIe time summary:\n"
	  "Charge			%9.6f s\n"
				"Push			%9.6f s\n"
	  "Shift			%9.6f s\n"
	  "\nExecution time summary on GPU device:\n"
	  "Charge			%9.6f s\n"
	  "Push			%9.6f s\n"
	  "Shift			%9.6f s\n"
	  "Particle sort	%9.6f s\n"
				"Particle bin	%9.6f s\n"
	  "charge reset   %9.6f s\n"
	  "charge initialization %9.6f s\n"
	  "charge interpolation  %9.6f s\n"
	  "push point interpolation %9.6f s\n"
	  "push gyro interpolation  %9.6f s\n",
	  gpu_kernel_args->gpu_timing.memtransfer_charge_time,
	  gpu_kernel_args->gpu_timing.memtransfer_push_time,
	  gpu_kernel_args->gpu_timing.memtransfer_shift_time,
	  gpu_kernel_args->gpu_timing.device_charge_time,
	  gpu_kernel_args->gpu_timing.device_push_time,
	  gpu_kernel_args->gpu_timing.device_shift_time,
	  gpu_kernel_args->gpu_timing.device_particle_sort_time,
	  gpu_kernel_args->gpu_timing.device_particle_bin_time,
	  gpu_kernel_args->gpu_timing.memreset_charge_time,
	  gpu_kernel_args->gpu_timing.initialization_charge_time,
	  gpu_kernel_args->gpu_timing.interpolation_charge_time,
	  gpu_kernel_args->gpu_timing.interpolation_push_point_time,
	  gpu_kernel_args->gpu_timing.interpolation_push_gyro_time);
}


