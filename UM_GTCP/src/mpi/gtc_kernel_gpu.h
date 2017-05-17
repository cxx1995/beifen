#ifndef _GTC_KERNEL_BENCH_GPU_
#define _GTC_KERNEL_BENCH_GPU_

#include <stdlib.h>
#include <stdio.h>
#include "bench_gtc.h"

#include <driver_types.h>

#ifdef __cplusplus
 extern "C" {
#endif

#define GPU_ALIGNEMENT 64
#define GPU_PADDING 16
#define THREAD_BLOCK 64

#define MAX_MULIPROCESSOR 32
#define SM_20
#define CHUNK_BLOCK 512
#define CHUNK_LIMIT (2*1024*1024)
#define USE_TEXTURE 1
#if USE_TEXTURE
#define USE_SM35_TEXTURE  0
#endif

#define PREFER_L1 0
#define SHARED_8BYTE_BANK       1

#define TRACK_SHIFTED_PARTICLES 0
#define CUDA_TIMER 1

#define GPU_DEBUG               0

#define OPTIMIZE_ACCESS 1
#define USE_CONSTANTMEM 1
#define USE_FAST_MATH 1
#define REMOVE_CONTROL_FLOW 1
#define PTX_STREAM_INTRINSICS 1

#define PARTICLE_BINNING 	1
#define RADIAL_THETA_BIN_PERIOD 10

#define MAX_INT 0x7fffffff

#if GPU_ACCEL
#define ONTHEFLY_PUSHAUX    1
#define GYRO_LOCAL_COMPUTE  0
#endif

#define FOURPOINT    0
/* number of cells processed per thread */ 
#define CLUSTER_SIZE 4
/* extra memory for storing four points in bucket sort */
/* the required extra memory is reduced by increasing CLUSTER_SIZE */
/* eg. CLUSTER_SIZE 2, EXTRA_BUFFER 1.85 */
/*     CLUSTER_SIZE 4, EXTRA_BUFFER 1.75 */    
#define EXTRA_BUFFER 3.0

#if FOURPOINT
#if GPU_ACCEL
#define ONTHEFLY_PUSHAUX    1
#define GYRO_LOCAL_COMPUTE  0
#endif

#endif

#define PERMUTE_ZION_R    0
#define AUX_STORE1        1
#define AUX_STORE2        1

#define MIN_MEM_USAGE     1

#define COOPERATIVE_THREADING    	1

#define SYNERGESTIC_SORT_SHIFT          1

#define THETA_BIN_ORG    	        1

#define USE_AUX				1
#if USE_AUX
#else
#define AUX_STORE1        	0
#define AUX_STORE2        	0
#endif 

#define MZETA_MAX 4
#define MFLUX 5
#define MAX_MPSI  721
#define MAXIMUM_SHIFT_PERCENT 10
   
   typedef struct {
     double memtransfer_charge_time;
     double memtransfer_push_time;
     double memtransfer_shift_time;
     double memreset_charge_time;
     double initialization_charge_time;
     double interpolation_charge_time;
     double interpolation_push_point_time;
     double interpolation_push_gyro_time;
     double device_charge_time;
     double device_particle_sort_time;
     double device_particle_bin_time;
     double device_push_time;
     double device_shift_time;
     cudaEvent_t start, stop;
   } gpu_timing_t;   
   
   typedef struct {
     int *d_sort_key;
     int *d_value;
     real *d_aux_zion05;
   } gtc_sort_particle_t;
   
   typedef struct {
     struct cudaDeviceProp deviceProp;
     int nthreads;
     int d_mimax;
     int d_max_shift_mi;
     int d_extra_mimax;
     int d_nloc_over_cluster;
     int nblocks;
     int charge_mi_per_thread;
     int irk;
     int istep;
     int idiag;
     gpu_timing_t gpu_timing;
     
     gtc_particle_data_t      d_zion;
     gtc_particle_data_t      d_auxs_zion;
     gtc_aux_particle_data_t  d_aux_zion;
     gtc_aux_particle_point_t d_aux_point;
     gtc_field_data_t         d_grid;
     gtc_sort_particle_t      d_sort;
     gtc_diagnosis_data_t     d_diagnosis;
     
     gtc_particle_data_t      *ptr_d_zion;
     gtc_particle_data_t      *ptr_d_auxs_zion;
     gtc_aux_particle_data_t  *ptr_d_aux_zion;
     gtc_aux_particle_point_t *ptr_d_aux_point;
     real                     *ptr_d_zion_shift;
     gtc_field_data_t  	      *ptr_d_grid;
     gtc_diagnosis_data_t     *ptr_d_diagnosis;
   } gpu_kernel_args_t;
      
#define RNG_SEED 232323
   
   typedef struct {
     int key;
     real w1;
     real w2;
     real m1;
     real m2;
   } cd_update_t;
   
   void gpu_setup(gtc_bench_data_t* gtc_input, gpu_kernel_args_t* gpu_kernel_args);
   void cpy_gtc_data_to_device(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_args); 
   void free_gtc_data_on_device(gpu_kernel_args_t* gpu_kernel_input);

   void gpu_charge_init(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input);
   int diagnosis(gtc_bench_data_t*);   

   int gpu_chargei(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input);
   int gpu_shifti_toroidal(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input);
   int gpu_shifti_radial(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input);
   int gpu_pushi(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input);
   int gpu_bin_particles(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int shift_direction);
   
   void call_gpu_charge_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int idiag);
   void call_gpu_charge_4p_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int istep, int idiag);
   //void call_gpu_shift_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input);
   void call_gpu_push_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int idiag);
   void call_gpu_push_4p_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int idiag);
   void call_gpu_bin_particles_kernel(gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int shift_direction);
   
   void call_gpu_shifti_extract_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, unsigned int tops[2], real *sends[2], int shift_direction);
   void call_gpu_shifti_append_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int mi_append, real *particle_data);
   
   void print_gpu_timing(gpu_kernel_args_t* gpu_kernel_args);
   
#ifdef __cplusplus
 }
#endif 
#endif

