#ifndef _BENCH_GTC_OPT
#define _BENCH_GTC_OPT

#define FINE_TIMER           1

#define BARRIER_ON           0
/* Set to 1 when ntoroidal = mzetamax in experiments */
#define ASSUME_MZETA_EQUALS1 1

/* Use SSE intrinsics for some steps in charge/push */
/* Set this to 1 only when mzeta=1 and on x86 platforms */
#define SIMD_CODE            0

/* Avoid lookups to the pgyro/gyro arrays in address calculation step.
 * This reduces the memory footprint while increasing the floating 
 * point computation. Should be a good strategy on GPUs. */
#if (!GPU_ACCEL)
#define GYRO_LOCAL_COMPUTE   0

/* This setting avoids loads from auxiliary arrays in push,
   and stores in charge */
#define ONTHEFLY_PUSHAUX     0
#endif

/* debug CPU code */
#define CPU_DEBUG            0

/* Useful when npartdom > 1 */
#define UNIFORM_PARTICLE_LOADING 0

/* 1: place MPI rank along toroidal dimension first
   0: place MPI rank along radial dimension first */
#define TOROIDAL_FIRST      1

/* Charge density grid isn't updated if this is set to 0.
   Used to quantify performance impact of irregular accesses 
   to the charge density grid. */
#define DO_CHARGE_UPDATES    1

#define IDEAL_ALIGNMENT      64

/* phase space remapping */
#define REMAPPING            0

/* two weights scheme of Hu and Krommes */
/* note: TWO_WEIGHTS == 1 for GPU_ACCEL ==1 hasn't been implemented */
#define TWO_WEIGHTS          0

/* calculate moments for diagnosis */
#define CALC_MOMENTS         0

/* call diagnosis */
#define DIAGNOSIS            0

/* call restart */
#define RESTART              0

/* profile shape 0: exp(x^6): 1: sech2(x)*/
#define PROFILE_SHAPE        0

/* normal distibution: 1: loading particle with normal(maxwellian) distribution in velocity space */
#define NORMAL_DIST          1

/* flat backgroup profile */
#define FLAT_PROFILE         1

/* Do not modify below this point */
/*****************************************************/

/* Currently unsupported options, carried 
   over from pthreads code  */

#ifndef SINGLE_PRECISION
#define SINGLE_PRECISION     0
#else
#define SINGLE_PRECISION     1
#endif

#ifndef MIXED_PRECISION  
#define MIXED_PRECISION 0
#else
#define MIXED_PRECISION 1
#endif

#define RADIAL_FINEPART      0

#define PRINT_CHECKSUM       0

/* Permute radial coordinate of particles. 
 * This is set to 0 by default.*/
#define PERMUTE_ZION_R       0

#define SQRT_PRECOMPUTED     1

#define LOCKING              1

#if defined(__x86_64__)
#if (SINGLE_PRECISION | MIXED_PRECISION)
#define SIMD_INCR_DOUBLEPAIR 0
#else
#define SIMD_INCR_DOUBLEPAIR 1
#endif
#else
#define SIMD_INCR_DOUBLEPAIR 0
#endif

#define NUM_TRIALS 10
#define RNG_SEED 232323

#endif
