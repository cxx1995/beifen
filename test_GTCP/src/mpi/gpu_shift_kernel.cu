#ifdef MULTIPLE_FILE

#define GPU_KERNEL
#include <bench_gtc.h>
#include <cutil.h>

extern __device__ __constant__ real temp[MAX_MPSI] __align__ (16);
extern __device__ __constant__ real dtemp[MAX_MPSI] __align__ (16);
extern __device__ __constant__ real rtemi[MAX_MPSI] __align__ (16);
//extern __device__ __constant__ real pfluxpsi[MFLUX+1] __align__ (16);
extern __device__ gtc_global_params_t params __align__ (16);
extern __device__ __constant__ real qtinv[MAX_MPSI] __align__ (16);
extern __device__ __constant__ real delt[MAX_MPSI] __align__ (16);
extern __device__ __constant__ int igrid[MAX_MPSI] __align__ (16);
extern __device__ __constant__ int mtheta[MAX_MPSI] __align__ (16);
extern __device__ gtc_radial_decomp_t radial_decomp __align__ (16);
#endif

#define BASE_SHIFT 		0x00
#define LEFT_SHIFT 		0x01
#define RIGHT_SHIFT		0x02
#define SHIFT_COUNT		0x03
#define GRID_DIM MAX_MULIPROCESSOR

__device__ int mi_tops[GRID_DIM+1];
__device__ int mi_ends[GRID_DIM+1];

__device__ unsigned int shift_flush_top[SHIFT_COUNT] __align__ (16);

#define DTHREAD_BLOCK (THREAD_BLOCK * 2)
#define NPARAMS 12

#define SHARED_BUF_SIZE (DTHREAD_BLOCK*NPARAMS)

#define OPTIMIZED_SHIFT 		1
#define TWO_STAGE_PARTICLE_LOAD         0

#define PROFILE_SHIFT 			0

#define _DEBUG_GPU 0

//#define SHARED_BYPASS			1
/* 	We should not use bypassing  for global data becuase it will cause race condition 
	as we cannot gaurantee that the race condition will not happen.
	The input and the output shared array can overlap. 
*/

#define	copy_zion_g2ns(zion_dest,zion_src,src) {\
	zion_dest[0+tid] = zion_src##1[src];\
	zion_dest[THREAD_BLOCK*1+tid] = zion_src##2[src];\
	zion_dest[THREAD_BLOCK*2+tid] = zion_src##3[src];\
	zion_dest[THREAD_BLOCK*3+tid] = zion_src##4[src];\
	zion_dest[THREAD_BLOCK*4+tid] = zion_src##5[src];\
	zion_dest[THREAD_BLOCK*5+tid] = zion_src##6[src];\
	zion_dest[THREAD_BLOCK*6+tid] = zion_src##01[src];\
	zion_dest[THREAD_BLOCK*7+tid] = zion_src##02[src];\
	zion_dest[THREAD_BLOCK*8+tid] = zion_src##03[src];\
	zion_dest[THREAD_BLOCK*9+tid] = zion_src##04[src];\
	zion_dest[THREAD_BLOCK*10+tid] = zion_src##05[src];\
	zion_dest[THREAD_BLOCK*11+tid] = zion_src##06[src];}

#define	copy_zion_g2gs(zion_dest,zion_src,src) {\
	zion_dest##1[0+tid] = zion_src##1[src];\
	zion_dest##2[THREAD_BLOCK*1+tid] = zion_src##2[src];\
	zion_dest##3[THREAD_BLOCK*2+tid] = zion_src##3[src];\
	zion_dest##4[THREAD_BLOCK*3+tid] = zion_src##4[src];\
	zion_dest##5[THREAD_BLOCK*4+tid] = zion_src##5[src];\
	zion_dest##6[THREAD_BLOCK*5+tid] = zion_src##6[src];\
	zion_dest##01[THREAD_BLOCK*6+tid] = zion_src##01[src];\
	zion_dest##02[THREAD_BLOCK*7+tid] = zion_src##02[src];\
	zion_dest##03[THREAD_BLOCK*8+tid] = zion_src##03[src];\
	zion_dest##04[THREAD_BLOCK*9+tid] = zion_src##04[src];\
	zion_dest##05[THREAD_BLOCK*10+tid] = zion_src##05[src];\
	zion_dest##06[THREAD_BLOCK*11+tid] = zion_src##06[src];}




#define	copy_zion_s2st(zion_dest,dest_base,zion_src) {\
	zion_dest[dest_base] = zion_src[0+tid];\
	zion_dest[dest_base+1] = zion_src[THREAD_BLOCK*1+tid];\
	zion_dest[dest_base+2] = zion_src[THREAD_BLOCK*2+tid];\
	zion_dest[dest_base+3] = zion_src[THREAD_BLOCK*3+tid];\
	zion_dest[dest_base+4] = zion_src[THREAD_BLOCK*4+tid];\
	zion_dest[dest_base+5] = zion_src[THREAD_BLOCK*5+tid];\
	zion_dest[dest_base+6] = zion_src[THREAD_BLOCK*6+tid];\
	zion_dest[dest_base+7] = zion_src[THREAD_BLOCK*7+tid];\
	zion_dest[dest_base+8] = zion_src[THREAD_BLOCK*8+tid];\
	zion_dest[dest_base+9] = zion_src[THREAD_BLOCK*9+tid];\
	zion_dest[dest_base+10] = zion_src[THREAD_BLOCK*10+tid];\
	zion_dest[dest_base+11] = zion_src[THREAD_BLOCK*11+tid];}

#define	copy_zion_s2s(zion_dest,dest_base,zion_src) {\
	zion_dest[dest_base] = zion_src[0+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*1] = zion_src[THREAD_BLOCK*1+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*2] = zion_src[THREAD_BLOCK*2+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*3] = zion_src[THREAD_BLOCK*3+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*4] = zion_src[THREAD_BLOCK*4+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*5] = zion_src[THREAD_BLOCK*5+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*6] = zion_src[THREAD_BLOCK*6+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*7] = zion_src[THREAD_BLOCK*7+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*8] = zion_src[THREAD_BLOCK*8+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*9] = zion_src[THREAD_BLOCK*9+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*10] = zion_src[THREAD_BLOCK*10+tid];\
	zion_dest[dest_base+DTHREAD_BLOCK*11] = zion_src[THREAD_BLOCK*11+tid];}


#define	unified_copy_zion_s2s(zion_dest,dest_base,step,zion_src) {\
	zion_dest[dest_base] = zion_src[0+tid];\
	zion_dest[dest_base+step*1] = zion_src[THREAD_BLOCK*1+tid];\
	zion_dest[dest_base+step*2] = zion_src[THREAD_BLOCK*2+tid];\
	zion_dest[dest_base+step*3] = zion_src[THREAD_BLOCK*3+tid];\
	zion_dest[dest_base+step*4] = zion_src[THREAD_BLOCK*4+tid];\
	zion_dest[dest_base+step*5] = zion_src[THREAD_BLOCK*5+tid];\
	zion_dest[dest_base+step*6] = zion_src[THREAD_BLOCK*6+tid];\
	zion_dest[dest_base+step*7] = zion_src[THREAD_BLOCK*7+tid];\
	zion_dest[dest_base+step*8] = zion_src[THREAD_BLOCK*8+tid];\
	zion_dest[dest_base+step*9] = zion_src[THREAD_BLOCK*9+tid];\
	zion_dest[dest_base+step*10] = zion_src[THREAD_BLOCK*10+tid];\
	zion_dest[dest_base+step*11] = zion_src[THREAD_BLOCK*11+tid];}


#define	one_stage_copy_zion_s2s(zion_dest,dest_base,step,zion_src,src) {\
	zion_dest[dest_base] = zion_src##1[src];\
	zion_dest[dest_base+step*1] = zion_src##2[src];\
	zion_dest[dest_base+step*2] = zion_src##3[src];\
	zion_dest[dest_base+step*3] = zion_src##4[src];\
	zion_dest[dest_base+step*4] = zion_src##5[src];\
	zion_dest[dest_base+step*5] = zion_src##6[src];\
	zion_dest[dest_base+step*6] = zion_src##01[src];\
	zion_dest[dest_base+step*7] = zion_src##02[src];\
	zion_dest[dest_base+step*8] = zion_src##03[src];\
	zion_dest[dest_base+step*9] = zion_src##04[src];\
	zion_dest[dest_base+step*10] = zion_src##05[src];\
	zion_dest[dest_base+step*11] = zion_src##06[src];}


#define copy_zion_g2s(zion_dest,dest_base,zion_src,src_offset) {\
	zion_dest[dest_base] = zion_src##1[src_offset];\
	zion_dest[dest_base+1] = zion_src##2[src_offset];\
	zion_dest[dest_base+2] = zion_src##3[src_offset];\
	zion_dest[dest_base+3] = zion_src##4[src_offset];\
	zion_dest[dest_base+4] = zion_src##5[src_offset];\
	zion_dest[dest_base+5] = zion_src##6[src_offset];\
	zion_dest[dest_base+6] = zion_src##01[src_offset];\
	zion_dest[dest_base+7] = zion_src##02[src_offset];\
	zion_dest[dest_base+8] = zion_src##03[src_offset];\
	zion_dest[dest_base+9] = zion_src##04[src_offset];\
	zion_dest[dest_base+10] = zion_src##05[src_offset];\
	zion_dest[dest_base+11] = zion_src##06[src_offset];}



#define copy_zion_s2g(zion_dest,dest_base,zion_src,src_offset) {\
	zion_dest##1[dest_base] = zion_src[src_offset];\
	zion_dest##2[dest_base] = zion_src[src_offset+1*DTHREAD_BLOCK];\
	zion_dest##3[dest_base] = zion_src[src_offset+2*DTHREAD_BLOCK];\
	zion_dest##4[dest_base] = zion_src[src_offset+3*DTHREAD_BLOCK];\
	zion_dest##5[dest_base] = zion_src[src_offset+4*DTHREAD_BLOCK];\
	zion_dest##6[dest_base] = zion_src[src_offset+5*DTHREAD_BLOCK];\
	zion_dest##01[dest_base] = zion_src[src_offset+6*DTHREAD_BLOCK];\
	zion_dest##02[dest_base] = zion_src[src_offset+7*DTHREAD_BLOCK];\
	zion_dest##03[dest_base] = zion_src[src_offset+8*DTHREAD_BLOCK];\
	zion_dest##04[dest_base] = zion_src[src_offset+9*DTHREAD_BLOCK];\
	zion_dest##05[dest_base] = zion_src[src_offset+10*DTHREAD_BLOCK];\
	zion_dest##06[dest_base] = zion_src[src_offset+11*DTHREAD_BLOCK];}



#define copy_zion_s2f(zion_dest,dest_base,zion_src,src_off) {\
	zion_dest[dest_base] = zion_src[src_off];\
	zion_dest[dest_base+THREAD_BLOCK] = zion_src[(src_off+1*THREAD_BLOCK)];\
	zion_dest[dest_base+2*THREAD_BLOCK] = zion_src[(src_off+2*THREAD_BLOCK)];\
	zion_dest[dest_base+3*THREAD_BLOCK] = zion_src[(src_off+3*THREAD_BLOCK)];\
	zion_dest[dest_base+4*THREAD_BLOCK] = zion_src[(src_off+4*THREAD_BLOCK)];\
	zion_dest[dest_base+5*THREAD_BLOCK] = zion_src[(src_off+5*THREAD_BLOCK)];\
	zion_dest[dest_base+6*THREAD_BLOCK] = zion_src[(src_off+6*THREAD_BLOCK)];\
	zion_dest[dest_base+7*THREAD_BLOCK] = zion_src[(src_off+7*THREAD_BLOCK)];\
	zion_dest[dest_base+8*THREAD_BLOCK] = zion_src[(src_off+8*THREAD_BLOCK)];\
	zion_dest[dest_base+9*THREAD_BLOCK] = zion_src[(src_off+9*THREAD_BLOCK)];\
	zion_dest[dest_base+10*THREAD_BLOCK] = zion_src[(src_off+10*THREAD_BLOCK)];\
	zion_dest[dest_base+11*THREAD_BLOCK] = zion_src[(src_off+11*THREAD_BLOCK)]; }



#define uncoalesced_copy_zion_s2f(zion_dest,dest_base,zion_src,src_base) {\
	zion_dest[dest_base] = zion_src[src_base];\
	zion_dest[dest_base+1] = zion_src[src_base+1];\
	zion_dest[dest_base+2] = zion_src[src_base+2];\
	zion_dest[dest_base+3] = zion_src[src_base+3];\
	zion_dest[dest_base+4] = zion_src[src_base+4];\
	zion_dest[dest_base+5] = zion_src[src_base+5];\
	zion_dest[dest_base+6] = zion_src[src_base+6];\
	zion_dest[dest_base+7] = zion_src[src_base+7];\
	zion_dest[dest_base+8] = zion_src[src_base+8];\
	zion_dest[dest_base+9] = zion_src[src_base+9];\
	zion_dest[dest_base+10] = zion_src[src_base+10];\
	zion_dest[dest_base+11] = zion_src[src_base+11]; }

/*
 * The number of thread blocks should equal to the number of multiprocessor.
 *
 */
#if _DEBUG_GPU
__device__ unsigned int arrived_blocks = 0;
#endif


#if PROFILE_SHIFT
#define RECORD_PHASE_TIME(x) if(tid==0) {\
	end_time = clock();\
	phase_time[x] += end_time-start_time;\
	start_time = end_time;}
#else

#define RECORD_PHASE_TIME(x) 

#endif


__global__ static void 
__launch_bounds__(THREAD_BLOCK/*maxThreadsPerBlock*/, 1/*minBlocksPerMultiprocessor*/)
  gpu_shifti_extract_kernel(gtc_particle_data_t * __restrict__ zion, real * __restrict__ zionshift, int shift_mi) {
  const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
    const int tid = tidy *blockDim.x + tidx;
    const int bid = blockIdx.x +gridDim.x * blockIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
	const int nblocks = gridDim.x * gridDim.y;
	const int step = nthreads;
#if PROFILE_SHIFT
	int i;
#define PHASE_COUNT 16
	clock_t phase_time[PHASE_COUNT];
	clock_t start_time, end_time;
	if(tid==0) {
		for(i=0;i<PHASE_COUNT;i++)
			phase_time[i] = 0;
		start_time = clock();
	}
#endif
	int mi = params.mi;

 	int mi_blocks=mi/nthreads;
	if(mi%nthreads !=0) mi_blocks++;
	int mi_block_share = mi_blocks/nblocks;
	int mi_block_rem =  mi_blocks%nblocks;
	int mi_beg, mi_end;

	if(bid <mi_block_rem) {
		int assignment =  (mi_block_share+1) * nthreads;
		mi_beg =  assignment * bid;
		mi_end =  mi_beg + assignment;
	}else  {
		int assignment =  (mi_block_share) * nthreads;
		mi_beg =  (mi_block_share+1) * nthreads * mi_block_rem + (bid - mi_block_rem) * assignment;
		mi_end =  mi_beg + assignment;
	}

	unsigned int common_buf_flush_left = 0, common_buf_flush_right = 0, common_buf_flush_keep = 0;
	unsigned int src_buf_flush;
	__shared__ unsigned int common_dest_buf_flush;
	unsigned int my_dest_buf_flush;
	unsigned int dest_buf_flush_keep;
	__shared__ unsigned int buf_top[3];
	//aliasses
	unsigned int &buf_top_left=buf_top[1],&buf_top_right=buf_top[2],&buf_top_keep=buf_top[0];
//	__shared__ unsigned int buf_top_left,buf_top_right,buf_top_keep;
	if(tid == 0) {
 		buf_top_keep = 0;
	 	buf_top_left = 0;
	 	buf_top_right = 0;
		mi_ends[bid] = min(mi_end,mi);
		mi_tops[bid] = -1;
	}
	mi_beg += tid;
	dest_buf_flush_keep = mi_beg;

    const real pi2 = params.pi*2;
	const real pi2_inv = params.pi2_inv;
	const real zetamin = params.zetamin;
	const real zetamax = params.zetamax;
 	real * __restrict__ dzion1 = zion->z0;
  	real * __restrict__ dzion2 = zion->z1;
  	real * __restrict__ dzion3 = zion->z2;
  	real * __restrict__ dzion4 = zion->z3;
  	real * __restrict__ dzion5 = zion->z4;
  	real * __restrict__ dzion6 = zion->z5;
 	real * __restrict__ dzion01 = zion->z00;
  	real * __restrict__ dzion02 = zion->z01;
  	real * __restrict__ dzion03 = zion->z02;
  	real * __restrict__ dzion04 = zion->z03;
  	real * __restrict__ dzion05 = zion->z04;
  	real * __restrict__ dzion06 = zion->z05;

	/* 
     *	0: no shift
	 *	1: shift left
	 *	2: shift right
	 */

	__shared__ real szion_buf_base[SHARED_BUF_SIZE*3];
	real * szion_buf_keep = szion_buf_base; 
	real * szion_buf_left = szion_buf_base+SHARED_BUF_SIZE;
	real * szion_buf_right = szion_buf_base+2*SHARED_BUF_SIZE;

#if TWO_STAGE_PARTICLE_LOAD 
	__shared__ real szion_buf_new[THREAD_BLOCK*NPARAMS];
#endif
#if OPTIMIZED_SHIFT
	 real *selected_szion_buf;
	 int mask = (1 << tidx) -1;
#endif
	
	__syncthreads();

//	const int top_max_left = shift_mi/2,top_max_right = shift_mi;
	RECORD_PHASE_TIME(0);
	int m;
	for(m = mi_beg;m<mi_end;m+=step) {
#if TWO_STAGE_PARTICLE_LOAD
			copy_zion_g2ns(szion_buf_new,dzion,m);
			real z3val = szion_buf_new[2*THREAD_BLOCK+tid];
#else
			real z3val = dzion3[m];
#endif
  			real zetaright=min(pi2,z3val)-zetamax;
   	 		real zetaleft = z3val - zetamin;
			RECORD_PHASE_TIME(1);
#if OPTIMIZED_SHIFT
			int shift = zetaright*zetaleft>0;
			zetaright = zetaright*pi2_inv;
        	        zetaright = zetaright-floor(zetaright);
			int right = zetaright<0.5;
			int outside_range = m>=mi; 
			/* 0: keep; 1: left; 2: right */
			int shift_index = shift * (1+ right) - 100 * outside_range;
			unsigned int count[3];
			/* All should call this routine because it implies synchronization */
 			int index = sorting_ballot(shift_index,count,tidx,tidy,mask);
			if(m<mi) {
				index +=buf_top[shift_index];
				index &= (DTHREAD_BLOCK-1);
				int transpose = shift_index>0;
				int dest_step = (DTHREAD_BLOCK-1) * (transpose==0)  + 1;
				int dest_off = index*(1+(NPARAMS-1)*transpose);

				selected_szion_buf = szion_buf_base + shift_index*SHARED_BUF_SIZE;
#if _DEBUG_GPU
			if(index<0 || index>127)
				printf("index problem tid %d, bid: %d\n", tid, bid,index);
			if(dest_off >127*12)
				printf("dest_offset error tid %d, bid: %d, dest_off %d\n", tid, bid,dest_off);
			if(dest_step!=1 &&  dest_step!=128)
				printf("dest_step error tid %d, bid: %d, dest_step %d\n", tid, bid,dest_step);
#endif


#if TWO_STAGE_PARTICLE_LOAD
				unified_copy_zion_s2s(selected_szion_buf,dest_off,dest_step,szion_buf_new);
#else
				one_stage_copy_zion_s2s(selected_szion_buf,dest_off,dest_step,dzion,m);

#endif
			}
			if(tid == (nthreads - 1)) {
#if _DEBUG_GPU
				int total_keep = count[0]+count[1]+count[2];
				if(total_keep!=64)
					printf("thread tid %d, bid: %d, total_keep %d\n", tid, bid, total_keep);
#endif				
				buf_top_keep = (buf_top_keep+count[0])&(DTHREAD_BLOCK-1);
				buf_top_left = (buf_top_left+count[1])&(DTHREAD_BLOCK-1);
				buf_top_right = (buf_top_right+count[2])&(DTHREAD_BLOCK-1);
			}
#else
		if(m<mi) {
			int my_pos;
			if(zetaright*zetaleft > 0) {
 				zetaright = zetaright*pi2_inv;
        	    zetaright = zetaright-floor(zetaright);          
			  if(zetaright < 0.5) { 
				my_pos = NPARAMS * atomicInc(&(buf_top_right),DTHREAD_BLOCK-1);
				copy_zion_s2st(szion_buf_right,my_pos,szion_buf_new);
			  } else {
				my_pos = NPARAMS * atomicInc(&(buf_top_left),DTHREAD_BLOCK-1);
				copy_zion_s2st(szion_buf_left,my_pos,szion_buf_new);
			  }
			} else {
				my_pos = atomicInc(&(buf_top_keep),DTHREAD_BLOCK-1);
				copy_zion_s2s(szion_buf_keep,my_pos,szion_buf_new);
			}
		}
#endif
		
		__syncthreads();
		RECORD_PHASE_TIME(2);

		if(((buf_top_keep - common_buf_flush_keep + DTHREAD_BLOCK) % DTHREAD_BLOCK) >= THREAD_BLOCK) {
			src_buf_flush = common_buf_flush_keep + tid;
			copy_zion_s2g(dzion,dest_buf_flush_keep,szion_buf_keep,src_buf_flush);
			common_buf_flush_keep = (common_buf_flush_keep + THREAD_BLOCK) % DTHREAD_BLOCK;	
			dest_buf_flush_keep+=THREAD_BLOCK;	
		}
//		__syncthreads();
		RECORD_PHASE_TIME(3);

		if(((buf_top_left - common_buf_flush_left + DTHREAD_BLOCK) % DTHREAD_BLOCK) >= THREAD_BLOCK) {
				if(tid==0) 
					common_dest_buf_flush = NPARAMS * atomicAdd(shift_flush_top+1,THREAD_BLOCK);
				__syncthreads();
				my_dest_buf_flush = common_dest_buf_flush +tid;
				src_buf_flush = common_buf_flush_left *NPARAMS +tid;
				copy_zion_s2f(zionshift,my_dest_buf_flush,szion_buf_left,src_buf_flush);
				common_buf_flush_left = (common_buf_flush_left + THREAD_BLOCK) % DTHREAD_BLOCK;	
		}
//		__syncthreads();
		RECORD_PHASE_TIME(4);

		if(((buf_top_right - common_buf_flush_right + DTHREAD_BLOCK) % DTHREAD_BLOCK) >= THREAD_BLOCK) {
			if(tid==0) 
				common_dest_buf_flush = NPARAMS * atomicAdd(shift_flush_top+2,THREAD_BLOCK);
			src_buf_flush = common_buf_flush_right *NPARAMS +tid;
			__syncthreads();
			my_dest_buf_flush = common_dest_buf_flush +tid;
			copy_zion_s2f(zionshift,my_dest_buf_flush,szion_buf_right,src_buf_flush);
			common_buf_flush_right = (common_buf_flush_right + THREAD_BLOCK) % DTHREAD_BLOCK;	
		}
//		__syncthreads();
		RECORD_PHASE_TIME(5);

	}
/* 
 * Flush the remaining blocks
 */
		int remaining = (buf_top_keep - common_buf_flush_keep + DTHREAD_BLOCK) % DTHREAD_BLOCK;
		if(tid<remaining) {
			src_buf_flush = common_buf_flush_keep + tid;
			copy_zion_s2g(dzion,dest_buf_flush_keep,szion_buf_keep,src_buf_flush);
			dest_buf_flush_keep+= remaining;
		}
//		__syncthreads();
		RECORD_PHASE_TIME(6);

		remaining = (buf_top_left - common_buf_flush_left + DTHREAD_BLOCK) % DTHREAD_BLOCK;
		if(remaining>0) {
				if(tid==0) 
					common_dest_buf_flush = atomicAdd(shift_flush_top+1,remaining);
				__syncthreads();
				int src_off = NPARAMS*(common_buf_flush_left+tid);
				int dest = NPARAMS*(common_dest_buf_flush+tid);
				if(tid<remaining)
					uncoalesced_copy_zion_s2f(zionshift,dest,szion_buf_left,src_off);
		}
//		__syncthreads();
		RECORD_PHASE_TIME(7);

		remaining = (buf_top_right - common_buf_flush_right + DTHREAD_BLOCK) % DTHREAD_BLOCK;
		if(remaining>0) {
				if(tid==0) 
					common_dest_buf_flush = atomicAdd(shift_flush_top+2,remaining);
				__syncthreads();
				int src_off = NPARAMS*(common_buf_flush_right+tid);
				int dest_off = NPARAMS*(common_dest_buf_flush+tid);
				if(tid<remaining)
					uncoalesced_copy_zion_s2f(zionshift,dest_off,szion_buf_right,src_off);
		}
//		__syncthreads();
		RECORD_PHASE_TIME(8);

		if(tid == 0)
			mi_tops[bid] = dest_buf_flush_keep;

#if _DEBUG_GPU
		if(tid ==0) {
			unsigned int value = atomicInc(&arrived_blocks, nblocks);
			if(value == (nblocks - 1)) {
				printf("shift left top %d\n",shift_flush_top[1]);
				printf("shift right top %d\n",shift_flush_top[2]);
			}
		}
#endif

#if _DEBUG_GPU
		if(tid ==0) {
			unsigned int value = atomicInc(&arrived_blocks, nblocks);
			if(value == (nblocks - 1)) {
				printf("Shift left particles %d, shift right particle %dn",shift_flush_top[1],shift_flush_top[1]);
				arrived_blocks=0;
				real zetaright,zetaleft,z2val;
				int i;
				int misclassified =0;
				for(i=0;i<shift_flush_top[1];i++) {
					z2val = zionshift[i*NPARAMS+2];
 	 				zetaright = min(pi2, z2val) - zetamax;
    	        	zetaleft  = z2val - zetamin;
				   if (zetaright * zetaleft > 0.0) {
    	            	zetaright = zetaright*pi2_inv; 
        	        	zetaright = zetaright-floor(zetaright);             
	                	if (zetaright < 0.5) {
							printf("mis-classified s left particle %d zval %f, min: %f, max %f\n",i, z2val,zetamin,zetamax);
						misclassified++;
						}
					}
				}
				if(misclassified)
					printf("XXX: Misclassified right bid %d, particles %d of %d\n",bid, misclassified,shift_flush_top[1]);
				misclassified = 0;


				for(i=shift_mi/2;i<shift_flush_top[2];i++) {
					z2val = zionshift[i*NPARAMS+2];
 	 				zetaright = min(pi2, z2val) - zetamax;
    	        	zetaleft  = z2val - zetamin;
				   if (zetaright * zetaleft > 0.0) {
    	            	zetaright = zetaright*pi2_inv; 
        	        	zetaright = zetaright-floor(zetaright);             
	                	if (zetaright >= 0.5) {
							printf("mis-classified shift right particle zval %f, min: %f, max %f\n", z2val,zetamin,zetamax);
							misclassified++;
						}
					}
				}
				if(misclassified)
					printf("XXXX: Misclassified left bid %d, particles %d of %d\n",bid, misclassified,shift_flush_top[2]-shift_mi/2);
			}
		}
#endif


#if PROFILE_SHIFT
	if(bid==0 && (tid==0)) {
		printf("Profiling phases of Shift routine:\n");
		for(i=0;i<PHASE_COUNT;i++)
			if(phase_time[i]>0)
				printf("ph %d: %lld\n",i,phase_time[i]);
			else
				break; 
	}
#endif
}

__global__ static void
__launch_bounds__(THREAD_BLOCK/*maxThreadsPerBlock*/, 1/*minBlocksPerMultiprocessor*/)
  gpu_shifti_radial_extract_kernel(gtc_particle_data_t * __restrict__ zion, real * __restrict__ zionshift, int shift_mi) {

  const int tidx = threadIdx.x;
  const int tidy = threadIdx.y;
  const int tid = tidy *blockDim.x + tidx;
  const int bid = blockIdx.x +gridDim.x * blockIdx.y;
  const int nthreads = blockDim.x * blockDim.y;
  const int nblocks = gridDim.x * gridDim.y;
  const int step = nthreads;
#if PROFILE_SHIFT
  int i;
#define PHASE_COUNT 16
  clock_t phase_time[PHASE_COUNT];
  clock_t start_time, end_time;
  if(tid==0) {
    for(i=0;i<PHASE_COUNT;i++)
      phase_time[i] = 0;
    start_time = clock();
  }
#endif

  int mi = params.mi;

  int mi_blocks=mi/nthreads;
  if(mi%nthreads !=0) mi_blocks++;
  int mi_block_share = mi_blocks/nblocks;
  int mi_block_rem =  mi_blocks%nblocks;
  int mi_beg, mi_end;

  if(bid <mi_block_rem) {
    int assignment =  (mi_block_share+1) * nthreads;
    mi_beg =  assignment * bid;
    mi_end =  mi_beg + assignment;
  }else  {
    int assignment =  (mi_block_share) * nthreads;
    mi_beg =  (mi_block_share+1) * nthreads * mi_block_rem + (bid - mi_block_rem) * assignment;
    mi_end =  mi_beg + assignment;
  }

  unsigned int common_buf_flush_left = 0, common_buf_flush_right = 0, common_buf_flush_keep = 0;
  unsigned int src_buf_flush;
  __shared__ unsigned int common_dest_buf_flush;
  unsigned int my_dest_buf_flush;
  unsigned int dest_buf_flush_keep;
  __shared__ unsigned int buf_top[3];
  //aliasses                                                                                                                                      
  unsigned int &buf_top_left=buf_top[1],&buf_top_right=buf_top[2],&buf_top_keep=buf_top[0];
  //    __shared__ unsigned int buf_top_left,buf_top_right,buf_top_keep;                                                                          

  if (tid==0){
    buf_top_keep = 0;
    buf_top_left = 0;
    buf_top_right = 0;
    mi_ends[bid] = min(mi_end,mi);
    mi_tops[bid] = -1;
  }
  mi_beg += tid;
  dest_buf_flush_keep = mi_beg;
  
  const real a_nover_in = radial_decomp.a_nover_in;
  const real a_nover_out = radial_decomp.a_nover_out;
  const int myrank_radiald = radial_decomp.myrank_radiald;
  const int nradial_dom = radial_decomp.nradial_dom;

  real * __restrict__ dzion1 = zion->z0;
  real * __restrict__ dzion2 = zion->z1;
  real * __restrict__ dzion3 = zion->z2;
  real * __restrict__ dzion4 = zion->z3;
  real * __restrict__ dzion5 = zion->z4;
  real * __restrict__ dzion6 = zion->z5;
  real * __restrict__ dzion01 = zion->z00;
  real * __restrict__ dzion02 = zion->z01;
  real * __restrict__ dzion03 = zion->z02;
  real * __restrict__ dzion04 = zion->z03;
  real * __restrict__ dzion05 = zion->z04;
  real * __restrict__ dzion06 = zion->z05;

  //    0: no shift                                                                                                                               
  //    1: shift left                                                                                                                             
  //    2: shift right                                                                                                                            

  __shared__ real szion_buf_base[SHARED_BUF_SIZE*3];
  real * szion_buf_keep = szion_buf_base;
  real * szion_buf_left = szion_buf_base+SHARED_BUF_SIZE;
  real * szion_buf_right = szion_buf_base+2*SHARED_BUF_SIZE;

#if TWO_STAGE_PARTICLE_LOAD
  __shared__ real szion_buf_new[THREAD_BLOCK*NPARAMS];
#endif
#if OPTIMIZED_SHIFT
  real *selected_szion_buf;
  int mask = (1 << tidx) -1;
#endif
  
  __syncthreads();
  //    const int top_max_left = shift_mi/2,top_max_right = shift_mi;                                                                             
  RECORD_PHASE_TIME(0);
  int m;
  for(m = mi_beg;m<mi_end;m+=step) {
#if TWO_STAGE_PARTICLE_LOAD
    copy_zion_g2ns(szion_buf_new,dzion,m);
    //real z3val = szion_buf_new[2*THREAD_BLOCK+tid];                                                                                             
    real z1val = szion_buf_new[0*THREAD_BLOCK+tid];
#else
    //real z3val = dzion3[m];                                                                                                                     
    real z1val = dzion1[m];
#endif

#if SQRT_PRECOMPUTED
    real r = z1val;
#else
    real r = sqrt(2.0*z1val);
#endif

    RECORD_PHASE_TIME(1);
#if OPTIMIZED_SHIFT
  //int shift = zetaright*zetaleft>0;                                                                                                           
  //zetaright = zetaright*pi2_inv;                                                                                                              
  //zetaright = zetaright-floor(zetaright);                                                                                                     
  //int right = zetaright<0.5;                                                                                    
    

    int shift = ((r<a_nover_in && myrank_radiald > 0)||(r>a_nover_out && myrank_radiald < (nradial_dom-1)));
    int right = (r>a_nover_out && myrank_radiald < (nradial_dom-1));

    //int shift = (r<a_nover_in || r > a_nover_out);
    //int right = r > a_nover_out;

  int outside_range = m>=mi;
  // 0: keep; 1: left; 2: right                                                                                                                 
  int shift_index = shift * (1+ right) - 100 * outside_range;
  unsigned int count[3];
  // All should call this routine because it implies synchronization                                                                            
  int index = sorting_ballot(shift_index,count,tidx,tidy,mask);
  if(m<mi) {
    index +=buf_top[shift_index];
    index &= (DTHREAD_BLOCK-1);
    int transpose = shift_index>0;
    int dest_step = (DTHREAD_BLOCK-1) * (transpose==0)  + 1;
    int dest_off = index*(1+(NPARAMS-1)*transpose);

    selected_szion_buf = szion_buf_base + shift_index*SHARED_BUF_SIZE;
#if _DEBUG_GPU
    if(index<0 || index>127)
      printf("index problem tid %d, bid: %d\n", tid, bid,index);
    if(dest_off >127*12)
      printf("dest_offset error tid %d, bid: %d, dest_off %d\n", tid, bid,dest_off);
    if(dest_step!=1 &&  dest_step!=128)
      printf("dest_step error tid %d, bid: %d, dest_step %d\n", tid, bid,dest_step);
#endif
    
#if TWO_STAGE_PARTICLE_LOAD
    unified_copy_zion_s2s(selected_szion_buf,dest_off,dest_step,szion_buf_new);
#else
    one_stage_copy_zion_s2s(selected_szion_buf,dest_off,dest_step,dzion,m);
#endif
  }
  if(tid == (nthreads - 1)) {
#if _DEBUG_GPU
    int total_keep = count[0]+count[1]+count[2];
    if(total_keep!=64)
      printf("thread tid %d, bid: %d, total_keep %d\n", tid, bid, total_keep);
#endif
    buf_top_keep = (buf_top_keep+count[0])&(DTHREAD_BLOCK-1);
    buf_top_left = (buf_top_left+count[1])&(DTHREAD_BLOCK-1);
    buf_top_right = (buf_top_right+count[2])&(DTHREAD_BLOCK-1);
  }
#else
  if(m<mi) {
    int my_pos;
    //if(zetaright*zetaleft > 0) {                                                                                                              
    if ((r<a_nover_in && myrank_radiald > 0)||(r>a_nover_out && myrank_radiald < (nradial_dom-1) ){
	//if ((r<a_nover_in || r > a_nover_out)){
      //zetaright = zetaright*pi2_inv;                                                                                                          
      //zetaright = zetaright-floor(zetaright);                                                                                                 
      //      if(zetaright < 0.5) {                                                                                                             
      if (r>a_nover_out){
	my_pos = NPARAMS * atomicInc(&(buf_top_right),DTHREAD_BLOCK-1);
	copy_zion_s2st(szion_buf_right,my_pos,szion_buf_new);
      } else {
	my_pos = NPARAMS * atomicInc(&(buf_top_left),DTHREAD_BLOCK-1);
	copy_zion_s2st(szion_buf_left,my_pos,szion_buf_new);
      }
    } else {
      my_pos = atomicInc(&(buf_top_keep),DTHREAD_BLOCK-1);
      copy_zion_s2s(szion_buf_keep,my_pos,szion_buf_new);
    }
  }
#endif
  __syncthreads();
  RECORD_PHASE_TIME(2);

  if(((buf_top_keep - common_buf_flush_keep + DTHREAD_BLOCK) % DTHREAD_BLOCK) >= THREAD_BLOCK) {
    src_buf_flush = common_buf_flush_keep + tid;
    copy_zion_s2g(dzion,dest_buf_flush_keep,szion_buf_keep,src_buf_flush);
    common_buf_flush_keep = (common_buf_flush_keep + THREAD_BLOCK) % DTHREAD_BLOCK;
    dest_buf_flush_keep+=THREAD_BLOCK;
  }
  //          __syncthreads();                                                                                                                  
  RECORD_PHASE_TIME(3);
  if(((buf_top_left - common_buf_flush_left + DTHREAD_BLOCK) % DTHREAD_BLOCK) >= THREAD_BLOCK) {
    if(tid==0)
      common_dest_buf_flush = NPARAMS * atomicAdd(shift_flush_top+1,THREAD_BLOCK);
    __syncthreads();
    my_dest_buf_flush = common_dest_buf_flush +tid;
    src_buf_flush = common_buf_flush_left *NPARAMS +tid;
    copy_zion_s2f(zionshift,my_dest_buf_flush,szion_buf_left,src_buf_flush);
    common_buf_flush_left = (common_buf_flush_left + THREAD_BLOCK) % DTHREAD_BLOCK;
  }
  //          __syncthreads();                                                                                                                  
  RECORD_PHASE_TIME(4);

  if(((buf_top_right - common_buf_flush_right + DTHREAD_BLOCK) % DTHREAD_BLOCK) >= THREAD_BLOCK) {
    if(tid==0)
      common_dest_buf_flush = NPARAMS * atomicAdd(shift_flush_top+2,THREAD_BLOCK);
    src_buf_flush = common_buf_flush_right *NPARAMS +tid;
    __syncthreads();
    my_dest_buf_flush = common_dest_buf_flush +tid;
    copy_zion_s2f(zionshift,my_dest_buf_flush,szion_buf_right,src_buf_flush);
    common_buf_flush_right = (common_buf_flush_right + THREAD_BLOCK) % DTHREAD_BLOCK;
  }
  //          __syncthreads();                                                                                                                  
  RECORD_PHASE_TIME(5);
} // end mi_beg       
  //Flush the remaining blocks                                                                                                                    
  int remaining = (buf_top_keep - common_buf_flush_keep + DTHREAD_BLOCK) % DTHREAD_BLOCK;
  if(tid<remaining) {
    src_buf_flush = common_buf_flush_keep + tid;
    copy_zion_s2g(dzion,dest_buf_flush_keep,szion_buf_keep,src_buf_flush);
    dest_buf_flush_keep+= remaining;
  }
  //              __syncthreads();                                                                                                                
  RECORD_PHASE_TIME(6);

  remaining = (buf_top_left - common_buf_flush_left + DTHREAD_BLOCK) % DTHREAD_BLOCK;
  if(remaining>0) {
    if(tid==0)
      common_dest_buf_flush = atomicAdd(shift_flush_top+1,remaining);
    __syncthreads();
    int src_off = NPARAMS*(common_buf_flush_left+tid);
    int dest = NPARAMS*(common_dest_buf_flush+tid);
    if(tid<remaining)
      uncoalesced_copy_zion_s2f(zionshift,dest,szion_buf_left,src_off);
  }
  //            __syncthreads();                                                                                                                  
  RECORD_PHASE_TIME(7);

  remaining = (buf_top_right - common_buf_flush_right + DTHREAD_BLOCK) % DTHREAD_BLOCK;
  if(remaining>0) {
    if(tid==0)
      common_dest_buf_flush = atomicAdd(shift_flush_top+2,remaining);
    __syncthreads();
    int src_off = NPARAMS*(common_buf_flush_right+tid);
    int dest_off = NPARAMS*(common_dest_buf_flush+tid);
    if(tid<remaining)
      uncoalesced_copy_zion_s2f(zionshift,dest_off,szion_buf_right,src_off);
  }
  //            __syncthreads();                                                                                                                  
  RECORD_PHASE_TIME(8);

  if(tid == 0)
    mi_tops[bid] = dest_buf_flush_keep;

#if _DEBUG_GPU
  if(tid ==0) {
    unsigned int value = atomicInc(&arrived_blocks, nblocks);
    if(value == (nblocks - 1)) {
      printf("shift left top %d\n",shift_flush_top[1]);
      printf("shift right top %d\n",shift_flush_top[2]);
    }
  }
#endif

#if _DEBUG_GPU
  if(tid ==0) {
    unsigned int value = atomicInc(&arrived_blocks, nblocks);
    if(value == (nblocks - 1)) {
      printf("Shift left particles %d, shift right particle %d\n",shift_flush_top[1],shift_flush_top[2]-shift_mi/2);
      arrived_blocks=0;
      //      real zetaright,zetaleft,z2val;                                                                                                      
      real z0val,r;
      int i;
      int misclassified =0;
      for(i=0;i<shift_flush_top[1];i++) {
        //z2val = zionshift[i*NPARAMS+2];                                                                                                         
        //zetaright = min(pi2, z2val) - zetamax;                                                                                                  
        //zetaleft  = z2val - zetamin;                                                                                                            
        //if (zetaright * zetaleft > 0.0) {                                                                                                       
        //  zetaright = zetaright*pi2_inv;                                                                                                        
        //  zetaright = zetaright-floor(zetaright);                                                                                               
        //  if (zetaright < 0.5) {                                                                                                                
        //    printf("mis-classified s left particle %d zval %f, min: %f, max %f\n",i, z2val,zetamin,zetamax);                                    
        //    misclassified++;                                                                                                                    
        //  }                                                                                                                                     
        z0val = zionshift[i*NPARAMS+0];
#if SQRT_PRECOMPUTED
        r = z0val;
#else
        r = sqrt(2.0*z0val);
#endif
        if ((r<a_nover_in && myrank_radiald>0)||(r>a_nover_out && myrank_radiald<(nradial_dom-1))){
	  //if ((r<a_nover_in || r >a_nover_out)){
          //if (r<a_nover_in){
	  if (r>a_nover_out){
            printf("myrank_radiald=%d mis-classified s left particle %d zval %f, a_nover_in: %f\n", myrank_radiald, i, z0val, a_nover_in);
            misclassified++;
          }
        }
      }

  if(misclassified)
    printf("XXX: Misclassified right bid %d, particles %d of %d\n",bid, misclassified,shift_flush_top[1]);
  misclassified = 0;

  for(i=shift_mi/2;i<shift_flush_top[2];i++) {
    //z2val = zionshift[i*NPARAMS+2];                                                                                                         
    //zetaright = min(pi2, z2val) - zetamax;                                                                                                  
    //zetaleft  = z2val - zetamin;                                                                                                            
    //if (zetaright * zetaleft > 0.0) {                                                                                                       
    //  zetaright = zetaright*pi2_inv;                                                                                                        
    //  zetaright = zetaright-floor(zetaright);                                                                                               
    //  if (zetaright >= 0.5) {                                                                                                               
    //    printf("mis-classified shift right particle zval %f, min: %f, max %f\n", z2val,zetamin,zetamax);                                    
    //    misclassified++;                                                                                                                    
    //  }                                                                                                                                       
    z0val =zionshift[i*NPARAMS+0];
#if SQRT_PRECOMPUTED
    r = z0val;
#else
    r = sqrt(2.0*z0val);
#endif
    if ((r<a_nover_in && myrank_radiald>0)||(r>a_nover_out && myrank_radiald<(nradial_dom-1))){
      //if ((r<a_nover_in || r>a_nover_out)){
      //if (r>a_nover_out){
      if (r<a_nover_in) {
	printf("mrank_radiald=%d mis-classified s right particle %d zval %f, a_nover_out: %f\n", myrank_radiald, i, z0val, a_nover_out);
	misclassified++;
      }
    }
}
  
  if(misclassified)
    printf("XXXX: Misclassified left bid %d, particles %d of %d\n",bid, misclassified,shift_flush_top[2]-shift_mi/2);
    }
  }
#endif
  
#if PROFILE_SHIFT
  if(bid==0 && (tid==0)) {
    printf("Profiling phases of Shift routine:\n");
    for(i=0;i<PHASE_COUNT;i++)
      if(phase_time[i]>0)
	printf("ph %d: %lld\n",i,phase_time[i]);
      else
        break;
  }
#endif

}

extern "C"
void call_gpu_shifti_extract_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, unsigned int tops[3], real *sends[3], int shift_direction) {
  int nblocks=gpu_kernel_input->nblocks;
  dim3 dimBlock(gpu_kernel_input->deviceProp.warpSize, THREAD_BLOCK/gpu_kernel_input->deviceProp.warpSize);
  dim3 dimGrid(nblocks,1);
  tops[0] = 0;
  tops[1] = 0;
  tops[2] = gpu_kernel_input->d_max_shift_mi/2;
  gpu_timer_start(gpu_kernel_input);

  CUDA_SAFE_CALL(cudaMemcpyToSymbol(shift_flush_top, tops, 3 * sizeof(unsigned int),0,cudaMemcpyHostToDevice));
  
  gpu_kernel_input->gpu_timing.memtransfer_shift_time += gpu_timer_measure(gpu_kernel_input);
  
  //if (shift_direction==0&&gtc_input->radial_decomp.myrank_radiald==0) printf("Entering shift extract\n");
  if (shift_direction==0) 
    gpu_shifti_extract_kernel<<<dimGrid,dimBlock>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_zion_shift, gpu_kernel_input->d_max_shift_mi);
  else if (shift_direction==1)
    gpu_shifti_radial_extract_kernel<<<dimGrid,dimBlock>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_zion_shift, gpu_kernel_input->d_max_shift_mi);
  else
    printf("other shift_direction options are not available\n");
  cudaError_t lasterror = cudaGetLastError();
  if(lasterror != cudaSuccess)
    printf("Error in launching gpu_shift_extract_kernel routine: %s\n", cudaGetErrorString(lasterror));
  gpu_kernel_input->gpu_timing.device_shift_time += gpu_timer_measure(gpu_kernel_input);
  //if (shift_direction==0&&gtc_input->radial_decomp.myrank_radiald==0) printf("Exiting shift extract\n");
  
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(tops, shift_flush_top, sizeof(unsigned int) * 3,0,cudaMemcpyDeviceToHost));
  if((tops[1]>gpu_kernel_input->d_max_shift_mi/2) || (tops[2] >gpu_kernel_input->d_max_shift_mi)) {
    fprintf(stderr,"small shift buffer on GPU increase the size and rerun. left %d, right %d \n",tops[1],tops[2]);
    exit(1);
  }
  
  gtc_particle_decomp_t *parallel_decomp = &(gtc_input->parallel_decomp);
  sends[1] = parallel_decomp->sendbuf;
#if 0
  sends[1] = (real*) malloc(sizeof(real) * NPARAMS * tops[1]);
#endif
  CUDA_SAFE_CALL(cudaMemcpy((void*)(sends[1]), gpu_kernel_input->ptr_d_zion_shift, NPARAMS * tops[1] * sizeof(real), cudaMemcpyDeviceToHost));
  tops[2] -= gpu_kernel_input->d_max_shift_mi/2; 
  sends[2] = parallel_decomp->sendbuf + NPARAMS*tops[1];
#if 0
  sends[2] = (real *) malloc(sizeof(real) * NPARAMS * tops[2]);
#endif

  int size = gpu_kernel_input->d_max_shift_mi/2;
  CUDA_SAFE_CALL(cudaMemcpy((void*)(sends[2]), (gpu_kernel_input->ptr_d_zion_shift + size*NPARAMS), NPARAMS * tops[2] * sizeof(real), cudaMemcpyDeviceToHost));
  gpu_kernel_input->gpu_timing.memtransfer_shift_time += gpu_timer_measure_end(gpu_kernel_input);
  //printf("tops %d %d, mi: %d\n", tops[1],tops[2],	gtc_input->global_params.mi);
  gtc_input->global_params.mi -= (tops[1] + tops[2]);
}

// For a cache line of 128 bytes 16 doubles are stored.

#define ALIGNMENT_MASK 0x0F

#define copy_zion_g2g(zion_dest,dest_base,zion_src,src_offset) {\
	zion_dest##1[dest_base] = zion_src[src_offset];\
	zion_dest##2[dest_base] = zion_src[src_offset+1];\
	zion_dest##3[dest_base] = zion_src[src_offset+2];\
	zion_dest##4[dest_base] = zion_src[src_offset+3];\
	zion_dest##5[dest_base] = zion_src[src_offset+4];\
	zion_dest##6[dest_base] = zion_src[src_offset+5];\
	zion_dest##01[dest_base] = zion_src[src_offset+6];\
	zion_dest##02[dest_base] = zion_src[src_offset+7];\
	zion_dest##03[dest_base] = zion_src[src_offset+8];\
	zion_dest##04[dest_base] = zion_src[src_offset+9];\
	zion_dest##05[dest_base] = zion_src[src_offset+10];\
	zion_dest##06[dest_base] = zion_src[src_offset+11];}



__device__ static void gtc_shifti_copy_kernel_transpose(gtc_particle_data_t * __restrict__ zion_dest, real * __restrict__ zion_src, int start_dest, int start_src, int size, int tid, int nthreads) 
{
	real * dzion1 = zion_dest->z0;
	real * dzion2 = zion_dest->z1;
	real * dzion3 = zion_dest->z2;
	real * dzion4 = zion_dest->z3;
	real * dzion5 = zion_dest->z4;
	real * dzion6 = zion_dest->z5;
 	real * dzion01 = zion_dest->z00;
	real * dzion02 = zion_dest->z01;
	real * dzion03 = zion_dest->z02;
	real * dzion04 = zion_dest->z03;
	real * dzion05 = zion_dest->z04;
	real * dzion06 = zion_dest->z05;
	int dest_base = start_dest+tid;
	int src_offset = NPARAMS * (start_src + tid);
	for(int i=0;i<size;i+=THREAD_BLOCK) {
		int new_particles = min(THREAD_BLOCK,size-i);
		if(tid<new_particles)
		 	copy_zion_g2g(dzion,dest_base,zion_src,src_offset);
		 dest_base += THREAD_BLOCK;
		 src_offset += THREAD_BLOCK*NPARAMS;
	}


#if _DEBUG_GPU	
	int i;
	if(tid ==0) {
	    const int bid = blockIdx.x +gridDim.x * blockIdx.y;

		for(i=0;i<size;i++)
			if((zion_src[start_off+i*NPARAMS ] !=  zion_dest->z0[start_dest+i]) ||
				(zion_src[start_off+i*NPARAMS+1] !=  zion_dest->z1[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+2] !=  zion_dest->z2[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+3] !=  zion_dest->z3[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+4] !=  zion_dest->z4[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+5] !=  zion_dest->z5[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+6] !=  zion_dest->z00[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+7] !=  zion_dest->z01[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+8] !=  zion_dest->z02[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+9] !=  zion_dest->z03[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+10] !=  zion_dest->z04[start_dest+i]) || 
				(zion_src[start_off+i*NPARAMS+11] !=  zion_dest->z05[start_dest+i])) 
				printf("Bid: %d: Mismatch in copy 1 at %d of %d\n",bid,i,size);
	}
	
#endif

	__syncthreads();
}

#define copy_zion(zion_dest,dest_base,zion_src,src_offset) {\
	zion_dest##1[dest_base] = zion_src##1[src_offset];\
	zion_dest##2[dest_base] = zion_src##2[src_offset];\
	zion_dest##3[dest_base] = zion_src##3[src_offset];\
	zion_dest##4[dest_base] = zion_src##4[src_offset];\
	zion_dest##5[dest_base] = zion_src##5[src_offset];\
	zion_dest##6[dest_base] = zion_src##6[src_offset];\
	zion_dest##01[dest_base] = zion_src##01[src_offset];\
	zion_dest##02[dest_base] = zion_src##02[src_offset];\
	zion_dest##03[dest_base] = zion_src##03[src_offset];\
	zion_dest##04[dest_base] = zion_src##04[src_offset];\
	zion_dest##05[dest_base] = zion_src##05[src_offset];\
	zion_dest##06[dest_base] = zion_src##06[src_offset]; }


__device__ static void gtc_shifti_copy_kernel2(gtc_particle_data_t * __restrict__ zion_dest,gtc_particle_data_t * __restrict__  zion_src, int start_dest, int start_src, int size, int tid, int nthreads) 
{
	real * dzion1 = zion_dest->z0+start_dest;
	real * dzion2 = zion_dest->z1+start_dest;
	real * dzion3 = zion_dest->z2+start_dest;
	real * dzion4 = zion_dest->z3+start_dest;
	real * dzion5 = zion_dest->z4+start_dest;
	real * dzion6 = zion_dest->z5+start_dest;
 	real * dzion01 = zion_dest->z00+start_dest;
	real * dzion02 = zion_dest->z01+start_dest;
	real * dzion03 = zion_dest->z02+start_dest;
	real * dzion04 = zion_dest->z03+start_dest;
	real * dzion05 = zion_dest->z04+start_dest;
	real * dzion06 = zion_dest->z05+start_dest;

	real * szion1 = zion_src->z0+start_src;
	real * szion2 = zion_src->z1+start_src;
	real * szion3 = zion_src->z2+start_src;
	real * szion4 = zion_src->z3+start_src;
	real * szion5 = zion_src->z4+start_src;
	real * szion6 = zion_src->z5+start_src;
 	real * szion01 = zion_src->z00+start_src;
	real * szion02 = zion_src->z01+start_src;
	real * szion03 = zion_src->z02+start_src;
	real * szion04 = zion_src->z03+start_src;
	real * szion05 = zion_src->z04+start_src;
	real * szion06 = zion_src->z05+start_src;

	for(int i=0;i<size;i+=THREAD_BLOCK) {
		int new_particles = min(THREAD_BLOCK,size-i);
		if(tid<new_particles) {
			copy_zion(dzion,(i+tid),szion,(i+tid));
		}
//		__syncthreads();
	}
	
#if _DEBUG_GPU
	int i;
	if(tid ==0) {
		printf("checking copy 2\n");

		for(i=0;i<size;i++)
			if((zion_dest->z0[start_dest+i] != zion_src->z0[start_src+i]) ||
				(zion_dest->z1[start_dest+i] != zion_src->z1[start_src+i]) || 
				(zion_dest->z2[start_dest+i] != zion_src->z2[start_src+i]) || 
				(zion_dest->z3[start_dest+i] != zion_src->z3[start_src+i]) || 
				(zion_dest->z4[start_dest+i] != zion_src->z4[start_src+i]) || 
				(zion_dest->z5[start_dest+i] != zion_src->z5[start_src+i]) || 
				(zion_dest->z00[start_dest+i] != zion_src->z00[start_src+i]) || 
				(zion_dest->z01[start_dest+i] != zion_src->z01[start_src+i]) || 
				(zion_dest->z02[start_dest+i] != zion_src->z02[start_src+i]) || 
				(zion_dest->z03[start_dest+i] != zion_src->z03[start_src+i]) || 
				(zion_dest->z04[start_dest+i] != zion_src->z04[start_src+i]) || 
				(zion_dest->z05[start_dest+i] != zion_src->z05[start_src+i]))
				printf("Mismatch in copy 2 at %d of %d\n",i,size);
	}
#endif
	


}


__global__ 
__launch_bounds__(THREAD_BLOCK/*maxThreadsPerBlock*/, 1/*minBlocksPerMultiprocessor*/)
static void gpu_shifti_append_kernel(gtc_particle_data_t *zion, real *zion_append, int append_mi) {
	const int tidx = threadIdx.x;
    const int tid = threadIdx.y *blockDim.x + tidx;
    const int bid = blockIdx.x +gridDim.x * blockIdx.y;
    const int nthreads = blockDim.x * blockDim.y;
	const int nblocks = gridDim.x * gridDim.y;


	__shared__ int mi_gaps[GRID_DIM+1];
	__shared__ int mi_scan_gaps[GRID_DIM+1];
	__shared__ int mi_elements[GRID_DIM+1];

	if(tid < nblocks){
		if(tid==0)
			mi_elements[tid] = mi_tops[tid];
		else 
			mi_elements[tid] = mi_tops[tid] - mi_ends[tid-1];
		mi_gaps[tid] = mi_ends[tid] - mi_tops[tid];
		mi_scan_gaps[tid] = mi_gaps[tid];
	}

	int offset = 1;
	for (int d = nblocks>>1; d > 0; d >>= 1) {// build sum in place up the tree
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			mi_scan_gaps[bi] += mi_scan_gaps[ai];
		}
		offset *= 2;
	}
	if (tid == 0) { mi_scan_gaps[nblocks-1] = 0; } 
	for (int d = 1; d < nblocks; d *= 2) { // traverse down tree & build scan	
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			int t = mi_scan_gaps[ai];
			mi_scan_gaps[ai] = mi_scan_gaps[bi];
			mi_scan_gaps[bi] += t;
		}
	}
	__syncthreads();
	if (tid == 0) { mi_scan_gaps[nblocks] = mi_scan_gaps[nblocks-1] + mi_gaps[nblocks-1]; }

#if _DEBUG_GPU
	if((tid ==0) && (bid ==0))  {
		int i;
		printf("Block: GAP Element ElementScan top end\n");
		for(i=0;i<nblocks+1;i++) {
			printf("%d: %d %d %d %d %d\n",i,mi_gaps[i],mi_elements[i],mi_scan_gaps[i],mi_tops[i],mi_ends[i]);
		} 
	}
#endif

	__syncthreads();
	
	
 	if((bid == nblocks-1) && ( mi_scan_gaps[bid] <= append_mi)) {

#if _DEBUG_GPU
	if(tid ==0) 
		printf("second condition for %d particles\n",append_mi - mi_scan_gaps[bid]);
#endif

		gtc_shifti_copy_kernel_transpose(zion, zion_append, mi_tops[bid], mi_scan_gaps[bid], append_mi - mi_scan_gaps[bid], tid, nthreads);	
	} else	if(mi_scan_gaps[bid+1] <= append_mi) {

#if _DEBUG_GPU
	if(tid ==0) 
		printf("first condition for %d particles\n", mi_gaps[bid+1]);
#endif

		gtc_shifti_copy_kernel_transpose(zion, zion_append, mi_tops[bid], mi_scan_gaps[bid], mi_gaps[bid], tid, nthreads);
	}
	else {

#if _DEBUG_GPU
		printf("third condition, bid %d\n",bid,append_mi);
#endif

		__shared__ int fill_gap,src_seg;
		int i;
		__shared__ int first_shift_bid;
		__shared__ int element_pool;
		if(tid == 0) {
			fill_gap = 1;
			first_shift_bid = 0;
			for(i=0;i<nblocks;i++)
		  		if(mi_scan_gaps[i+1] > append_mi) {
					first_shift_bid = i;
					break;
				}
			src_seg = nblocks-1;
			element_pool = mi_elements[src_seg];
			if(first_shift_bid<bid) {
				mi_gaps[first_shift_bid] -= (append_mi - mi_scan_gaps[first_shift_bid]); 
			  for(i=first_shift_bid;i<bid;) {
				if(mi_gaps[i] < element_pool) {
					element_pool -= mi_gaps[i];
					i++;
				} else {
					mi_gaps[i] -= element_pool;
					src_seg--;
					if(src_seg == i) {
						fill_gap = 0;
						break;
					}
					element_pool = mi_elements[src_seg];
				}
			 }
			}
		}
		__syncthreads();
		if(fill_gap) {
		 	int start_dest, remainder;
			if(bid == first_shift_bid) {
				int size = append_mi - mi_scan_gaps[bid];
				if(size>0)
					gtc_shifti_copy_kernel_transpose(zion, zion_append, mi_tops[bid], mi_scan_gaps[bid], size, tid, nthreads);
				remainder = mi_gaps[bid]-size;
				start_dest = mi_tops[bid]+size;
			} else {
				remainder = mi_gaps[bid];
				start_dest = mi_tops[bid];
			}
			while((remainder>0)&&(bid != src_seg)) {
				int size = min(remainder, element_pool);
				gtc_shifti_copy_kernel2(zion,zion, start_dest, mi_ends[src_seg-1] + element_pool-size ,size,tid,nthreads);
				start_dest+=size;
				remainder -= size;
				src_seg--;
				element_pool = mi_elements[src_seg];
			}
		}
	}

	if((bid ==nblocks-1) && tid==0) {
	 
#if _DEBUG_GPU
		printf("total gaps :  %d, new particles %d\n",mi_scan_gaps[nblocks],append_mi);
#endif	
	 
		params.mi = params.mi + append_mi - mi_scan_gaps[nblocks];
	}	
}

extern "C"
void call_gpu_shifti_append_kernel (gtc_bench_data_t *gtc_input, gpu_kernel_args_t* gpu_kernel_input, int mi_append, real *particle_data)
{
	int nblocks=gpu_kernel_input->nblocks;
	dim3 dimBlock(gpu_kernel_input->deviceProp.warpSize, THREAD_BLOCK/gpu_kernel_input->deviceProp.warpSize);
	dim3 dimGrid(nblocks,1);
	gpu_timer_start(gpu_kernel_input);
	if(mi_append!=0)
		CUDA_SAFE_CALL(cudaMemcpy(gpu_kernel_input->ptr_d_zion_shift,(void*)(particle_data), NPARAMS *mi_append * sizeof(real), cudaMemcpyHostToDevice));
	gpu_kernel_input->gpu_timing.memtransfer_shift_time +=  gpu_timer_measure(gpu_kernel_input);

	gpu_shifti_append_kernel<<<dimGrid,dimBlock>>>(gpu_kernel_input->ptr_d_zion, gpu_kernel_input->ptr_d_zion_shift,mi_append);
	cudaError_t lasterror = cudaGetLastError();
	if(lasterror != cudaSuccess) {
				printf("Error in launching routine: %s\n", cudaGetErrorString(lasterror));
	}
	gpu_kernel_input->gpu_timing.device_shift_time += gpu_timer_measure_end(gpu_kernel_input);
	gtc_input->global_params.mi +=  mi_append;
}


