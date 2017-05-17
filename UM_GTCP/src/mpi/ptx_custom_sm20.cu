
 __device__ __forceinline__ int custom_popc(unsigned int x)
{
	int ret;
    asm volatile ("{\n\t"
			".reg .u32 t1,t2;\n\t"
			"mov.u32 t1,%1;\n\t"
			"popc.b32 t2,t1;\n\t"
			"mov.u32 %0,t2;\n\t" 
			"}"
			: "=r"(ret) : "r"(x));
    return ret;
}

 __device__ __forceinline__ unsigned int select_gt_u32(unsigned int left, unsigned right)
{
	/*d = (sel == 1) ? a : b;*/
	unsigned int ret;
    asm volatile ("{\n\t"
			".reg .u32 a,b,c,d;\n\t"
			".reg .u32 l,r;\n\t"
			".reg .pred p;\n\t"
			"mov.u32 l,%1;\n\t"
			"mov.u32 r,%2;\n\t"
			"setp.u32.gt p,l,r;\n\t"
			"selp.u32 d,1,0,p;\n\t"
			"mov.u32 %0,d;\n\t"
			"}"
			: "=r"(ret) : "r"(left), "r"(right));
    return ret;
}

 __device__ __forceinline__ unsigned int select_eq_u32(unsigned int left, unsigned right)
{
	/*d = (sel == 1) ? a : b;*/
	unsigned int ret;
    asm ("{\n\t"
			".reg .u32 d;\n\t"
			".reg .u32 l,r;\n\t"
			".reg .pred p;\n\t"
			"mov.u32 l,%1;\n\t"
			"mov.u32 r,%2;\n\t"
			"setp.u32.eq p,l,r;\n\t"
			"selp.u32 d,1,0,p;\n\t"
			"mov.u32 %0,d;\n\t" 
			"}"
			: "=r"(ret) : "r"(left), "r"(right));
    return ret;
}


 __device__ __forceinline__ unsigned int select_gt_f64(real left, real right)
{
	/*d = (sel == 1) ? a : b;*/
	unsigned int ret;
    asm ("{\n\t"
			".reg .u32 d;\n\t"
			".reg .f64 l,r;\n\t"
			".reg .pred p;\n\t"
			"mov.f64 l,%1;\n\t"
			"mov.f64 r,%2;\n\t"
			"setp.f64.gt p,l,r;\n\t"
			"selp.u32 d,1,0,p;\n\t"
			"mov.u32 %0,d;\n\t" 
			"}"
			: "=r"(ret) : "d"(left), "d"(right));
    return ret;
}

 __device__ __forceinline__ unsigned int select_eq_f64(real left, real right)
{
	/*d = (sel == 1) ? a : b;*/
	unsigned int ret;
    asm volatile ("{\n\t"
			".reg .u32 a,b,c,d;\n\t"
			".reg .f64 l,r;\n\t"
			".reg .pred p;\n\t"
			"mov.f64 l,%1;\n\t"
			"mov.f64 r,%2;\n\t"
			"setp.f64.eq p,l,r;\n\t"
			"selp.u32 d,1,0,p;\n\t"
			"mov.u32 %0,d;\n\t" 
			"}"
			: "=r"(ret) : "d"(left), "d"(right));
    return ret;
}



__device__ __forceinline__ int sorting_ballot(int sel, unsigned int count[3], int tidx, int tidy,unsigned int mask)
{
	int keep = (sel==0);
	int left = (sel==1);
	int right = (sel==2);
/*
	mask can be computed with but the compiler may not generate efficient code.
	asm ("{\n\t"
			".reg .u32 a;\n\t"
			"mov.u32 a, %lanemask_le;\n\t"
			"mov.u32 %0,a;\n\t" 
			"}"
			: "=r"(mask) : );

*/

	__shared__ unsigned int count_right_low, count_left_low,count_keep_low;
	unsigned int my_left_ballot = ballot(left);
	unsigned int my_right_ballot = ballot(right);
	unsigned int my_keep_ballot = ballot(keep);

	unsigned int my_count_right = custom_popc(my_right_ballot);
	unsigned int my_count_left = custom_popc(my_left_ballot);
	unsigned int my_count_keep = custom_popc(my_keep_ballot);
	int my_before_ballot = mask&(left*my_left_ballot+right*my_right_ballot+keep*my_keep_ballot);

	int order = custom_popc(my_before_ballot);
	if((tidx==0)&&(tidy==0)) {
		count_right_low = my_count_right;
		count_left_low = my_count_left;
		count_keep_low = my_count_keep;
	}
 	__syncthreads();
 	order += tidy*(count_right_low*right+count_left_low*left + count_keep_low*keep);

	if((tidx==31)&&(tidy==1)) {
		count[2] =  my_count_right + count_right_low;
		count[1] =  my_count_left + count_left_low;
		count[0] =  my_count_keep + count_keep_low;
	}
	return order;
}


/**
 * Enumeration of data movement cache modifiers.
 * adopted from back40 http://code.google.com/p/back40computing/
 */
enum CacheModifier {
	CG,
	CS, 
	CA,
	LU,
	CV,
	WB,
	WT

};

#if defined(__LP64__)
	#define _LP64_ true			
	// 64-bit register modifier for inlined asm
	#define _ASM_PTR_ "l"
#else
	#define _LP64_ false
	// 32-bit register modifier for inlined asm
	#define _ASM_PTR_ "r"
#endif

template <typename T, CacheModifier CACHE_MODIFIER> struct TunedLoad;

#define DEFINE_BASE_GLOBAL_LOAD(base_type, ptx_type, reg_mod)													\
		template <> struct TunedLoad<base_type, CG> {															\
			__device__ __forceinline__ static void Ld(base_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\
		template <> struct TunedLoad<base_type, CS> {															\
			__device__ __forceinline__ static void Ld(base_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\
		template <> struct TunedLoad<base_type, LU> {															\
			__device__ __forceinline__ static void Ld(base_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.lu."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\
		template <> struct TunedLoad<base_type, CV> {															\
			__device__ __forceinline__ static void Ld(base_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.cv."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\
		template <> struct TunedLoad<base_type, CA> {															\
			__device__ __forceinline__ static void Ld(base_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.ca."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};					


#define DEFINE_GLOBAL_LOAD(base_type, dest_type, short_type, ptx_type, reg_mod)									\
		template <> struct TunedLoad<base_type, CG> {															\
			__device__ __forceinline__ static void Ld(dest_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.cg."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\
		template <> struct TunedLoad<base_type, CS> {															\
			__device__ __forceinline__ static void Ld(dest_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.cs."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\
		template <> struct TunedLoad<base_type, LU> {															\
			__device__ __forceinline__ static void Ld(dest_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.lu."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\
		template <> struct TunedLoad<base_type, CV> {															\
			__device__ __forceinline__ static void Ld(dest_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.cv."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\
		template <> struct TunedLoad<base_type, CA> {															\
			__device__ __forceinline__ static void Ld(dest_type &val, const base_type* d_ptr, int offset) {		\
				asm("ld.global.ca."#ptx_type" %0, [%1];" : "="#reg_mod(val) : _ASM_PTR_(d_ptr + offset));		\
			}																									\
		};																										\

	// Cache-modified loads for built-in structures
//	DEFINE_GLOBAL_LOAD(char, signed char, char, s8, r)
//	DEFINE_BASE_GLOBAL_LOAD(signed char, s8, r)		
	DEFINE_GLOBAL_LOAD(short, short, short, s16, h)
	DEFINE_GLOBAL_LOAD(int, int, int, s32, r)
	DEFINE_GLOBAL_LOAD(long, long, long, s64, l)
	DEFINE_GLOBAL_LOAD(long long, long long, longlong, s64, l)
//	DEFINE_GLOBAL_LOAD(unsigned char, unsigned char, uchar, u8, r)
	DEFINE_GLOBAL_LOAD(unsigned short, unsigned short, ushort, u16, h)
	DEFINE_GLOBAL_LOAD(unsigned int, unsigned int, uint, u32, r)
	DEFINE_GLOBAL_LOAD(unsigned long, unsigned long, ulong, u64, l)
	DEFINE_GLOBAL_LOAD(unsigned long long, unsigned long long, ulonglong, u64, l)
	DEFINE_GLOBAL_LOAD(float, float, float, f32, f)
	DEFINE_BASE_GLOBAL_LOAD(double, f64, d)


template <typename T, CacheModifier CACHE_MODIFIER> struct TunedStore;

#define DEFINE_BASE_GLOBAL_STORE(base_type, ptx_type, reg_mod)													\
		template <> struct TunedStore<base_type, CG> {															\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {		\
				asm("st.global.cg."#ptx_type" [%0], %1;" : : _ASM_PTR_(d_ptr + offset), #reg_mod(val));			\
			}																									\
		};																										\
		template <> struct TunedStore<base_type, CS> {															\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {		\
				asm("st.global.cs."#ptx_type" [%0], %1;" : : _ASM_PTR_(d_ptr + offset), #reg_mod(val));			\
			}																									\
		};																										\
		template <> struct TunedStore<base_type, WT> {															\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {		\
				asm("st.global.wt."#ptx_type" [%0], %1;" : : _ASM_PTR_(d_ptr + offset), #reg_mod(val));			\
			}																									\
		};																										\
		template <> struct TunedStore<base_type, WB> {															\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {		\
				asm("st.global.wb."#ptx_type" [%0], %1;" : : _ASM_PTR_(d_ptr + offset), #reg_mod(val));			\
			}																									\
		};																																								
#define DEFINE_GLOBAL_STORE(base_type, dest_type, short_type, ptx_type, reg_mod)								\
		template <> struct TunedStore<base_type, CG> {															\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {		\
				asm("st.global.cg."#ptx_type" [%0], %1;" : : _ASM_PTR_(d_ptr + offset), #reg_mod(val));			\
			}																									\
		};																										\
		template <> struct TunedStore<base_type, CS> {															\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {		\
				asm("st.global.cs."#ptx_type" [%0], %1;" : : _ASM_PTR_(d_ptr + offset), #reg_mod(val));			\
			}																									\
		};																										\
		template <> struct TunedStore<base_type, WT> {															\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {		\
				asm("st.global.wt."#ptx_type" [%0], %1;" : : _ASM_PTR_(d_ptr + offset), #reg_mod(val));			\
			}																									\
		};																										\
		template <> struct TunedStore<base_type, WB> {															\
			__device__ __forceinline__ static void St(const base_type &val, base_type* d_ptr, int offset) {		\
				asm("st.global.wb."#ptx_type" [%0], %1;" : : _ASM_PTR_(d_ptr + offset), #reg_mod(val));			\
			}																									\
		};																										\
	        // Cache-modified stores for built-in structures                                                                                                                          
//      DEFINE_GLOBAL_STORE(char, signed char, char, s8, r)                                                                                                                       
//      DEFINE_BASE_GLOBAL_STORE(signed char, s8, r)                                                                                                                              
DEFINE_GLOBAL_STORE(short, short, short, s16, h)
DEFINE_GLOBAL_STORE(int, int, int, s32, r)
DEFINE_GLOBAL_STORE(long, long, long, s64, l)
DEFINE_GLOBAL_STORE(long long, long long, longlong, s64, l)
//      DEFINE_GLOBAL_STORE(unsigned char, unsigned char, uchar, u8, r)                                                                                                           
DEFINE_GLOBAL_STORE(unsigned short, unsigned short, ushort, u16, h)
DEFINE_GLOBAL_STORE(unsigned int, unsigned int, uint, u32, r)
DEFINE_GLOBAL_STORE(unsigned long, unsigned long, ulong, u64, l)
DEFINE_GLOBAL_STORE(unsigned long long, unsigned long long, ulonglong, u64, l)
DEFINE_GLOBAL_STORE(float, float, float, f32, f)
DEFINE_BASE_GLOBAL_STORE(double, f64, d)

