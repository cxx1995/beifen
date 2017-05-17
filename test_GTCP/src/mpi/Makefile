top_srcdir = ../..
include $(top_srcdir)/Makefile.inc

ifeq ($(CC),icc)
  COPTREPORTFLAGS = -opt-report 3 -opt-report-file=opt-report.log
else
  COPTREPORTFLAGS =
endif

COMMON_HDRS = bench_gtc.h
COMMON_LDFLAGS = $(CLDFLAGS)

#=======================================================================

TARGETS =  
ifeq ($(BENCH_GTC_MPI),1)
  TARGETS += bench_gtc$(EXEEXT)
endif
CLEANFILES = $(TARGETS)


CLEANFILES = $(TARGETS)

all: $(TARGETS)

#=======================================================================

bench_gtc_HDRS = $(COMMON_HDRS)
bench_gtc_SRCS = setup.c chargei_reduction.c smooth.c push.c shifti.c poisson.c field.c \
    RngStream.c diagnosis.c restart_ion.c ghost_comm.c main.c collision.c remap.c calc_moments.c \
    gpu_interface.c
bench_gtc_GPU_SRCS = gpu_setup.cu
cuda_aux_SRCS = gpu_shift_kernel.cu gpu_pushi_kernel.cu gpu_charge_kernel.cu gpu_charge_4p_kernel.cu gpu_bin_shift.cu ptx_custom.cu
bench_gtc_OBJS = $(bench_gtc_SRCS:.c=.o) $(bench_gtc_GPU_SRCS:.cu=.o)
bench_gtc_DEPS = $(bench_gtc_SRCS:.c=.d)  $(bench_gtc_GPU_SRCS:.cu=.d)  $(cuda_aux_GPU_SRCS:.cu=.d)
bench_gtc_LDFLAGS = $(COMMON_LDFLAGS)
-include $(bench_gtc_DEPS)

RngStream.o: RngStream.c RngStream.h
	$(CC) -g -c RngStream.c

bench_gtc$(EXEEXT): $(bench_gtc_OBJS)
	$(MPICC) $(CFLAGS_STD) $(CFLAGSOMP) $(COPTFLAGS) $(NVCC_LINK_FLAGS) \
	  -o $@ $(bench_gtc_OBJS) $(bench_gtc_LDFLAGS)

CLEANFILES += main.o $(bench_gtc_OBJS)
CLEANFILES += $(bench_gtc_DEPS)
ifeq ($(CC),icc)
  CLEANFILES += opt-report.log
endif

#=======================================================================
