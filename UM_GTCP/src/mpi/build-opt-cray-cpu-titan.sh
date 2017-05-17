#!/bin/csh
module unload cudatoolkit PrgEnv-intel PrgEnv-cray PrgEnv-gnu PrgEnv-pgi
module load PrgEnv-cray cudatoolkit

make ARCH=titan-opt-cpu-cray clean
make ARCH=titan-opt-cpu-cray

#pat_build -O apa bench_gtc_titan_opt_cpu_cray 

