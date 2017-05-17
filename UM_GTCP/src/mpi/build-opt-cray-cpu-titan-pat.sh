#!/bin/bash

source $MODULESHOME/init/bash

module unload cudatoolkit PrgEnv-intel PrgEnv-cray PrgEnv-gnu PrgEnv-pgi
module load PrgEnv-cray cudatoolkit perftools

make ARCH=titan-opt-cpu-cray clean
make ARCH=titan-opt-cpu-cray

#pat_build -w -T shifti_toroidal -g mpi bench_gtc_titan_opt_cpu_cray 
pat_build -O apa ./bench_gtc_titan_opt_cpu_cray



