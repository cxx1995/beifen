#!/bin/csh
module unload cudatoolkit PrgEnv-intel PrgEnv-cray PrgEnv-gnu PrgEnv-pgi
module load PrgEnv-cray cudatoolkit
#module switch PrgEnv-pgi PrgEnv-cray

make ARCH=titan-debug-cray clean
make ARCH=titan-debug-cray

