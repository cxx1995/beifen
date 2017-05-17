#!/bin/csh
module unload cudatoolkit PrgEnv-intel PrgEnv-cray PrgEnv-gnu PrgEnv-pgi
module load PrgEnv-gnu cudatoolkit

make ARCH=titan-opt-gnu clean
make ARCH=titan-opt-gnu

