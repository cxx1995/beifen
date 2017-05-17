#!/bin/csh
module unload cudatoolkit PrgEnv-intel PrgEnv-cray PrgEnv-gnu PrgEnv-pgi
module load PrgEnv-intel cudatoolkit

make ARCH=titan-opt-intel clean
make ARCH=titan-opt-intel

