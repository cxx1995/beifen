#!/bin/bash

source $MODULESHOME/init/bash

module load cudatoolkit PrgEnv-cray
module list

make ARCH=titan-opt-cray clean
make ARCH=titan-opt-cray

