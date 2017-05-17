#!/bin/bash  
PPC=100

mkdir $MEMBERWORK/fus100/GTCP_GPU_Mar14/A$PPC
cp bench_gtc_titan_opt_cpu_cray+apa $MEMBERWORK/fus100/GTCP_GPU_Mar14/A$PPC
cp ./input/A.txt $MEMBERWORK/fus100/GTCP_GPU_Mar14/A$PPC
cp maxwell.dat $MEMBERWORK/fus100/GTCP_GPU_Mar14/A$PPC

qsub -v PPC=$PPC submit_titan_cpu_pat.pbs

