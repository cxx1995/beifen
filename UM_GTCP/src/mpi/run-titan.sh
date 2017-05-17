#!/bin/bash 

PPC=100

mkdir $MEMBERWORK/fus100/GTCP_GPU_Mar14/A$PPC
cp bench_gtc_titan_opt_cray $MEMBERWORK/fus100/GTCP_GPU_Mar14/A$PPC
cp ./input/A.txt $MEMBERWORK/fus100/GTCP_GPU_Mar14/A$PPC
cp maxwell.dat $MEMBERWORK/fus100/GTCP_GPU_Mar14/A$PPC

qsub -v PPC=$PPC submit_titanA.pbs

mkdir $MEMBERWORK/fus100/GTCP_GPU_Mar14/B$PPC
cp bench_gtc_titan_opt_cray $MEMBERWORK/fus100/GTCP_GPU_Mar14/B$PPC
cp ./input/B.txt $MEMBERWORK/fus100/GTCP_GPU_Mar14/B$PPC
cp maxwell.dat $MEMBERWORK/fus100/GTCP_GPU_Mar14/B$PPC

qsub -v PPC=$PPC submit_titanB.pbs

mkdir $MEMBERWORK/fus100/GTCP_GPU_Mar14/C$PPC
cp bench_gtc_titan_opt_cray $MEMBERWORK/fus100/GTCP_GPU_Mar14/C$PPC
cp ./input/C.txt $MEMBERWORK/fus100/GTCP_GPU_Mar14/C$PPC
cp maxwell.dat $MEMBERWORK/fus100/GTCP_GPU_Mar14/C$PPC

qsub -v PPC=$PPC submit_titanC.pbs

mkdir $MEMBERWORK/fus100/GTCP_GPU_Mar14/D$PPC
cp bench_gtc_titan_opt_cray $MEMBERWORK/fus100/GTCP_GPU_Mar14/D$PPC
cp ./input/D.txt $MEMBERWORK/fus100/GTCP_GPU_Mar14/D$PPC
cp maxwell.dat $MEMBERWORK/fus100/GTCP_GPU_Mar14/D$PPC

qsub -v PPC=$PPC submit_titanD.pbs
