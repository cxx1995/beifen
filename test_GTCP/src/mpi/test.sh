#!/bin/bash
#Run GTC-P 10 times
#2017/05/10 CXX version-1.0
echo "input file: C.txt";
cycle=1
start=$(date +%s.%N)
while (( $cycle<=10 ))
do
        ./bench_gtc-minsky-nvcc input/C.txt 120 1
        let "cycle++"
done

