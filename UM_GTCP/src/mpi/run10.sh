#!/bin/bash
#Run GTC-P 10 times
#2017/03/23 CXX version-1.0
echo "input file: A.txt";
cycle=1
start=$(date +%s.%N)
while (( $cycle<=10 ))
do
        ./bench_gtc-minsky-nvcc input/A.txt 100 1
        let "cycle++"
done

echo "input file: B.txt";
cycle=1
start=$(date +%s.%N)
while (( $cycle<=10 ))
do
        ./bench_gtc-minsky-nvcc input/B.txt 100 1
        let "cycle++"
done

echo "input file: C.txt";
cycle=1
start=$(date +%s.%N)
while (( $cycle<=10 ))
do
        ./bench_gtc-minsky-nvcc input/C.txt 100 1
        let "cycle++"
done

echo "input file: D.txt";
cycle=1
start=$(date +%s.%N)
while (( $cycle<=10 ))
do
        ./bench_gtc-minsky-nvcc input/D.txt 100 1
        let "cycle++"
done





