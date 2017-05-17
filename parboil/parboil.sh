#!/bin/bash
#Run the parboil benchmarks suite 10 times
#count the average results
#2017/03/23 CXX version-1.0

# arg1=start, arg2=end, format: %s.%N  
function getTiming() {  
    start=$1  
    end=$2  
     
    start_s=$(echo $start | cut -d '.' -f 1)  
    start_ns=$(echo $start | cut -d '.' -f 2)  
    end_s=$(echo $end | cut -d '.' -f 1)  
    end_ns=$(echo $end | cut -d '.' -f 2)  
  
  
# for debug..  
#    echo $start  
#    echo $end  

     time=$(( ( 10#$end_s - 10#$start_s ) * 1000 + ( 10#$end_ns / 1000000 - 10#$start_ns / 1000000 ) ))  
  
  
     echo "$time ms"  
}  
cycle=1
./parboil clean stencil cuda
./parboil compile stencil cuda
start=$(date +%s.%N) 
while (( $cycle<=10 ))
do
	./parboil run stencil cuda default
	let "cycle++"
done
end=$(date +%s.%N)
getTiming $start $end
