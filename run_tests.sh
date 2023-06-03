#!/bin/bash

mkdir results

for graph in /home/cardone/Desktop/graphs/gra/* ; do
graph_name=${graph##*/}
echo "Working on $graph_name..."
mkdir results/$graph_name
#./build/GraphColoring/jp_cuda_double -r 20 $graph > results/$graph_name/double_stream.txt
#./build_no_stream/GraphColoring/jp_cuda_double -r 20 $graph > results/$graph_name/double_no_stream.txt
./build/GraphColoring/jp_cuda_single -r 20 $graph > results/$graph_name/single_stream.txt
./build_no_stream/GraphColoring/jp_cuda_single -r 20 $graph > results/$graph_name/single_no_stream.txt
done
