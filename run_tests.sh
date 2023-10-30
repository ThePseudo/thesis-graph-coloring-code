#!/bin/bash

mkdir results
./build/GraphColoring/jp_cuda_double -r 20 /home/cardone/Desktop/graphs/gra/apache2.gra > /dev/null
for graph in /home/cardone/Desktop/graphs/gra/* ; do
    graph_name=${graph##*/}
    if [ "$graph_name" != "new" ] ; then
    echo "Stream on double: $graph_name..."
    mkdir results/$graph_name
    time ./build/GraphColoring/jp_cuda_double -r 20 $graph > results/$graph_name/double_stream.txt
    fi
done

./build_no_stream/GraphColoring/jp_cuda_double -r 20 /home/cardone/Desktop/graphs/gra/apache2.gra > /dev/null
for graph in /home/cardone/Desktop/graphs/gra/* ; do
    graph_name=${graph##*/}
    echo "Vanilla on double: $graph_name..."
    ./build_no_stream/GraphColoring/jp_cuda_double -r 20 $graph > results/$graph_name/double_no_stream.txt
done

./build/GraphColoring/jp_cuda_single -r 20 /home/cardone/Desktop/graphs/gra/apache2.gra > /dev/null
for graph in /home/cardone/Desktop/graphs/gra/* ; do
    graph_name=${graph##*/}
    if [ "$graph_name" != "new" ] ; then

    echo "Stream on single: $graph_name..."
    ./build/GraphColoring/jp_cuda_single -r 20 $graph > results/$graph_name/single_stream.txt
    fi
done

./build_no_stream/GraphColoring/jp_cuda_single -r 20 /home/cardone/Desktop/graphs/gra/apache2.gra > /dev/null
for graph in /home/cardone/Desktop/graphs/gra/* ; do
    graph_name=${graph##*/}
    echo "Vanilla on single: $graph_name..."
    ./build_no_stream/GraphColoring/jp_cuda_single -r 20 $graph > results/$graph_name/single_no_stream.txt
done
