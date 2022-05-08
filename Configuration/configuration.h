#ifndef _CONFIGURATION_H
#define _CONFIGURATION_H

////////////////////////////////////////////////////////
/////////////////////////////////// GRAPH REPRESENTATION
////////////////////////////////////////////////////////

//#define GRAPH_REPRESENTATION_ADJ_MATRIX
#define GRAPH_REPRESENTATION_CSR

////////////////////////////////////////////////////////
////////////////////////////////////////////// FILE LOAD
////////////////////////////////////////////////////////

//#define PARALLEL_INPUT_LOAD
#define PARTITIONED_INPUT_LOAD
//#define SEQUENTIAL_INPUT_LOAD

////////////////////////////////////////////////////////
///////////////////////////////////// COLORING ALGORITHM
////////////////////////////////////////////////////////

//#define COLORING_ALGORITHM_GREEDY
#define COLORING_ALGORITHM_JP
//#define COLORING_ALGORITHM_GM
//#define COLORING_ALGORITHM_CUSPARSE

#define PARALLEL_GRAPH_COLOR
//#define SEQUENTIAL_GRAPH_COLOR

////////////////////////////////////////////////////////
///////////////////////////////////////// GREEDY OPTIONS
////////////////////////////////////////////////////////

#define SORT_LARGEST_DEGREE_FIRST
//#define SORT_SMALLEST_DEGREE_FIRST
//#define SORT_VERTEX_ORDER
//#define SORT_VERTEX_ORDER_REVERSED

//#define PARALLEL_RECOLOR
#define SEQUENTIAL_RECOLOR

////////////////////////////////////////////////////////
//////////////////////////////// JONES-PLASSMANN OPTIONS
////////////////////////////////////////////////////////

//#define PARTITION_VERTICES_EQUALLY
#define PARTITION_VERTICES_BY_EDGE_NUM

//#define USE_CUDA_ALGORITHM

////////////////////////////////////////////////////////
////////////////////////////// GEBREMEDHIN-MANNE OPTIONS
////////////////////////////////////////////////////////

#define COLORING_SYNCHRONOUS
//#define COLORING_ASYNCHRONOUS

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
/////////////////////////////////// REMOVING DUPLICATE FLAGS
///////////////////////////////////////// WITHIN SAME OPTION
////////////////////////////////////////////////////////////

#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
#undef GRAPH_REPRESENTATION_CSR
#endif

#ifdef GRAPH_REPRESENTATION_CSR
#undef GRAPH_REPRESENTATION_ADJ_MATRIX
#endif

#ifdef PARALLEL_INPUT_LOAD
#undef PARTITIONED_INPUT_LOAD
#undef SEQUENTIAL_INPUT_LOAD
#endif

#ifdef PARTITIONED_INPUT_LOAD
#undef PARALLEL_INPUT_LOAD
#undef SEQUENTIAL_INPUT_LOAD
#endif

#ifdef SEQUENTIAL_INPUT_LOAD
#undef PARALLEL_INPUT_LOAD
#undef PARTITIONED_INPUT_LOAD
#endif

#ifdef SORT_LARGEST_DEGREE_FIRST
#undef SORT_SMALLEST_DEGREE_FIRST
#undef SORT_VERTEX_ORDER
#undef SORT_VERTEX_ORDER_REVERSED 
#endif

#ifdef SORT_SMALLEST_DEGREE_FIRST
#undef SORT_LARGEST_DEGREE_FIRST
#undef SORT_VERTEX_ORDER
#undef SORT_VERTEX_ORDER_REVERSED 
#endif

#ifdef SORT_VERTEX_ORDER
#undef SORT_LARGEST_DEGREE_FIRST
#undef SORT_SMALLEST_DEGREE_FIRST
#undef SORT_VERTEX_ORDER_REVERSED 
#endif

#ifdef SORT_VERTEX_ORDER_REVERSED 
#undef SORT_LARGEST_DEGREE_FIRST
#undef SORT_SMALLEST_DEGREE_FIRST
#undef SORT_VERTEX_ORDER
#endif

#ifdef COLORING_ALGORITHM_GREEDY
#undef COLORING_ALGORITHM_JP
#undef COLORING_ALGORITHM_GM
#undef COLORING_ALGORITHM_CUSPARSE
#endif

#ifdef COLORING_ALGORITHM_JP
#undef COLORING_ALGORITHM_GREEDY
#undef COLORING_ALGORITHM_GM
#undef COLORING_ALGORITHM_CUSPARSE
#endif

#ifdef COLORING_ALGORITHM_GM
#undef COLORING_ALGORITHM_GREEDY
#undef COLORING_ALGORITHM_JP
#undef COLORING_ALGORITHM_CUSPARSE
#endif

#ifdef COLORING_ALGORITHM_CUSPARSE
#indef COLORING_ALGORITHM_GREEDY
#undef COLORING_ALGORITHM_JP
#undef COLORING_ALGORITHM_GM
#endif

#ifdef PARALLEL_GRAPH_COLOR
#undef SEQUENTIAL_GRAPH_COLOR
#endif

#ifdef SEQUENTIAL_GRAPH_COLOR
#undef PARALLEL_GRAPH_COLOR
#endif

#ifdef PARTITION_VERTICES_EQUALLY
#undef PARTITION_VERTICES_BY_EDGE_NUM
#endif

#ifdef PARTITION_VERTICES_BY_EDGE_NUM
#undef PARTITION_VERTICES_EQUALLY
#endif

#ifdef COLORING_SYNCHRONOUS
#undef COLORING_ASYNCHRONOUS
#endif

#ifdef ASYNCHRONOUS
#undef COLORING_ASYNCHRONOUS
#endif

#ifdef PARALLEL_RECOLOR
#undef SEQUENTIAL_RECOLOR
#endif

#ifdef SEQUENTIAL_RECOLOR
#undef PARALLEL_RECOLOR
#endif

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
///////////////////////////////////// SETUP VARIABLES NEEDED
////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////
//////////////////////////// SETUP GRAPH REPRESENTATION TYPE
////////////////////////////////////////////////////////////

#ifdef GRAPH_REPRESENTATION_ADJ_MATRIX
#define GRAPH_REPR_T AdjacencyMatrix
#endif

#ifdef GRAPH_REPRESENTATION_CSR
#define GRAPH_REPR_T CompressedSparseRow
#endif

////////////////////////////////////////////////////////////
////////////////////////////// SETUP COLORING ALGORITHM TYPE
////////////////////////////////////////////////////////////

#ifdef COLORING_ALGORITHM_GREEDY
#define COLORING_ALGO_T Greedy
#endif

#ifdef COLORING_ALGORITHM_GM
#define COLORING_ALGO_T GebremedhinManne
#endif

#ifdef COLORING_ALGORITHM_JP
#define COLORING_ALGO_T JonesPlassmann
#endif

#ifdef COLORING_ALGORITHM_CUSPARSE
#define COLORING_ALGO_T CusparseColoring
#endif

#endif // !_CONFIGURATION_H