set(INCLUDE_DIRS	barrier
					benchmark
					coloralgo
					coloralgo/cusparse
					coloralgo/gebremedhinmanne
					coloralgo/greedy
					coloralgo/jonesplassmann
					cudakernels
					graphrepresentation
					graphrepresentation/GraphRepresentation
					graphrepresentation/adjacencymatrix
					graphrepresentation/csr
)

# start source files definition
set(SRC_BARRIER barrier/barrier.cpp)
set(SRC_BENCHMARK benchmark/benchmark.cpp)
set(SRC_COLORALGO	coloralgo/ColoringAlgorithm.cpp
					coloralgo/cusparse/CusparseColoring.cpp
					coloralgo/gebremedhinmanne/GebremedhinManne.cpp 
					coloralgo/greedy/Greedy.cpp
					coloralgo/jonesplassmann/JonesPlassmann.cpp
)
set(SRC_CUDAKERNELS cudakernels/cudaKernels.cu)
set(SRC_GRAPHREPRESENTATION	graphrepresentation/GraphRepresentation.cpp
							graphrepresentation/adjacencymatrix/AdjacencyMatrix.cpp
							graphrepresentation/csr/CompressedSparseRow.cpp
)
# end source files definition

#start compile definitions
set(D_COLORALGO_CUSPARSE_E	PARALLEL_GRAPH_COLOR
							COLORING_ALGORITHM_CUSPARSE
							COLORING_ALGO_T=CusparseColoring
)
set(D_COLORALGO_JP_C1	PARALLEL_GRAPH_COLOR
						COLORING_ALGORITHM_JP
						PARTITION_VERTICES_EQUALLY
						COLORING_ALGO_T=JonesPlassmann
)
set(D_COLORALGO_JP_C2	PARALLEL_GRAPH_COLOR
						COLORING_ALGORITHM_JP
						PARTITION_VERTICES_BY_EDGE_NUM
						COLORING_ALGO_T=JonesPlassmann
)
set(D_COLORALGO_JPL_D1	PARALLEL_GRAPH_COLOR
						COLORING_ALGORITHM_JP
						USE_CUDA_ALGORITHM
						COLOR_MAX_INDEPENDENT_SET
						COLORING_ALGO_T=JonesPlassmann
)
set(D_COLORALGO_JPL_D2	PARALLEL_GRAPH_COLOR
						COLORING_ALGORITHM_JP
						USE_CUDA_ALGORITHM
						COLOR_MIN_MAX_INDEPENDENT_SET
						COLORING_ALGO_T=JonesPlassmann
)
set(D_COLORALGO_GREEDY_A1 	SEQUENTIAL_GRAPH_COLOR
							COLORING_ALGORITHM_GREEDY
							SORT_VERTEX_ORDER
							SEQUENTIAL_RECOLOR
							COLORING_ALGO_T=Greedy
)
set(D_COLORALGO_GREEDY_A2	SEQUENTIAL_GRAPH_COLOR
							COLORING_ALGORITHM_GREEDY
							SORT_VERTEX_ORDER_REVERSED
							SEQUENTIAL_RECOLOR
							COLORING_ALGO_T=Greedy
)
set(D_COLORALGO_GREEDY_A3	SEQUENTIAL_GRAPH_COLOR
							COLORING_ALGORITHM_GREEDY
							SORT_LARGEST_DEGREE_FIRST
							SEQUENTIAL_RECOLOR
							COLORING_ALGO_T=Greedy
)
set(D_COLORALGO_GREEDY_A4	SEQUENTIAL_GRAPH_COLOR
							COLORING_ALGORITHM_GREEDY
							SORT_SMALLEST_DEGREE_FIRST
							SEQUENTIAL_RECOLOR
							COLORING_ALGO_T=Greedy
)
set(D_COLORALGO_GM_B1	PARALLEL_GRAPH_COLOR
						COLORING_ALGORITHM_GM
						COLORING_SYNCHRONOUS
						USE_STANDARD_ALGORITHM
						COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_B2	PARALLEL_GRAPH_COLOR
							COLORING_ALGORITHM_GM
							COLORING_SYNCHRONOUS
							USE_IMPROVED_ALGORITHM
							COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_B3	PARALLEL_GRAPH_COLOR
							COLORING_ALGORITHM_GM
							COLORING_ASYNCHRONOUS
							USE_STANDARD_ALGORITHM
							COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_B4	PARALLEL_GRAPH_COLOR
							COLORING_ALGORITHM_GM
							COLORING_ASYNCHRONOUS
							USE_IMPROVED_ALGORITHM
							COLORING_ALGO_T=GebremedhinManne
)

set(D_GRAPHREPRESENTATION_ADJM	GRAPH_REPRESENTATION_ADJ_MATRIX
								PARTITIONED_INPUT_LOAD
								GRAPH_REPR_T=AdjacencyMatrix
)
set(D_GRAPHREPRESENTATION_CSR	GRAPH_REPRESENTATION_CSR
								GRAPH_REPR_T=CompressedSparseRow
)
# end compile definitions

# start libraries definition
add_library(barrier STATIC EXCLUDE_FROM_ALL ${SRC_BARRIER})

add_library(benchmark STATIC EXCLUDE_FROM_ALL ${SRC_BENCHMARK})

add_library(coloralgo_cusparse_e STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_cusparse_e PUBLIC ${D_COLORALGO_CUSPARSE_E} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_cusparse_e benchmark)

# add_library(coloralgo_jp_seq STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
# target_compile_definitions(coloralgo_jp_seq PUBLIC ${D_COLORALGO_JP_SEQ} ${D_GRAPHREPRESENTATION_CSR})
# target_link_libraries(coloralgo_jp_seq benchmark)

add_library(coloralgo_jp_c1 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jp_c1 PUBLIC ${D_COLORALGO_JP_C1} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jp_c1 benchmark barrier ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_jp_c2 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jp_c2 PUBLIC ${D_COLORALGO_JP_C2} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jp_c2 benchmark barrier ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_jpl_d1 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jpl_d1 PUBLIC ${D_COLORALGO_JPL_D1} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jpl_d1 benchmark barrier)

add_library(coloralgo_jpl_d2 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jpl_d2 PUBLIC ${D_COLORALGO_JPL_D2} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jpl_d2 benchmark barrier)

add_library(coloralgo_greedy_a1 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_greedy_a1 PUBLIC ${D_COLORALGO_GREEDY_A1} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_greedy_a1 benchmark ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_greedy_a2 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_greedy_a2 PUBLIC ${D_COLORALGO_GREEDY_A2} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_greedy_a2 benchmark ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_greedy_a3 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_greedy_a3 PUBLIC ${D_COLORALGO_GREEDY_A3} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_greedy_a3 benchmark ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_greedy_a4 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_greedy_a4 PUBLIC ${D_COLORALGO_GREEDY_A4} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_greedy_a4 benchmark ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_gm_b1 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_b1 PUBLIC ${D_COLORALGO_GM_B1} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_b1 benchmark barrier ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_gm_b2 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_b2 PUBLIC ${D_COLORALGO_GM_B2} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_b2 benchmark barrier ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_gm_b3 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_b3 PUBLIC ${D_COLORALGO_GM_B3} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_b3 benchmark barrier ${CMAKE_THREAD_LIBS_INIT})

add_library(coloralgo_gm_b4 STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_b4 PUBLIC ${D_COLORALGO_GM_B4} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_b4 benchmark barrier ${CMAKE_THREAD_LIBS_INIT})

add_library(cudakernels_e STATIC EXCLUDE_FROM_ALL ${SRC_CUDAKERNELS})
target_compile_definitions(cudakernels_e PUBLIC ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(cudakernels_e ${CUDA_LIBRARIES} benchmark)

add_library(cudakernels_jpl_d1 STATIC EXCLUDE_FROM_ALL ${SRC_CUDAKERNELS})
target_compile_definitions(cudakernels_jpl_d1 PUBLIC ${D_COLORALGO_JPL_D1} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(cudakernels_jpl_d1 ${CUDA_LIBRARIES} benchmark)

add_library(cudakernels_jpl_d2 STATIC EXCLUDE_FROM_ALL ${SRC_CUDAKERNELS})
target_compile_definitions(cudakernels_jpl_d2 PUBLIC ${D_COLORALGO_JPL_D2} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(cudakernels_jpl_d2 ${CUDA_LIBRARIES} benchmark)

add_library(graphrepresentation_adjm STATIC EXCLUDE_FROM_ALL ${SRC_GRAPHREPRESENTATION})
target_compile_definitions(graphrepresentation_adjm PUBLIC ${D_GRAPHREPRESENTATION_ADJM})
target_link_libraries(graphrepresentation_adjm benchmark ${CMAKE_THREAD_LIBS_INIT})

add_library(graphrepresentation_csr STATIC EXCLUDE_FROM_ALL ${SRC_GRAPHREPRESENTATION})
target_compile_definitions(graphrepresentation_csr PUBLIC ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(graphrepresentation_csr benchmark)
# end libraries definition

#start targets definition
add_executable(cusparse_e main.cpp)
# target_compile_definitions(cusparse_e PUBLIC ${D_COLORALGO_CUSPARSE_E} ${D_GRAPHREPRESENTATION_CSR})
include_directories(cusparse_e ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(cusparse_e PRIVATE barrier benchmark coloralgo_cusparse_e cudakernels_e graphrepresentation_csr)
target_link_libraries(cusparse_e barrier benchmark coloralgo_cusparse_e cudakernels_e graphrepresentation_csr)

add_executable(jp_c1 main.cpp)
# target_compile_definitions(jp_c1 PUBLIC ${D_COLORALGO_JP_C1} ${D_GRAPHREPRESENTATION_CSR})
include_directories(jp_c1 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(jp_c1 PRIVATE barrier benchmark coloralgo_jp_c1 graphrepresentation_csr)
target_link_libraries(jp_c1 barrier benchmark coloralgo_jp_c1 graphrepresentation_csr)

add_executable(jp_c2 main.cpp)
# target_compile_definitions(jp_c2 PUBLIC ${D_COLORALGO_JP_C1} ${D_GRAPHREPRESENTATION_CSR})
include_directories(jp_c2 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(jp_c2 PRIVATE barrier benchmark coloralgo_jp_c2 graphrepresentation_csr)
target_link_libraries(jp_c2 barrier benchmark coloralgo_jp_c2 graphrepresentation_csr)

add_executable(jpl_d1 main.cpp)
# target_compile_definitions(jpl_d1 PUBLIC ${D_COLORALGO_JPL_D1} ${D_GRAPHREPRESENTATION_CSR})
include_directories(jpl_d1 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(jpl_d1 PRIVATE barrier benchmark coloralgo_jpl_d1 cudakernels_jpl_d1 graphrepresentation_csr)
target_link_libraries(jpl_d1 barrier benchmark coloralgo_jpl_d1 cudakernels_jpl_d1 graphrepresentation_csr)

add_executable(jpl_d2 main.cpp)
# target_compile_definitions(jpl_d2 PUBLIC ${D_COLORALGO_JPL_D2} ${D_GRAPHREPRESENTATION_CSR})
include_directories(jpl_d2 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(jpl_d2 PRIVATE barrier benchmark coloralgo_jpl_d2 cudakernels_jpl_d2 graphrepresentation_csr)
target_link_libraries(jpl_d2 barrier benchmark coloralgo_jpl_d2 cudakernels_jpl_d2 graphrepresentation_csr)

add_executable(greedy_a1 main.cpp)
# target_compile_definitions(greedy_a1 PUBLIC ${D_COLORALGO_GREEDY_A1} ${D_GRAPHREPRESENTATION_CSR})
include_directories(greedy_a1 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(greedy_a1 PRIVATE barrier benchmark coloralgo_greedy_a1 graphrepresentation_csr)
target_link_libraries(greedy_a1 barrier benchmark coloralgo_greedy_a1 graphrepresentation_csr)

add_executable(greedy_a2 main.cpp)
# target_compile_definitions(greedy_a2 PUBLIC ${D_COLORALGO_GREEDY_A2} ${D_GRAPHREPRESENTATION_CSR})
include_directories(greedy_a2 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(greedy_a2 PRIVATE barrier benchmark coloralgo_greedy_a2 graphrepresentation_csr)
target_link_libraries(greedy_a2 barrier benchmark coloralgo_greedy_a2 graphrepresentation_csr)

add_executable(greedy_a3 main.cpp)
# target_compile_definitions(greedy_a3 PUBLIC ${D_COLORALGO_GREEDY_A3} ${D_GRAPHREPRESENTATION_CSR})
include_directories(greedy_a3 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(greedy_a3 PRIVATE barrier benchmark coloralgo_greedy_a3 graphrepresentation_csr)
target_link_libraries(greedy_a3 barrier benchmark coloralgo_greedy_a3 graphrepresentation_csr)

add_executable(greedy_a4 main.cpp)
# target_compile_definitions(greedy_a4 PUBLIC ${D_COLORALGO_GREEDY_A4} ${D_GRAPHREPRESENTATION_CSR})
include_directories(greedy_a4 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(greedy_a4 PRIVATE barrier benchmark coloralgo_greedy_a4 graphrepresentation_csr)
target_link_libraries(greedy_a4 barrier benchmark coloralgo_greedy_a4 graphrepresentation_csr)

add_executable(gm_b2 main.cpp)
# target_compile_definitions(gm_b2 PUBLIC ${D_COLORALGO_GM_B2} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_b2 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_b2 PRIVATE barrier benchmark coloralgo_gm_b2 graphrepresentation_csr)
target_link_libraries(gm_b2 barrier benchmark coloralgo_gm_b2 graphrepresentation_csr)

add_executable(gm_b3 main.cpp)
# target_compile_definitions(gm_b3 PUBLIC ${D_COLORALGO_GM_B3} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_b3 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_b3 PRIVATE barrier benchmark coloralgo_gm_b3 graphrepresentation_csr)
target_link_libraries(gm_b3 barrier benchmark coloralgo_gm_b3 graphrepresentation_csr)

add_executable(gm_b4 main.cpp)
# target_compile_definitions(gm_b4 PUBLIC ${D_COLORALGO_GM_B4} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_b4 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME}) 
target_link_directories(gm_b4 PRIVATE barrier benchmark coloralgo_gm_b4 graphrepresentation_csr)
target_link_libraries(gm_b4 barrier benchmark coloralgo_gm_b4 graphrepresentation_csr)

add_executable(gm_b1 main.cpp)
# target_compile_definitions(gm_b1 PUBLIC ${D_COLORALGO_GM_B1} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_b1 ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME}) 
target_link_directories(gm_b1 PRIVATE barrier benchmark coloralgo_gm_b1 graphrepresentation_csr)
target_link_libraries(gm_b1 barrier benchmark coloralgo_gm_b1 graphrepresentation_csr)

#end target definition


configure_file(GraphColoringConfig.h.in GraphColoringConfig.h)
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
