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
set(D_COLORALGO_CUSPARSE	PARALLEL_GRAPH_COLOR
							COLORING_ALGORITHM_CUSPARSE
							COLORING_ALGO_T=CusparseColoring
)
set(D_COLORALGO_JP_SEQ	SEQUENTIAL_GRAPH_COLOR
						COLORING_ALGORITHM_JP
						COLORING_ALGO_T=JonesPlassmann
)
set(D_COLORALGO_JP_PAR	PARALLEL_GRAPH_COLOR
						COLORING_ALGORITHM_JP
						PARTITION_VERTICES_BY_EDGE_NUM
						COLORING_ALGO_T=JonesPlassmann
)
set(D_COLORALGO_JP_CUDA_SINGLE	PARALLEL_GRAPH_COLOR
								COLORING_ALGORITHM_JP
								USE_CUDA_ALGORITHM
								COLOR_MAX_INDEPENDENT_SET
								COLORING_ALGO_T=JonesPlassmann
)
set(D_COLORALGO_JP_CUDA_DOUBLE	PARALLEL_GRAPH_COLOR
								COLORING_ALGORITHM_JP
								USE_CUDA_ALGORITHM
								COLOR_MIN_MAX_INDEPENDENT_SET
								COLORING_ALGO_T=JonesPlassmann
)
set(D_COLORALGO_GREEDY_SEQ	SEQUENTIAL_GRAPH_COLOR
							COLORING_ALGORITHM_GREEDY
							SORT_LARGEST_DEGREE_FIRST
							SEQUENTIAL_RECOLOR
							COLORING_ALGO_T=Greedy
)
set(D_COLORALGO_GREEDY_PAR	PARALLEL_GRAPH_COLOR
							COLORING_ALGORITHM_GREEDY
							SORT_LARGEST_DEGREE_FIRST
							SEQUENTIAL_RECOLOR
							COLORING_ALGO_T=Greedy
)
set(D_COLORALGO_GM_SEQ_SYNC_STD	SEQUENTIAL_GRAPH_COLOR
								COLORING_ALGORITHM_GM
								COLORING_SYNCHRONOUS
								USE_STANDARD_ALGORITHM
								COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_SEQ_SYNC_IMP	SEQUENTIAL_GRAPH_COLOR
								COLORING_ALGORITHM_GM
								COLORING_SYNCHRONOUS
								USE_IMPROVED_ALGORITHM
								COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_SEQ_ASYNC_STD	SEQUENTIAL_GRAPH_COLOR
									COLORING_ALGORITHM_GM
									COLORING_ASYNCHRONOUS
									USE_STANDARD_ALGORITHM
									COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_SEQ_ASYNC_IMP	SEQUENTIAL_GRAPH_COLOR
									COLORING_ALGORITHM_GM
									COLORING_ASYNCHRONOUS
									USE_IMPROVED_ALGORITHM
									COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_PAR_SYNC_STD	PARALLEL_GRAPH_COLOR
								COLORING_ALGORITHM_GM
								COLORING_SYNCHRONOUS
								USE_STANDARD_ALGORITHM
								COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_PAR_SYNC_IMP	PARALLEL_GRAPH_COLOR
								COLORING_ALGORITHM_GM
								COLORING_SYNCHRONOUS
								USE_IMPROVED_ALGORITHM
								COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_PAR_ASYNC_STD	PARALLEL_GRAPH_COLOR
									COLORING_ALGORITHM_GM
									COLORING_ASYNCHRONOUS
									USE_STANDARD_ALGORITHM
									COLORING_ALGO_T=GebremedhinManne
)
set(D_COLORALGO_GM_PAR_ASYNC_IMP	PARALLEL_GRAPH_COLOR
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
add_library(coloralgo_jp_vk_par STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jp_vk_par PUBLIC ${D_COLORALGO_JP_PAR_VK} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jp_vk_par benchmark barrier ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(barrier STATIC EXCLUDE_FROM_ALL ${SRC_BARRIER})

add_library(benchmark STATIC EXCLUDE_FROM_ALL ${SRC_BENCHMARK})

add_library(coloralgo_cusparse STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_cusparse PUBLIC ${D_COLORALGO_CUSPARSE} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_cusparse benchmark OpenMP::OpenMP_CXX)

add_library(coloralgo_jp_seq STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jp_seq PUBLIC ${D_COLORALGO_JP_SEQ} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jp_seq benchmark OpenMP::OpenMP_CXX)

add_library(coloralgo_jp_par STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jp_par PUBLIC ${D_COLORALGO_JP_PAR} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jp_par benchmark barrier ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(coloralgo_jp_cuda_single STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jp_cuda_single PUBLIC ${D_COLORALGO_JP_CUDA_SINGLE} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jp_cuda_single benchmark barrier OpenMP::OpenMP_CXX)

add_library(coloralgo_jp_cuda_double STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_jp_cuda_double PUBLIC ${D_COLORALGO_JP_CUDA_DOUBLE} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_jp_cuda_double benchmark barrier OpenMP::OpenMP_CXX)

add_library(coloralgo_greedy_seq STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_greedy_seq PUBLIC ${D_COLORALGO_GREEDY_SEQ} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_greedy_seq benchmark ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(coloralgo_greedy_par STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_greedy_par PUBLIC ${D_COLORALGO_GREEDY_PAR} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_greedy_par benchmark ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(coloralgo_gm_seq_sync_std STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_seq_sync_std PUBLIC ${D_COLORALGO_GM_SEQ_SYNC_STD} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_seq_sync_std benchmark barrier OpenMP::OpenMP_CXX)

add_library(coloralgo_gm_seq_sync_imp STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_seq_sync_imp PUBLIC ${D_COLORALGO_GM_SEQ_SYNC_IMP} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_seq_sync_imp benchmark barrier OpenMP::OpenMP_CXX)

add_library(coloralgo_gm_seq_async_std STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_seq_async_std PUBLIC ${D_COLORALGO_GM_SEQ_ASYNC_STD} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_seq_async_std benchmark barrier OpenMP::OpenMP_CXX)

add_library(coloralgo_gm_seq_async_imp STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_seq_async_imp PUBLIC ${D_COLORALGO_GM_SEQ_ASYNC_IMP} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_seq_async_imp benchmark barrier OpenMP::OpenMP_CXX)

add_library(coloralgo_gm_par_sync_std STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_par_sync_std PUBLIC ${D_COLORALGO_GM_PAR_SYNC_STD} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_par_sync_std benchmark barrier ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(coloralgo_gm_par_sync_imp STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_par_sync_imp PUBLIC ${D_COLORALGO_GM_PAR_SYNC_IMP} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_par_sync_imp benchmark barrier ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(coloralgo_gm_par_async_std STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_par_async_std PUBLIC ${D_COLORALGO_GM_PAR_ASYNC_STD} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_par_async_std benchmark barrier ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(coloralgo_gm_par_async_imp STATIC EXCLUDE_FROM_ALL ${SRC_COLORALGO})
target_compile_definitions(coloralgo_gm_par_async_imp PUBLIC ${D_COLORALGO_GM_PAR_ASYNC_IMP} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(coloralgo_gm_par_async_imp benchmark barrier ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(cudakernels STATIC EXCLUDE_FROM_ALL ${SRC_CUDAKERNELS})
target_compile_definitions(cudakernels PUBLIC ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(cudakernels ${CUDA_LIBRARIES} benchmark OpenMP::OpenMP_CXX)

add_library(cudakernels_jp_single STATIC EXCLUDE_FROM_ALL ${SRC_CUDAKERNELS})
target_compile_definitions(cudakernels_jp_single PUBLIC ${D_COLORALGO_JP_CUDA_SINGLE} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(cudakernels_jp_single ${CUDA_LIBRARIES} benchmark OpenMP::OpenMP_CXX)

add_library(cudakernels_jp_double STATIC EXCLUDE_FROM_ALL ${SRC_CUDAKERNELS})
target_compile_definitions(cudakernels_jp_double PUBLIC ${D_COLORALGO_JP_CUDA_DOUBLE} ${D_GRAPHREPRESENTATION_CSR})
target_link_libraries(cudakernels_jp_double ${CUDA_LIBRARIES} benchmark OpenMP::OpenMP_CXX)

add_library(graphrepresentation_adjm STATIC EXCLUDE_FROM_ALL ${SRC_GRAPHREPRESENTATION})
target_compile_definitions(graphrepresentation_adjm PUBLIC ${D_GRAPHREPRESENTATION_ADJM})
target_link_libraries(graphrepresentation_adjm benchmark ${CMAKE_THREAD_LIBS_INIT} OpenMP::OpenMP_CXX)

add_library(graphrepresentation_csr STATIC EXCLUDE_FROM_ALL ${SRC_GRAPHREPRESENTATION})
target_compile_definitions(graphrepresentation_csr PUBLIC ${D_GRAPHREPRESENTATION_CSR} OpenMP::OpenMP_CXX)
target_link_libraries(graphrepresentation_csr benchmark)
# end libraries definition

#start targets definition
add_executable(cusparse main.cpp)
# target_compile_definitions(cusparse PUBLIC ${D_COLORALGO_CUSPARSE} ${D_GRAPHREPRESENTATION_CSR})
include_directories(cusparse ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(cusparse PRIVATE barrier benchmark coloralgo_cusparse cudakernels graphrepresentation_csr)
target_link_libraries(cusparse barrier benchmark coloralgo_cusparse cudakernels graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(jp_seq main.cpp)
# target_compile_definitions(jp_seq PUBLIC ${D_COLORALGO_JP_SEQ} ${D_GRAPHREPRESENTATION_CSR})
include_directories(jp_seq ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(jp_seq PRIVATE barrier benchmark coloralgo_jp_seq graphrepresentation_csr)
target_link_libraries(jp_seq barrier benchmark coloralgo_jp_seq graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(jp_par main.cpp)
# target_compile_definitions(jp_par PUBLIC ${D_COLORALGO_JP_PAR} ${D_GRAPHREPRESENTATION_CSR})
include_directories(jp_par ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(jp_par PRIVATE barrier benchmark coloralgo_jp_par graphrepresentation_csr)
target_link_libraries(jp_par barrier benchmark coloralgo_jp_par graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(jp_cuda_single main.cpp)
# target_compile_definitions(jp_cuda_single PUBLIC ${D_COLORALGO_JP_CUDA_SINGLE} ${D_GRAPHREPRESENTATION_CSR})
include_directories(jp_cuda_single ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(jp_cuda_single PRIVATE barrier benchmark coloralgo_jp_cuda_single cudakernels_jp_single graphrepresentation_csr)
target_link_libraries(jp_cuda_single barrier benchmark coloralgo_jp_cuda_single cudakernels_jp_single graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(jp_cuda_double main.cpp)
# target_compile_definitions(jp_cuda_double PUBLIC ${D_COLORALGO_JP_CUDA_DOUBLE} ${D_GRAPHREPRESENTATION_CSR})
include_directories(jp_cuda_double ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(jp_cuda_double PRIVATE barrier benchmark coloralgo_jp_cuda_double cudakernels_jp_double graphrepresentation_csr)
target_link_libraries(jp_cuda_double barrier benchmark coloralgo_jp_cuda_double cudakernels_jp_double graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(greedy_seq main.cpp)
# target_compile_definitions(greedy_seq PUBLIC ${D_COLORALGO_GREEDY_SEQ} ${D_GRAPHREPRESENTATION_CSR})
include_directories(greedy_seq ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(greedy_seq PRIVATE barrier benchmark coloralgo_greedy_seq graphrepresentation_csr)
target_link_libraries(greedy_seq barrier benchmark coloralgo_greedy_seq graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(greedy_par main.cpp)
# target_compile_definitions(greedy_par PUBLIC ${D_COLORALGO_GREEDY_PAR} ${D_GRAPHREPRESENTATION_CSR})
include_directories(greedy_par ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(greedy_par PRIVATE barrier benchmark coloralgo_greedy_par graphrepresentation_csr)
target_link_libraries(greedy_par barrier benchmark coloralgo_greedy_par graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(gm_seq_sync_std main.cpp)
# target_compile_definitions(gm_seq_sync_std PUBLIC ${D_COLORALGO_GM_SEQ_SYNC_STD} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_seq_sync_std ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_seq_sync_std PRIVATE barrier benchmark coloralgo_gm_seq_sync_std graphrepresentation_csr)
target_link_libraries(gm_seq_sync_std barrier benchmark coloralgo_gm_seq_sync_std graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(gm_par_sync_imp main.cpp)
# target_compile_definitions(gm_par_sync_imp PUBLIC ${D_COLORALGO_GM_PAR_SYNC_IMP} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_par_sync_imp ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_par_sync_imp PRIVATE barrier benchmark coloralgo_gm_par_sync_imp graphrepresentation_csr)
target_link_libraries(gm_par_sync_imp barrier benchmark coloralgo_gm_par_sync_imp graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(gm_par_async_std main.cpp)
# target_compile_definitions(gm_par_async_std PUBLIC ${D_COLORALGO_GM_PAR_ASYNC_STD} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_par_async_std ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_par_async_std PRIVATE barrier benchmark coloralgo_gm_par_async_std graphrepresentation_csr)
target_link_libraries(gm_par_async_std barrier benchmark coloralgo_gm_par_async_std graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(gm_par_async_imp main.cpp)
# target_compile_definitions(gm_par_async_imp PUBLIC ${D_COLORALGO_GM_PAR_ASYNC_IMP} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_par_async_imp ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_par_async_imp PRIVATE barrier benchmark coloralgo_gm_par_async_imp graphrepresentation_csr)
target_link_libraries(gm_par_async_imp barrier benchmark coloralgo_gm_par_async_imp graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(gm_par_sync_std main.cpp)
# target_compile_definitions(gm_par_sync_std PUBLIC ${D_COLORALGO_GM_PAR_SYNC_STD} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_par_sync_std ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_par_sync_std PRIVATE barrier benchmark coloralgo_gm_par_sync_std graphrepresentation_csr)
target_link_libraries(gm_par_sync_std barrier benchmark coloralgo_gm_par_sync_std graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(gm_seq_sync_imp main.cpp)
# target_compile_definitions(gm_seq_sync_imp PUBLIC ${D_COLORALGO_GM_SEQ_SYNC_IMP} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_seq_sync_imp ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_seq_sync_imp PRIVATE barrier benchmark coloralgo_gm_seq_sync_imp graphrepresentation_csr)
target_link_libraries(gm_seq_sync_imp barrier benchmark coloralgo_gm_seq_sync_imp graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(gm_seq_async_std main.cpp)
# target_compile_definitions(gm_seq_async_std PUBLIC ${D_COLORALGO_GM_SEQ_ASYNC_STD} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_seq_async_std ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_seq_async_std PRIVATE barrier benchmark coloralgo_gm_seq_async_std graphrepresentation_csr)
target_link_libraries(gm_seq_async_std barrier benchmark coloralgo_gm_seq_async_std graphrepresentation_csr OpenMP::OpenMP_CXX)

add_executable(gm_seq_async_imp main.cpp)
# target_compile_definitions(gm_seq_async_imp PUBLIC ${D_COLORALGO_GM_SEQ_ASYNC_IMP} ${D_GRAPHREPRESENTATION_CSR})
include_directories(gm_seq_async_imp ${INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/${PROJECT_NAME})
target_link_directories(gm_seq_async_imp PRIVATE barrier benchmark coloralgo_gm_seq_async_imp graphrepresentation_csr)
target_link_libraries(gm_seq_async_imp barrier benchmark coloralgo_gm_seq_async_imp graphrepresentation_csr OpenMP::OpenMP_CXX)

configure_file(GraphColoringConfig.h.in GraphColoringConfig.h)
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
