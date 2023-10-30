cmake_minimum_required(VERSION 3.16.0)
project(color)

set(CMAKE_CXX_STANDARD 14)

set(D_GRAPHREPRESENTATION_ADJM	GRAPH_REPRESENTATION_ADJ_MATRIX
								PARTITIONED_INPUT_LOAD
								GRAPH_REPR_T=AdjacencyMatrix
)
set(D_GRAPHREPRESENTATION_CSR	GRAPH_REPRESENTATION_CSR
								GRAPH_REPR_T=CompressedSparseRow
)

find_package(Threads REQUIRED)
find_package(Vulkan REQUIRED)
add_subdirectory(./kompute)

include_directories(
    ./kompute/src/include
)
find_package(OpenMP REQUIRED)

set(GRAPH
    vulkanKernels.cpp
)

add_library(colorvk SHARED ${GRAPH})
target_compile_definitions(colorvk PUBLIC VULKAN ${D_GRAPHREPRESENTATION_CSR} COLORING_ALGO_T=JonesPlassmann PARALLEL_GRAPH_COLOR COLORING_ALGORITHM_JP PARTITION_VERTICES_BY_EDGE_NUM USE_CUDA_ALGORITHM)

target_link_libraries(colorvk Threads::Threads OpenMP::OpenMP_CXX kompute)

set(APP_OUTPUT_DIR "${PROJECT_BINARY_DIR}/shaders")
message("Output shader dir: ${APP_OUTPUT_DIR}")


set(GLSL_SOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/coloring.comp"
)

foreach(GLSL ${GLSL_SOURCE_FILES})
    get_filename_component(FILE_NAME ${GLSL} NAME)
    message("Shader ${FILE_NAME}")
    set(SPIRV "${APP_OUTPUT_DIR}/${FILE_NAME}.spv")
    message("will be compiled to ${SPIRV}")

    add_custom_command(
            OUTPUT ${SPIRV}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${APP_OUTPUT_DIR}"
            COMMAND glslang -gVS -V ${GLSL} -o ${SPIRV}
            DEPENDS ${GLSL})
    list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
        Shaders
        DEPENDS ${SPIRV_BINARY_FILES}
)

add_dependencies(colorvk Shaders)
