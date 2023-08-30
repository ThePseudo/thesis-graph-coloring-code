cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_ARCHITECTURES 80)

project(GraphColoring
		VERSION 1.0.0
		LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 17)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(REQUIRED_CUDA_VERSION 12.0)
FIND_PACKAGE(CUDA ${REQUIRED_CUDA_VERSION} REQUIRED)
# # /* enables relocatable device code linking */
SET(CUDA_SEPARABLE_COMPILATION ON)
if(CUDA_64_BIT_DEVICE_CODE)
  set(arch_suffix x86_64)
else()
  set(arch_suffix i386)
endif()
# # end /* Find and set CUDA arch */

# # begin /* SET GENCODE_SM */
set(GENCODE_SM10 10)
  # -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 13)
  # -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 20)
  # -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 30)
  # -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 35)
  # -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM37 37)
  # -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_37,code=compute_37)
set(GENCODE_SM50 50)
  # -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)
set(GENCODE_SM52 52)
  # -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52)
set(GENCODE_SM60 60)
  # -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60)
set(GENCODE_SM61 61)
  # -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)
set(GENCODE_SM70 70)
  # -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
set(GENCODE_SM72 72)
  # -gencode=arch=compute_72,code=sm_72 -gencode=arch=compute_72,code=compute_72)
set(GENCODE_SM75 75)
  # -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75)
set(GENCODE_SM80 80)
  # -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80)
#set(GENCODE_SM82 82)
  # -gencode=arch=compute_82,code=sm_82 -gencode=arch=compute_82,code=compute_82)

#set(GENCODE -gencode=arch=compute_10,code=compute_10) # at least generate PTX
# end /* SET GENCODE_SM */

# begin /* Configure build options */
option(GC_GENCODE_SM10
  "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
  OFF)

option(GC_GENCODE_SM13
  "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
  OFF)

option(GC_GENCODE_SM20
  "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
  OFF)

option(GC_GENCODE_SM30
  "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
  OFF)

option(GC_GENCODE_SM35
  "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K40)"
  OFF)

option(GC_GENCODE_SM37
  "ON to generate code for Compute Capability 3.7 devices (e.g. Tesla K80)"
  OFF)

option(GC_GENCODE_SM50
  "ON to generate code for Compute Capability 5.0 devices (e.g. GeForce GTX 750 TI)"
  OFF)

option(GC_GENCODE_SM52
  "ON to generate code for Compute Capability 5.2 devices (e.g. GeForce Titan X)"
  OFF)

#Pascal Architecture: CUDA 8
if (CUDA_VERSION VERSION_EQUAL "8.0" OR CUDA_VERSION VERSION_GREATER "8.0")
  option(GC_GENCODE_SM60
    "ON to generate code for Compute Capability 6.0 devices (e.g. Tesla P100)"
    OFF)
  option(GC_GENCODE_SM61
    "ON to generate code for Compute Capability 6.1 devices (e.g. GeForce GTX 1080)"
    OFF)
endif ()

#Volta Architecture: CUDA 9
if (CUDA_VERSION VERSION_EQUAL "9.0" OR CUDA_VERSION VERSION_GREATER "9.0")
  option(GC_GENCODE_SM70
    "ON to generate code for Compute Capability 7.0 devices (e.g. Volta V100)"
    OFF)
  option(GC_GENCODE_SM72
    "ON to generate code for Compute Capability 7.2 devices (e.g. Jetson AGX Xavier)"
    OFF)
endif ()

#Turing Architecture: CUDA 10
if (CUDA_VERSION VERSION_EQUAL "10.0" OR CUDA_VERSION VERSION_GREATER "10.0")
  option(GC_GENCODE_SM75
    "ON to generate code for Compute Capability 7.5 devices (e.g. GTX 1160 or RTX 2080)"
    OFF)
endif ()

#Ampere Architecture: CUDA 11
if (CUDA_VERSION VERSION_EQUAL "11.0" OR CUDA_VERSION VERSION_GREATER "11.0")
  option(GC_GENCODE_SM80
    "ON to generate code for Compute Capability 8.0 devices"
    ON)
  #option(GC_GENCODE_SM82
  #  "ON to generate code for Compute Capability 8.2 devices"
  #  ON)
endif ()

option (CUDA_AUTODETECT_GENCODE
  "On to enable autodetection of appropriate gencode=X options for each detected device"
  OFF)

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF)

if (CUDA_AUTODETECT_GENCODE)
  include(${CMAKE_SOURCE_DIR}/cmake/AutoDetectCudaArch.cmake)
  set(GENCODE ${GENCODE} ${CUDA_ARCHS})
endif ()

if (GC_GENCODE_SM10)
  set(GENCODE ${GENCODE} ${GENCODE_SM10})
endif(GC_GENCODE_SM10)

if (GC_GENCODE_SM13)
  set(GENCODE ${GENCODE} ${GENCODE_SM13})
endif(GC_GENCODE_SM13)

if (GC_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(GC_GENCODE_SM20)

if (GC_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(GC_GENCODE_SM30)

if (GC_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(GC_GENCODE_SM35)

if (GC_GENCODE_SM37)
  set(GENCODE ${GENCODE} ${GENCODE_SM37})
endif(GC_GENCODE_SM37)

if (GC_GENCODE_SM50)
  set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(GC_GENCODE_SM50)

if (GC_GENCODE_SM52)
  set(GENCODE ${GENCODE} ${GENCODE_SM52})
endif(GC_GENCODE_SM52)

if (GC_GENCODE_SM60)
  set(GENCODE ${GENCODE} ${GENCODE_SM60})
endif(GC_GENCODE_SM60)

if (GC_GENCODE_SM61)
  set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(GC_GENCODE_SM61)

if (GC_GENCODE_SM70)
  set(GENCODE ${GENCODE} ${GENCODE_SM70})
endif(GC_GENCODE_SM70)

if (GC_GENCODE_SM72)
  set(GENCODE ${GENCODE} ${GENCODE_SM72})
endif(GC_GENCODE_SM72)

if (GC_GENCODE_SM75)
  set(GENCODE ${GENCODE} ${GENCODE_SM75})
endif(GC_GENCODE_SM75)

if (GC_GENCODE_SM80)
  set(GENCODE ${GENCODE} ${GENCODE_SM80})
endif(GC_GENCODE_SM80)

#if (GC_GENCODE_SM82)
#  set(GENCODE ${GENCODE} ${GENCODE_SM82})
#endif(GC_GENCODE_SM82)

message(STATUS "Listing chosen GENCODE commands")
foreach(code IN LISTS GENCODE)
    message(STATUS "${code}")
endforeach()


if (CUDA_VERBOSE_PTXAS)
    set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)
# end /* Configure build options */

# c++11 is required
# Already set above. Causes build error on Ubuntu 18.04.1 LTS
set(CUDA_NVCC_FLAGS --expt-extended-lambda -rdc=true -lcudadevrt -O3)
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
message(STATUS ${CUDA_LIBRARIES})
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES ${GENCODE})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES ${GENCODE})
# begin /* Suppress all warnings from nvcc */
if (NOT ENABLE_WARNINGS)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w")
endif (NOT ENABLE_WARNINGS)
# end /* Suppress all warnings from nvcc */

add_subdirectory(GraphColoring)
