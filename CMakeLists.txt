cmake_minimum_required(VERSION 3.5.0)
project(convert_onnx LANGUAGES CXX CUDA)

# 设置 C++14 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")
  set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)
  set(CUDA_INSTALL_TARGET_DIR targets/aarch64-linux)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_C_COMPILER /usr/bin/gcc)
  set(CMAKE_CXX_COMPILER /usr/bin/g++)
  set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

# set(CMAKE_BUILD_TYPE "RELEASE")
set(CMAKE_BUILD_TYPE "DEBUG")

set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")

add_compile_options(-W)
add_compile_options(-std=c++11)

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 86 87 89)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 89)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}
  -ccbin ${CMAKE_CXX_COMPILER}"
)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL  "aarch64")       # orin
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w -gencode=arch=compute_87,code=sm_87")
  set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu/)
  set(TENSORRT_LIBRARY_DIRS /usr/lib/aarch64-linux-gnu/)
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w -gencode=arch=compute_89,code=sm_89")
  set(TENSORRT_INCLUDE_DIRS /mnt/linux_G/TensorRT-8.5.2.2/TensorRT-8.5.2.2/include)  # your tensorrt path
  set(TENSORRT_LIBRARY_DIRS /mnt/linux_G/TensorRT-8.5.2.2/TensorRT-8.5.2.2/lib)
  set(TENSORRT_LIBS_DIRS /mnt/linux_G/TensorRT-8.5.2.2/TensorRT-8.5.2.2/targets/x86_64-linux-gnu/lib)
endif()


set(CUDA_LIB_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/lib)
find_library(NVJPEG_LIBRARY nvjpeg ${CUDA_LIB_DIRS})
if(NVJPEG_LIBRARY)
  add_definitions(-D__HAVE_NVJPEG__)
  link_libraries(${NVJPEG_LIBRARY})
  message(STATUS ${NVJPEG_LIBRARY})
endif()

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
  # ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/plugin/efficientIdxNMSPlugin
  ${PROJECT_SOURCE_DIR}/plugin/
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
	${TENSORRT_LIBRARY_DIRS}
)

cuda_add_executable(export 
  src/export_engine.cpp
  ${PROJECT_SOURCE_DIR}/plugin//efficientIdxNMSPlugin/efficientIdxNMSPlugin.cpp
  ${PROJECT_SOURCE_DIR}/plugin/efficientIdxNMSPlugin/efficientIdxNMSInference.cu
  ${PROJECT_SOURCE_DIR}/plugin/common/plugin.cpp
  ${PROJECT_SOURCE_DIR}/plugin/common/common.cu
)


target_link_libraries(export 
  libnvinfer.so
  libnvonnxparser.so
  libnvinfer_plugin.so.8
  libz.so
)
