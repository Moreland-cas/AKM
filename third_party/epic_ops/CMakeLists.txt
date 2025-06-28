cmake_minimum_required(VERSION 3.17...3.24)

project(
  EpicOps
  VERSION 0.1
  LANGUAGES C CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# pytorch
find_package(Torch REQUIRED)
set(CMAKE_CXXFLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})
find_library(TORCH_PYTHON_LIBRARY torch_python REQUIRED PATHS "${TORCH_INSTALL_PREFIX}/lib")

# cuda
find_package(CUDAToolkit REQUIRED)

add_subdirectory(src)
