cmake_minimum_required (VERSION 3.18 FATAL_ERROR)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 50)

project ("VoxelOptix" LANGUAGES CXX CUDA)

find_package(CUDAToolkit)
find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
find_package(glm REQUIRED)

add_executable (VoxelOptix "src/main.cpp" "pch.h" "src/App.cpp" "src/Window.cpp" "src/Frame.cpp" "src/stb_implementation.cpp" "src/kernel.cu")

set_target_properties(VoxelOptix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_precompile_headers(VoxelOptix PUBLIC pch.h)
target_link_libraries(VoxelOptix PRIVATE glm::glm glfw glad::glad CUDA::cudart)
