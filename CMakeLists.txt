cmake_minimum_required (VERSION 3.18 FATAL_ERROR)
project("VoxelOptix" LANGUAGES CXX CUDA)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 50)

if (WIN32)
	find_path(searched_OptiX_INSTALL_DIR
		NAME include/optix.h
		PATHS
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.6.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
		"C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
	)
	mark_as_advanced(searched_OptiX_INSTALL_DIR)
  set(OPTIX7_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
else()
  set(OPTIX7_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
endif()

find_package(glfw3 REQUIRED)
find_package(glad REQUIRED)
find_package(glm REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(OptixPTX OBJECT "src/render/Programs.cu")
set_property(TARGET OptixPTX PROPERTY CUDA_PTX_COMPILATION ON)
target_link_libraries(OptixPTX PRIVATE CUDA::cudart)
target_include_directories(OptixPTX PRIVATE ${OPTIX7_DIR}/include)

add_executable (VoxelOptix 
    "src/main.cpp" 
    "pch.h" 
    "src/App.cpp" 
    "src/display/Window.cpp" 
    "src/display/Frame.cpp" 
    "src/misc/stb_implementation.cpp" 
    "src/render/Renderer.cpp"
    "src/render/Camera.cpp" 
    "src/display/Menu.cpp")

target_include_directories(VoxelOptix PRIVATE ${OPTIX7_DIR}/include)
target_precompile_headers(VoxelOptix PUBLIC pch.h)
target_link_libraries(VoxelOptix PRIVATE glm::glm glfw glad::glad CUDA::cudart)