cmake_minimum_required(VERSION 3.18)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

project(TRT-NNScaler LANGUAGES CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
        set(CMAKE_CUDA_ARCHITECTURES 62 72 87)
    else ()
        set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 89 90)
    endif ()
endif ()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8 /wd4996>")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8$<SEMICOLON>/wd4996>)
endif ()

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(TensorRT 8.6.0 REQUIRED COMPONENTS OnnxParser)

option(LOCAL_GFLAGS_GLOG "Use local GFlags and GLog" OFF)

if (LOCAL_GFLAGS_GLOG)
    set(GFLAGS_BUILD_SHARED_LIBS 0)
    set(GFLAGS_BUILD_STATIC_LIBS 1)
    set(GFLAGS_BUILD_gflags_LIB 0)
    set(GFLAGS_BUILD_gflags_nothreads_LIB 1)
    set(GFLAGS_INSTALL_STATIC_LIBS 0)
    set(GFLAGS_BUILD_TESTING 0)
    set(GFLAGS_BUILD_PACKAGING 0)
    add_subdirectory(gflags)

    set(WITH_GFLAGS 0)
    set(gflags_FOUND 1)
    set(HAVE_LIB_GFLAGS 1)
    set(gflags_NAMESPACE "gflags")
    set(BUILD_TESTING 0)
    set(BUILD_SHARED_LIBS 0)
    add_subdirectory(glog)

    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
else ()
    find_library(gflags REQUIRED)
    find_library(glog REQUIRED)
endif ()

add_subdirectory(libyuv)

add_library(reformat_cuda STATIC reformat/reformat_cuda.h reformat/reformat.cu)

add_library(reformat INTERFACE reformat/reformat.h reformat/reformat_cuda.h)
target_link_libraries(reformat INTERFACE CUDA::cudart reformat_cuda)

set(SOURCE_FILES
        md_view.h
        nn-scaler.h
        logging.h
        channel.h
        image_io.h
        optimize.cpp
        inference.cpp
        image_fpng.cpp
        workers.cpp
        main.cpp
        )

add_library(image_fpng STATIC fpng/fpng.h fpng/fpng.cpp)
target_compile_definitions(image_fpng PUBLIC FPNG_NO_STDIO)
target_compile_options(image_fpng PRIVATE -msse4.1 -mpclmul -fno-strict-aliasing)

add_executable(nn-scaler ${SOURCE_FILES})
target_compile_definitions(nn-scaler PRIVATE _UNICODE)
target_compile_options(nn-scaler PRIVATE -mavx2 -mfma)
target_link_libraries(nn-scaler PUBLIC
        image_fpng
        reformat

        yuv

        CUDA::cudart
        TensorRT::NvInfer
        TensorRT::OnnxParser

        gflags::gflags
        glog::glog
        )
target_include_directories(nn-scaler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libyuv/include)
set_target_properties(nn-scaler PROPERTIES
        OUTPUT_NAME NNScaler
        )

if (WIN32)
    option(USE_WIC_READER "Use WIC to read images" ON)
    option(USE_WUFFS_LIBJPEG_READER "Use Wuffs and libjpeg to read images" OFF)
else()
    option(USE_WUFFS_LIBJPEG_READER "Use Wuffs and libjpeg to read images" ON)
endif()

if (WIN32 AND USE_WIC_READER)
    add_library(image_wic OBJECT image_wic.cpp image_io.h)
    target_link_libraries(image_wic PUBLIC shlwapi windowscodecs)

    target_link_libraries(nn-scaler PUBLIC image_wic)
elseif (USE_WUFFS_LIBJPEG_READER)
    add_library(image_wuffs OBJECT image_io.h image_wuffs.cpp)
    find_package(PkgConfig)
    pkg_check_modules(JPEG REQUIRED IMPORTED_TARGET libjpeg)
    target_link_libraries(image_wuffs PUBLIC PkgConfig::JPEG)

    target_link_libraries(nn-scaler PUBLIC image_wuffs)
endif()

add_executable(inspector inspector.cpp)
target_link_libraries(inspector PUBLIC
        CUDA::cudart
        TensorRT::NvInfer
        gflags::gflags
        glog::glog
)
