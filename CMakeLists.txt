cmake_minimum_required(VERSION 3.18)

cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0097 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

project(TRT-NNScaler LANGUAGES CXX)

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

option(LOCAL_DEPENDENCIES "Build dependencies locally" OFF)

if (WIN32)
    option(USE_WIC_READER "Use WIC to read images" ON)
    option(USE_WUFFS_LIBJPEG_READER "Use Wuffs and libjpeg to read images" OFF)
else()
    option(USE_WUFFS_LIBJPEG_READER "Use Wuffs and libjpeg to read images" ON)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
        fpng
        GIT_REPOSITORY https://github.com/richgel999/fpng
        GIT_TAG        357d3a6b73c769e190dafa254895be8e958cfd13
        GIT_PROGRESS   TRUE
)

FetchContent_Declare(
        wuffs
        GIT_REPOSITORY https://github.com/google/wuffs-mirror-release-c
        GIT_TAG        e3f919ccfe3ef542cfc983a82146070258fb57f8
        GIT_PROGRESS   TRUE
)

FetchContent_Declare(
        abseil
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp
        GIT_TAG        20230125.3
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
)

FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf
        GIT_TAG        v23.1
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        GIT_SUBMODULES "third_party/jsoncpp"
)

FetchContent_Declare(
        grpc
        GIT_REPOSITORY https://github.com/grpc/grpc
        GIT_TAG        v1.55.0
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        GIT_SUBMODULES "third_party/boringssl-with-bazel;third_party/cares/cares;third_party/libuv;third_party/re2;third_party/zlib;third_party/envoy-api;third_party/googleapis;third_party/opencensus-proto;third_party/xds"
)

FetchContent_Declare(
        libyuv
        GIT_REPOSITORY https://chromium.googlesource.com/libyuv/libyuv
        GIT_TAG        8670bcf17faca69603d2d49fa6ebd0e45123471f
        GIT_PROGRESS   TRUE
)

FetchContent_GetProperties(fpng)
if(NOT fpng_POPULATED)
    FetchContent_Populate(fpng)
endif()

add_library(image_fpng STATIC ${fpng_SOURCE_DIR}/src/fpng.h ${fpng_SOURCE_DIR}/src/fpng.cpp)
target_compile_definitions(image_fpng PUBLIC FPNG_NO_STDIO)
target_compile_options(image_fpng PRIVATE -msse4.1 -mpclmul -fno-strict-aliasing)
target_include_directories(image_fpng INTERFACE ${fpng_SOURCE_DIR}/src)

if(USE_WUFFS_LIBJPEG_READER)
    FetchContent_GetProperties(wuffs)
    if(NOT wuffs_POPULATED)
        FetchContent_Populate(wuffs)
    endif()

    add_library(image_wuffs OBJECT image_io.h image_wuffs.cpp)

    target_include_directories(image_wuffs PRIVATE ${wuffs_SOURCE_DIR}/release/c)

    find_package(PkgConfig)
    pkg_check_modules(JPEG REQUIRED IMPORTED_TARGET libjpeg)
    target_link_libraries(image_wuffs PUBLIC PkgConfig::JPEG)
endif()

if (LOCAL_DEPENDENCIES)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    set(ABSL_PROPAGATE_CXX_STD 1)

    FetchContent_GetProperties(abseil)
    if(NOT abseil_POPULATED)
        FetchContent_Populate(abseil)
        add_subdirectory(${abseil_SOURCE_DIR} ${abseil_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    set(protobuf_INSTALL 0)
    set(protobuf_BUILD_TESTS 0)
    set(protobuf_BUILD_CONFORMANCE 0)
    set(protobuf_BUILD_EXAMPLES 0)
    set(protobuf_BUILD_SHARED_LIBS 0)
    set(utf8_range_ENABLE_TESTS 0)
    set(utf8_range_ENABLE_INSTALL 0)

    FetchContent_GetProperties(protobuf)
    if(NOT protobuf_POPULATED)
        FetchContent_Populate(protobuf)
        add_subdirectory(${protobuf_SOURCE_DIR} ${protobuf_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    set(gRPC_BUILD_TESTS 0)
    set(gRPC_BUILD_CSHARP_EXT 0)
    set(gRPC_INSTALL 0)
    set(gRPC_USE_PROTO_LITE 1)
    set(gRPC_BUILD_GRPC_CSHARP_PLUGIN 0)
    set(gRPC_BUILD_GRPC_NODE_PLUGIN 0)
    set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN 0)
    set(gRPC_BUILD_GRPC_PHP_PLUGIN 0)
    set(gRPC_BUILD_GRPC_PYTHON_PLUGIN 0)
    set(gRPC_BUILD_GRPC_RUBY_PLUGIN 0)
    set(gRPC_PROTOBUF_PROVIDER "package")
    set(gRPC_ABSL_PROVIDER "package")

    FetchContent_GetProperties(grpc)
    if(NOT grpc_POPULATED)
        FetchContent_Populate(grpc)
        add_subdirectory(${grpc_SOURCE_DIR} ${grpc_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    FetchContent_GetProperties(libyuv)
    if(NOT libyuv_POPULATED)
        FetchContent_Populate(libyuv)
        add_subdirectory(${libyuv_SOURCE_DIR} ${libyuv_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
else()
    find_library(abseil REQUIRED)
    find_library(grpc REQUIRED)
    find_library(libyuv REQUIRED)
endif()

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(TensorRT 8.6.0 REQUIRED COMPONENTS OnnxParser)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

enable_language(CUDA)
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

        absl::flags
        absl::flags_parse
        absl::flags_usage

        absl::log
        absl::log_flags
        absl::log_initialize
        )
target_include_directories(nn-scaler PRIVATE ${libyuv_SOURCE_DIR}/include)
set_target_properties(nn-scaler PROPERTIES
        OUTPUT_NAME NNScaler
        )

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

        absl::flags
        absl::flags_parse
        absl::flags_usage

        absl::log
        absl::log_flags
        absl::log_initialize
)
