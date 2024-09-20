cmake_minimum_required(VERSION 3.18)

cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0097 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

project(TRT-NNScaler LANGUAGES C CXX)

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
option(WITH_RPC_SERVICE "Build RPC service" ON)

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
        GIT_TAG v0.3.3
        GIT_PROGRESS   TRUE
)

FetchContent_Declare(
        abseil
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp
        GIT_TAG 20240722.0
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        OVERRIDE_FIND_PACKAGE
)

FetchContent_Declare(
        grpc
        GIT_REPOSITORY https://github.com/grpc/grpc
        GIT_TAG v1.66.1
        GIT_SHALLOW    TRUE
        GIT_PROGRESS   TRUE
        GIT_SUBMODULES "third_party/boringssl-with-bazel;third_party/cares/cares;third_party/re2;third_party/zlib;third_party/envoy-api;third_party/googleapis;third_party/opencensus-proto;third_party/xds;third_party/protobuf"
)

option(LIBYUV_USE_GITHUB_MIRROR "Use an UNOFFICIAL mirror for libyuv" OFF)

if (LIBYUV_USE_GITHUB_MIRROR)
    set(LIBYUV_GIT_REPOSITORY "https://github.com/lemenkov/libyuv")
else ()
    set(LIBYUV_GIT_REPOSITORY "https://chromium.googlesource.com/libyuv/libyuv")
endif ()
FetchContent_Declare(
        libyuv
        GIT_REPOSITORY ${LIBYUV_GIT_REPOSITORY}
        GIT_TAG 0d5a31eccb71d2745f0b5f231909f221743b0a5b
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
target_link_libraries(image_fpng PRIVATE CCCL::CCCL)

if(USE_WUFFS_LIBJPEG_READER)
    FetchContent_GetProperties(wuffs)
    if(NOT wuffs_POPULATED)
        FetchContent_Populate(wuffs)
    endif()

    add_library(image_wuffs OBJECT image_io.h image_wuffs.cpp)

    target_include_directories(image_wuffs PRIVATE ${wuffs_SOURCE_DIR}/release/c)

    find_package(PkgConfig)
    pkg_check_modules(JPEG REQUIRED IMPORTED_TARGET libjpeg)
    target_link_libraries(image_wuffs PRIVATE CCCL::CCCL)
    target_link_libraries(image_wuffs PUBLIC PkgConfig::JPEG)
endif()

if (LOCAL_DEPENDENCIES)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    if (WITH_RPC_SERVICE)
        set(ABSL_PROPAGATE_CXX_STD 1)

        FetchContent_GetProperties(abseil)
        if (NOT abseil_POPULATED)
            FetchContent_Populate(abseil)
            add_subdirectory(${abseil_SOURCE_DIR} ${abseil_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif ()

        set(protobuf_INSTALL OFF)
        set(protobuf_BUILD_TESTS OFF)
        set(protobuf_BUILD_CONFORMANCE OFF)
        set(protobuf_BUILD_EXAMPLES OFF)
        set(protobuf_BUILD_SHARED_LIBS OFF)
        set(protobuf_WITH_ZLIB OFF)
        set(protobuf_MSVC_STATIC_RUNTIME OFF)
        set(utf8_range_ENABLE_TESTS OFF)
        set(utf8_range_ENABLE_INSTALL OFF)

        set(gRPC_BUILD_TESTS OFF)
        set(gRPC_BUILD_CSHARP_EXT OFF)
        set(gRPC_INSTALL OFF)
        set(gRPC_USE_PROTO_LITE ON)
        set(gRPC_BUILD_GRPC_CSHARP_PLUGIN OFF)
        set(gRPC_BUILD_GRPC_NODE_PLUGIN OFF)
        set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN OFF)
        set(gRPC_BUILD_GRPC_PHP_PLUGIN OFF)
        set(gRPC_BUILD_GRPC_PYTHON_PLUGIN OFF)
        set(gRPC_BUILD_GRPC_RUBY_PLUGIN OFF)
        set(gRPC_PROTOBUF_PROVIDER "module")
        set(gRPC_ABSL_PROVIDER "package")
        set(gRPC_ZLIB_PROVIDER "module")

        FetchContent_GetProperties(grpc)
        if (NOT grpc_POPULATED)
            FetchContent_Populate(grpc)
            add_subdirectory(${grpc_SOURCE_DIR} ${grpc_BINARY_DIR} EXCLUDE_FROM_ALL)
            # Have totally no idea why I need to do this.
            include(${grpc_SOURCE_DIR}/third_party/protobuf/cmake/protobuf-generate.cmake)
        endif ()
        if (NOT TARGET gRPC::grpc++)
            add_library(gRPC::grpc++ ALIAS grpc++)
        endif ()
        if (NOT TARGET gRPC::grpc_cpp_plugin)
            add_executable(gRPC::grpc_cpp_plugin ALIAS grpc_cpp_plugin)
        endif ()
    endif()

    FetchContent_GetProperties(libyuv)
    if(NOT libyuv_POPULATED)
        FetchContent_Populate(libyuv)
        add_subdirectory(${libyuv_SOURCE_DIR} ${libyuv_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
else()
    if (WITH_RPC_SERVICE)
        find_package(absl REQUIRED)
        find_package(grpc REQUIRED)
    endif ()
    find_package(libyuv REQUIRED)
endif()

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(TensorRT 8.6.0 REQUIRED COMPONENTS OnnxParser)
find_package(CCCL 2 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

enable_language(CUDA)

# 997: virtual_function_decl_hidden  // suppress the warning of tensorrt hiding deprecated plugin methods
# 20208: double_for_long_double  // long double is required for user-defined literal. We are fine with float precision
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=997,20208>")

add_library(reformat_cuda STATIC reformat/reformat_cuda.h reformat/reformat.cu)

add_library(reformat INTERFACE reformat/reformat.h reformat/reformat_cuda.h)
target_link_libraries(reformat INTERFACE CUDA::cudart reformat_cuda CCCL::CCCL)

add_library(cmd_common STATIC
        md_view.h
        nn-scaler.h
        logging.h
        channel.h
        image_io.h
        timing.h
        cmd_common.h
        optimize.cpp
        inference.cpp
        image_fpng.cpp
        workers.cpp)

target_compile_definitions(cmd_common PUBLIC _UNICODE)
target_compile_options(cmd_common PRIVATE -mavx2 -mfma)
target_link_libraries(cmd_common PUBLIC
        image_fpng
        reformat

        yuv

        CCCL::CCCL
        CUDA::cudart
        TensorRT::NvInfer

        absl::flags
        absl::flags_parse
        absl::flags_usage

        absl::log
        absl::log_flags
        absl::log_initialize
)
target_include_directories(cmd_common PRIVATE ${libyuv_SOURCE_DIR}/include)

if (TensorRT_VERSION_STRING VERSION_LESS 9.0)
    add_library(trt_plugins STATIC
            plugins/plugin_base.h plugins/utils.h
            plugins/resize.cu
    )

    target_include_directories(trt_plugins PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/plugins)
    target_link_libraries(trt_plugins PUBLIC CUDA::cudart TensorRT::NvInfer CCCL::CCCL)
    target_link_libraries(cmd_common PUBLIC trt_plugins)
endif ()

if (WIN32 AND USE_WIC_READER)
    add_library(image_wic OBJECT image_wic.cpp image_io.h)
    target_link_libraries(image_wic PUBLIC shlwapi windowscodecs CCCL::CCCL)

    target_link_libraries(cmd_common PUBLIC image_wic)
elseif (USE_WUFFS_LIBJPEG_READER)
    target_link_libraries(cmd_common PUBLIC image_wuffs)
endif()

# Command line tool

add_executable(nn-scaler scaler_tool.cpp)
target_link_libraries(nn-scaler PUBLIC cmd_common TensorRT::OnnxParser)
set_target_properties(nn-scaler PROPERTIES
        OUTPUT_NAME NNScaler
)

# RPC Server

if (WITH_RPC_SERVICE)
    add_library(scaler-proto OBJECT scaler_server.proto)
    target_link_libraries(scaler-proto PUBLIC protobuf::libprotobuf gRPC::grpc++)

    set(PROTO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

    target_include_directories(scaler-proto PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(scaler-proto PUBLIC -Wno-inconsistent-missing-override)
    endif()

    protobuf_generate(
            TARGET scaler-proto
            PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

    protobuf_generate(
            TARGET scaler-proto
            LANGUAGE grpc
            GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
            PLUGIN "protoc-gen-grpc=\$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
            PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")

    add_executable(nn-scaler-server scaler_server.cpp)
    target_link_libraries(nn-scaler-server PUBLIC cmd_common scaler-proto)
    set_target_properties(nn-scaler-server PROPERTIES
            OUTPUT_NAME NNScalerServer
    )

    if (MSVC)
        target_link_libraries(nn-scaler-server PUBLIC delayimp TensorRT::OnnxParser)
        target_link_options(nn-scaler-server PUBLIC
                /DELAYLOAD:nvonnxparser.dll)
    elseif (UNIX)
        enable_language(ASM)
        add_library(_onnx_parser_shim OBJECT
                nvonnxparser_shim/libnvonnxparser.so.init.c
                nvonnxparser_shim/libnvonnxparser.so.tramp.S)
        target_link_libraries(nn-scaler-server PUBLIC _onnx_parser_shim)
    endif ()

    add_executable(nn-scaler-client scaler_client.cpp)
    target_link_libraries(nn-scaler-client PUBLIC
            scaler-proto

            absl::flags
            absl::flags_parse
            absl::flags_usage

            absl::log
            absl::log_flags
            absl::log_initialize
    )
    set_target_properties(nn-scaler-client PROPERTIES
            OUTPUT_NAME NNScalerClient
    )
endif ()
