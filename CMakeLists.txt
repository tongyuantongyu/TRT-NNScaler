cmake_minimum_required(VERSION 3.18)

cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0097 NEW)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

project(TRT-NNScaler LANGUAGES C CXX)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 80 86 89 90a 100a 120a)
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

add_library(image_fpng STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpng/src/fpng.h
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpng/src/fpng.cpp)
target_compile_definitions(image_fpng PUBLIC FPNG_NO_STDIO)
target_compile_options(image_fpng PRIVATE -msse4.1 -mpclmul -fno-strict-aliasing)
target_include_directories(image_fpng INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fpng/src)
target_link_libraries(image_fpng PRIVATE CCCL::CCCL)

if(USE_WUFFS_LIBJPEG_READER)
    add_library(image_wuffs OBJECT image_io.h image_wuffs.cpp)
    target_include_directories(image_wuffs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/wuffs/release/c)

    find_package(PkgConfig)
    pkg_check_modules(JPEG REQUIRED IMPORTED_TARGET libjpeg)
    target_link_libraries(image_wuffs PRIVATE CCCL::CCCL)
    target_link_libraries(image_wuffs PUBLIC PkgConfig::JPEG)
endif()

if (LOCAL_DEPENDENCIES)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

    if (WITH_RPC_SERVICE)
        set(ABSL_PROPAGATE_CXX_STD 1)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/abseil EXCLUDE_FROM_ALL SYSTEM)

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

        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/grpc EXCLUDE_FROM_ALL SYSTEM)
        # Have totally no idea why I need to do this.
        include(${grpc_SOURCE_DIR}/third_party/protobuf/cmake/protobuf-generate.cmake)
        if (NOT TARGET gRPC::grpc++)
            add_library(gRPC::grpc++ ALIAS grpc++)
        endif ()
        if (NOT TARGET gRPC::grpc_cpp_plugin)
            add_executable(gRPC::grpc_cpp_plugin ALIAS grpc_cpp_plugin)
        endif ()
    endif()

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/libyuv EXCLUDE_FROM_ALL SYSTEM)
else()
    if (WITH_RPC_SERVICE)
        find_package(absl REQUIRED)
        find_package(grpc REQUIRED)
    endif ()
    find_package(libyuv REQUIRED)
endif()

find_package(CUDAToolkit 12.0 REQUIRED COMPONENTS cudart)
find_package(TensorRT 8.6.0 REQUIRED COMPONENTS OnnxParser)
find_package(CCCL 2 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

enable_language(CUDA)

# 997: virtual_function_decl_hidden  // suppress the warning of tensorrt hiding deprecated plugin methods
# 20208: double_for_long_double  // long double is required for user-defined literal. We are fine with float precision
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=997,20208>")

option(CUDA_DEVICE_DEBUG "Enable CUDA Device Debug" OFF)
if (CUDA_DEVICE_DEBUG)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-G>")
endif()

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
if (LOCAL_DEPENDENCIES)
    target_include_directories(cmd_common PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libyuv/include)
endif()

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
