cmake_minimum_required(VERSION 3.10)
project(AIInferenceGateway CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# 查找依赖项
# ==============================================================================
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

find_library(MUDUO_BASE_LIB muduo_base PATHS /usr/local/lib)
find_library(MUDUO_NET_LIB muduo_net PATHS /usr/local/lib)
find_library(MUDUO_PROTOBUF_CODEC_LIB muduo_protobuf_codec PATHS /usr/local/lib)
find_library(MUDUO_PROTORPC_LIB muduo_protorpc PATHS /usr/local/lib)
find_library(MUDUO_PROTORPC_WIRE_LIB muduo_protorpc_wire PATHS /usr/local/lib)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib)
find_library(ZOOKEEPER_MT_LIB zookeeper_mt PATHS /usr/lib /usr/lib/aarch64-linux-gnu)

if(NOT MUDUO_BASE_LIB OR NOT MUDUO_NET_LIB OR NOT MUDUO_PROTOBUF_CODEC_LIB OR NOT MUDUO_PROTORPC_LIB OR NOT MUDUO_PROTORPC_WIRE_LIB)
    message(FATAL_ERROR "Muduo RPC libraries not found! Ensure Muduo protobuf/rpc components were installed correctly.")
endif()

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found!")
endif()

if(NOT ZOOKEEPER_MT_LIB)
    message(FATAL_ERROR "ZooKeeper multi-threaded client library not found!")
endif()

# ==============================================================================
# Protobuf 自动生成配置
# ==============================================================================
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/protos)
set(PROTO_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${PROTO_BUILD_DIR})
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRC_DIR}/inference.proto)

# ==============================================================================
# 目录配置
# ==============================================================================
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(PROJECT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
set(PROJECT_SCRIPT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/scripts)
set(PROJECT_THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

set(COMMON_INCLUDE_DIRS
    ${PROJECT_INCLUDE_DIR}
    ${PROJECT_THIRD_PARTY_DIR}
    ${PROTO_BUILD_DIR}
    /usr/local/include
    /usr/include
)

set(COMMON_RPC_LIBS
    ${Protobuf_LIBRARIES}
    ${MUDUO_NET_LIB}
    ${MUDUO_BASE_LIB}
    ${MUDUO_PROTOBUF_CODEC_LIB}
    ${MUDUO_PROTORPC_LIB}
    ${MUDUO_PROTORPC_WIRE_LIB}
    ${ONNXRUNTIME_LIB}
    ${Boost_LIBRARIES}
    ZLIB::ZLIB
    Threads::Threads
)

# ==============================================================================
# gateway 服务端
# ==============================================================================
add_executable(gateway
    ${PROJECT_SOURCE_DIR}/main.cpp
    ${PROJECT_SOURCE_DIR}/rpc_protocol.cpp
    ${PROJECT_SOURCE_DIR}/rpc_server.cpp
    ${PROJECT_SOURCE_DIR}/inference_service_impl.cpp
    ${PROJECT_SOURCE_DIR}/zk_client.cpp
    ${PROJECT_SOURCE_DIR}/service_registry.cpp
    ${PROJECT_SOURCE_DIR}/batching_queue.cpp
    ${PROJECT_SOURCE_DIR}/inference_engine.cpp
    ${PROTO_SRCS}
)
target_include_directories(gateway PRIVATE ${COMMON_INCLUDE_DIRS})
target_compile_definitions(gateway PRIVATE THREADED)
target_link_libraries(gateway
    ${COMMON_RPC_LIBS}
    ${ZOOKEEPER_MT_LIB}
)

# ==============================================================================
# 现有本地测试
# ==============================================================================
add_executable(test_inference
    ${PROJECT_TEST_DIR}/test_inference.cpp
    ${PROJECT_SOURCE_DIR}/inference_engine.cpp
)
target_include_directories(test_inference PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_inference
    ${ONNXRUNTIME_LIB}
    Threads::Threads
)

add_executable(test_batching
    ${PROJECT_TEST_DIR}/test_batching.cpp
    ${PROJECT_SOURCE_DIR}/batching_queue.cpp
    ${PROJECT_SOURCE_DIR}/inference_engine.cpp
)
target_include_directories(test_batching PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_batching
    ${ONNXRUNTIME_LIB}
    Threads::Threads
)

# ==============================================================================
# RPC 联通测试客户端
# ==============================================================================
add_executable(test_rpc_client
    ${PROJECT_SOURCE_DIR}/rpc_protocol.cpp
    ${PROJECT_TEST_DIR}/test_rpc_client.cpp
    ${PROTO_SRCS}
)
target_include_directories(test_rpc_client PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_rpc_client
    ${Protobuf_LIBRARIES}
    ${MUDUO_NET_LIB}
    ${MUDUO_BASE_LIB}
    ${MUDUO_PROTOBUF_CODEC_LIB}
    ZLIB::ZLIB
    Threads::Threads
)
