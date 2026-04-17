cmake_minimum_required(VERSION 3.10)
project(AIInferenceGateway CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# 查找依赖项
# ==============================================================================

# 1. Protobuf (必需)
find_package(Protobuf REQUIRED)

# Protobuf 自动生成配置
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROTO_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${PROTO_BUILD_DIR})

# 根据 proto 文件生成 .pb.cc 和 .pb.h 源文件
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRC_DIR}/inference.proto)

# 2. Muduo 网络库 (必需)
# 在我们的环境中，Muduo 安装在 /usr/local 目录下
find_library(MUDUO_BASE_LIB muduo_base PATHS /usr/local/lib)
find_library(MUDUO_NET_LIB muduo_net PATHS /usr/local/lib)

if(NOT MUDUO_BASE_LIB OR NOT MUDUO_NET_LIB)
    message(FATAL_ERROR "Muduo library not found! Ensure it was compiled and installed correctly.")
endif()

# 3. ONNX Runtime (必需)
# 在我们的环境中，ONNX Runtime 头文件和库文件安装在 /usr/local 目录下
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found!")
endif()

# 4. Boost (Muduo 的依赖库)
find_package(Boost REQUIRED)

# 5. Zlib (Protobuf/Muduo 的依赖库)
find_package(ZLIB REQUIRED)

# 6. Threads (线程库)
find_package(Threads REQUIRED)

# ==============================================================================
# 包含目录
# ==============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROTO_BUILD_DIR}
    /usr/local/include
    /usr/include
)

# ==============================================================================
# 目标可执行程序 (供未来开发阶段使用的占位代码)
# ==============================================================================

# 注意：在 Phase 2 及以后的阶段中，我们将逐步取消注释并完善此部分
# add_executable(gateway main.cpp ${PROTO_SRCS})
# target_link_libraries(gateway
#     ${Protobuf_LIBRARIES}
#     ${MUDUO_NET_LIB}
#     ${MUDUO_BASE_LIB}
#     ${ONNXRUNTIME_LIB}
#     ${Boost_LIBRARIES}
#     ZLIB::ZLIB
#     Threads::Threads
# )

add_executable(test_inference test_inference.cpp inference_engine.cpp)
target_link_libraries(test_inference
    ${ONNXRUNTIME_LIB}
    Threads::Threads
)