cmake_minimum_required(VERSION 3.10)
project(AIInferenceGateway CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# Find Dependencies
# ==============================================================================

# 1. Protobuf (REQUIRED)
find_package(Protobuf REQUIRED)

# Protobuf auto-generation
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROTO_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${PROTO_BUILD_DIR})

# Generate .pb.cc and .pb.h files from proto file
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRC_DIR}/inference.proto)

# 2. Muduo Networking Library (REQUIRED)
# In our environment, Muduo is installed under /usr/local
find_library(MUDUO_BASE_LIB muduo_base PATHS /usr/local/lib)
find_library(MUDUO_NET_LIB muduo_net PATHS /usr/local/lib)

if(NOT MUDUO_BASE_LIB OR NOT MUDUO_NET_LIB)
    message(FATAL_ERROR "Muduo library not found! Ensure it was compiled and installed correctly.")
endif()

# 3. ONNX Runtime (REQUIRED)
# In our environment, ONNX Runtime headers and libs are in /usr/local
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found!")
endif()

# 4. Boost (Muduo dependency)
find_package(Boost REQUIRED)

# 5. Zlib (Protobuf/Muduo dependency)
find_package(ZLIB REQUIRED)

# 6. Threads
find_package(Threads REQUIRED)

# ==============================================================================
# Include Directories
# ==============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROTO_BUILD_DIR}
    /usr/local/include
    /usr/include
)

# ==============================================================================
# Target Executables (Placeholders for future phases)
# ==============================================================================

# NOTE: Phase 2 and beyond will uncomment and expand this section
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
