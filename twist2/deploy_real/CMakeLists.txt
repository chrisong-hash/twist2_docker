# cmake_minimum_required(VERSION 3.10)
# project(g1_loco_redis_bridge)

# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_definitions(-D_GNU_SOURCE)

# # Find Unitree SDK2
set(UNITREE_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../unitree_sdk2" CACHE PATH "Path to Unitree SDK2")

# if(NOT EXISTS ${UNITREE_SDK_PATH})
#     message(FATAL_ERROR "Unitree SDK2 not found at ${UNITREE_SDK_PATH}")
# endif()

# # Detect architecture
# if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
#     set(ARCH "x86_64")
# elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
#     set(ARCH "aarch64")
# else()
#     message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
# endif()

# message(STATUS "Building for architecture: ${ARCH}")

# # Include directories
include_directories(
    ${UNITREE_SDK_PATH}/include
    # ${UNITREE_SDK_PATH}/thirdparty/include
    # ${UNITREE_SDK_PATH}/thirdparty/include/ddscxx
    # ${UNITREE_SDK_PATH}/thirdparty/include/dds
    /usr/include/hiredis
)

# # Link directories
# link_directories(
#     ${UNITREE_SDK_PATH}/lib/${ARCH}
#     ${UNITREE_SDK_PATH}/thirdparty/lib
# )

# # Add executable
add_executable(loco_bridge
    g1_loco_redis_bridge.cpp
)

# Link libraries
target_link_libraries(loco_bridge
    unitree_sdk2
    hiredis
    # pthread
    # ddsc
    # ddscxx
)

# Install
# install(TARGETS loco_bridge
#     RUNTIME DESTINATION bin
# )

