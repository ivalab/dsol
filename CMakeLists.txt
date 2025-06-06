cmake_minimum_required(VERSION 3.15)
project(dsol)

set(CMAKE_CXX_STANDARD 17)
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BUILD_ROS "Build ros" ON)
option(BUILD_PLAY "Build play" OFF)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_VECTORIZE "Build with -march=native" ON)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CC_TARGET_PREFIX sv)
include(CMakeHelpers)

add_library(sv_options INTERFACE)
include(CompilerWarnings)
enable_warnings(sv_options)
include(Sanitizers)
enable_sanitizers(sv_options)

find_package(
  catkin QUIET
  COMPONENTS roscpp
             tf2_ros
             tf2_eigen
             cv_bridge
             image_transport
             nav_msgs
             sensor_msgs
             visualization_msgs)

# this line must be put after catkin find_package
list(PREPEND CMAKE_PREFIX_PATH "$ENV{SLAM_OPENSOURCE_ROOT}/dsol")

find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(absl REQUIRED)
find_package(Boost REQUIRED) # circular buffer

find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
find_package(xtensor) # reading npy

find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

if(NOT catkin_FOUND)
  set(BUILD_ROS OFF)
  message(WARN "BUILD_ROS set to OFF because catkin not found.")
endif()

if(BUILD_ROS)
  catkin_package()
endif()

enable_testing()
add_subdirectory(sv)
