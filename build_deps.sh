#!/bin/bash

if [ -z "$SLAM_OPENSOURCE_ROOT" ]
then
    echo "Please configure SLAM workspace first, missing env variable $SLAM_OPENSOURCE_ROOT"
    exit
fi
mkdir $SLAM_OPENSOURCE_ROOT/dsol

# create 3rdparty in dsol
mkdir 3rdparty && cd 3rdparty

# # googletest
# git clone https://github.com/google/googletest.git
# cd googletest
# mkdir build && cd build
# cmake -DCMAKE_INSTALL_PREFIX=$SLAM_OPENSOURCE_ROOT/dsol ..
# make -j8 && make install
# cd ../..

# gflags
# git clone --branch v2.2.2 https://github.com/gflags/gflags.git
# cd gflags
# mkdir build && cd build
# cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$SLAM_OPENSOURCE_ROOT/dsol ..
# make -j8 && make install
# cd ../..

# glog
git clone --depth 1 --branch v0.6.0 https://github.com/google/glog.git
cd glog
cmake -S . -B build -G "Unix Makefiles" -DCMAKE_CXX_STANDARD=17 -DCMAKE_PREFIX_PATH=$SLAM_OPENSOURCE_ROOT/dsol -DCMAKE_INSTALL_PREFIX=$SLAM_OPENSOURCE_ROOT/dsol ..
cd build && make install
cd ../..

# fmt
git clone --depth 1 --branch 8.1.0 https://github.com/fmtlib/fmt.git
cd fmt
mkdir build && cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE -DCMAKE_CXX_STANDARD=17 -DFMT_TEST=False -DCMAKE_INSTALL_PREFIX=$SLAM_OPENSOURCE_ROOT/dsol ..
make install
cd ../..

# abseil
git clone --depth 1 --branch 20220623.0 https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
mkdir build && cd build
cmake -DABSL_BUILD_TESTING=OFF -DCMAKE_CXX_STANDARD=17 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$SLAM_OPENSOURCE_ROOT/dsol ..
make install
cd ../..

# sophus
git clone https://github.com/strasdat/Sophus.git
cd Sophus
mkdir build && cd build
git checkout 785fef3
cmake -DBUILD_SOPHUS_TESTS=OFF -DBUILD_SOPHUS_EXAMPLES=OFF -DCMAKE_CXX_STANDARD=17 -DCMAKE_INSTALL_PREFIX=$SLAM_OPENSOURCE_ROOT/dsol ..
make install
cd ../..

# benchmark
git clone --depth 1 --branch v1.6.2 https://github.com/google/benchmark.git
cd benchmark
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DCMAKE_INSTALL_PREFIX=$SLAM_OPENSOURCE_ROOT/dsol ..
cmake --build "build" --config Release --target install