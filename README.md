## Ubuntu 20.04 & ROS Noetic

### Build

```bash
mkdir -p ~/closedloop_ws/src

# Configure ros workspace.
cd ~/closedloop_ws
catkin build -j2 --no-status -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17

# Clone dsol.
cd src
git clone https://github.gatech.edu/RoboSLAM/dsol.git

# Build 3rdparty.
# Assuming $SLAM_OPENSOURCE_ROOT variable is defined, the 3rdparty library
# will be installed under "$SLAM_OPENSOURCE_ROOT/dsol/".

sudo apt install libgflags-dev libgmock-dev libgtest-dev

cd dsol
./build_deps.sh

# Build dsol.

cd ~/catkin_ws/src/dsol
catkin build -j2 --no-status -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 --this
```

### Run

```bash

# Please make sure roscore is running !!!

cd scripts

source ~/catkin_ws/devel/setup.bash

# Remember to configure the variables in the script.
python Run_EuRoC_Stereo_ROS.py

# Evaluation.
python Evaluate_EuRoC_Stereo.py

```

### Debug
If the error of any library missing is reported, just export the library path:
```bash
export LD_LIBRARY_PATH=${SLAM_OPENSOURCE_ROOT}/dsol/lib/:$LD_LIBRARY_PATH
```

---
---

# 🛢️ DSOL: Direct Sparse Odometry Lite

## Reference

Chao Qu, Shreyas S. Shivakumar, Ian D. Miller, Camillo J. Taylor

https://arxiv.org/abs/2203.08182

https://youtu.be/yunBYUACUdg

## Datasets

VKITTI2 https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

KITTI Odom 

TartanAir https://theairlab.org/tartanair-dataset/

Sample realsense data at

https://www.dropbox.com/s/bidng4gteeh8sx3/20220307_172336.bag?dl=0

https://www.dropbox.com/s/e8aefoji684dp3r/20220307_171655.bag?dl=0

Calib for realsense data is 

```
393.4910888671875 393.4910888671875 318.6263122558594 240.12942504882812 0.095150406
```
Put this in `calib.txt` and put it in the same folder of the realsense dataset generated by the python file.

## Build

This is a ros package, just put in a catkin workspace and build the workspace.

## Run
Open rviz using the config in `launch/dsol.rviz`

```
roslaunch dsol dsol_data.launch
```

See launch files for more details on different datasets.

See config folder for details on configs.

To run multithread and show timing every 5 frames do
```
roslaunch dsol dsol_data.launch tbb:=1 log:=5
```

## Dependencies and Install instructions

See CMakeLists.txt for dependencies. You may also check our [Github Action build
file](https://github.com/versatran01/dsol/blob/main/.github/workflows/build.yaml) for instructions on how to build DSOL in Ubuntu 20.04 with ROS Noetic.

## Disclaimer

For reproducing the results in the paper, place use the `iros22` branch.

This is the open-source version, advanced features are not included.
