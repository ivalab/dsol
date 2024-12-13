#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file Run_EuRoC_Stereo.py
@author Yanwei Du (duyanwei0702@gmail.com)
@date 08-09-2022
@version 1.0
@license Copyright (c) 2022
@desc None
"""

# This script is to run all the experiments in one program

import os
import subprocess
import time

DATA_ROOT = "/mnt/DATA/Datasets/EuRoC/"
SeqNameList = [
    "MH_01_easy",
    #  'MH_02_easy', 'MH_03_medium',
    # 'MH_04_difficult', 'MH_05_difficult',
    # 'V1_01_easy', 'V1_02_medium', 'V1_03_difficult',
    # 'V2_01_easy', 'V2_02_medium',
    # 'V2_03_difficult',
]
RESULT_ROOT = os.path.join(os.environ["SLAM_RESULT"], "dsol/EuRoC/Baseline/CoreDumpedDebug/")
NumRepeating = 1
SleepTime = 5  # 10 # 25 # second
FeaturePool = [800]  # [100, 150, 200] # , 800, 1000]
# FeaturePool = [800]
SpeedPool = [1.0]  # , 2.0, 3.0, 4.0, 5.0]  # , 2.0, 3.0, 4.0, 5.0] # x
EnableViewer = 0
EnableLogging = 1
EnableStepDebugging = 0
DSOL_PATH = os.path.join(os.path.expanduser("~"), "catkin_ws")

# enter svo directory
# cmd = 'cd ' + SVO_PATH
# subprocess.call(cmd, shell=True)
# cmd = 'source devel/setup.bash'
# subprocess.call(cmd, shell=True)
# ----------------------------------------------------------------------------------------------------------------------


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ALERT = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# predefine grid size from feature number
feature2cell = {
    100: "51",
    150: "45",
    200: "39",
    400: "30",
    600: "24",
    800: "21",
    1000: "19",
    1600: "14",
    2000: "13",
}

for feature in FeaturePool:

    feature_str = str(feature)
    result_1st_dir = os.path.join(RESULT_ROOT, feature_str)

    # loop over play speed
    for speed in SpeedPool:

        speed_str = str(speed)
        result_dir = os.path.join(result_1st_dir, "Fast" + speed_str)

        # create result dir first level
        cmd_mkdir = "mkdir -p " + result_dir
        subprocess.call(cmd_mkdir, shell=True)

        # loop over num of repeating
        for iteration in range(NumRepeating):

            # create result dir second level
            experiment_dir = os.path.join(result_dir, "Round" + str(iteration + 1))
            cmd_mkdir = "mkdir -p " + experiment_dir
            subprocess.call(cmd_mkdir, shell=True)

            # loop over sequence
            for sn, sname in enumerate(SeqNameList):

                print(
                    bcolors.ALERT
                    + "===================================================================="
                    + bcolors.ENDC
                )

                SeqName = SeqNameList[sn]
                print(
                    bcolors.OKGREEN
                    + f"Seq: {SeqName}; Feature: {feature_str}; Speed: {speed_str}; Round: {str(iteration + 1)};"
                )

                file_data = os.path.join(DATA_ROOT, SeqName)
                # file_timestamp = os.path.join(file_data, 'times.txt')
                file_traj = os.path.join(experiment_dir, SeqName)
                data_hz = str(20.0)
                # file_log = '> ' + file_traj + '_logging.txt' if EnableLogging else ''

                # compose cmd
                cmd_slam = (
                    "roslaunch dsol dsol_data.launch"
                    + " vis:="
                    + str(EnableViewer)
                    + " freq:="
                    + data_hz
                    + " save:="
                    + file_traj
                    + " wait_ms:="
                    + str(EnableStepDebugging)
                    + " cell_size:="
                    + feature2cell[feature]
                    + " min_log_level:=0"
                    + " also_log_to_console:=true"
                    + " verbose_logging:=0"
                    + " data_dir:="
                    + file_data
                )

                print(bcolors.WARNING + "cmd_slam: \n" + cmd_slam + bcolors.ENDC)

                print(bcolors.OKGREEN + "Launching SLAM" + bcolors.ENDC)
                # proc_slam = subprocess.Popen(cmd_slam, shell=True)  # starts a new shell and runs the result
                subprocess.call(cmd_slam, shell=True)
                # proc_slam.terminate()

                # launch bags
                # time.sleep(SleepTime)
                # cmd_bag = \
                # 'rosbag play ' + file_data + \
                # ' -r ' + speed_str
                # print(bcolors.WARNING + "Launching rosbag: \n" + cmd_bag + bcolors.ENDC)
                # subprocess.call(cmd_bag, shell=True)

                print(bcolors.OKGREEN + "Finished" + bcolors.ENDC)
                # subprocess.call('rosnode kill dsol_data', shell=True)
                time.sleep(SleepTime)
