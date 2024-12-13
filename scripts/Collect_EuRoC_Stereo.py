#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@file Collect_evaluation_result.py
@author Yanwei Du (duyanwei0702@gmail.com)
@date 07-20-2022
@version 1.0
@license Copyright (c) 2022
@desc None
"""

import json
import numpy as np
import os
import zipfile


DATA_ROOT = "/mnt/DATA/Datasets/EuRoC/"
SeqNameList = [
    "MH_01_easy",
    "MH_02_easy",
    "MH_03_medium",
    "MH_04_difficult",
    "MH_05_difficult",
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]
RESULT_ROOT = os.path.join(os.environ["SLAM_RESULT"], "dsol/EuRoC/Benchmark")
NumRepeating = 1
SleepTime = 1  # 10 # 25 # second
FeaturePool = [800]
SpeedPool = [1.0]  # , 2.0, 3.0, 4.0, 5.0]  # x
ResultFile = ["AllFrameTrajectory"]
RMSE_THRESH = 10.0  # m
TRACKING_THRESH = 0.6
DataFreq = 20.0
SeqDuration = {
    "MH_01_easy": 182,
    "MH_02_easy": 150,
    "MH_03_medium": 132,
    "MH_04_difficult": 99,
    "MH_05_difficult": 111,
    "V1_01_easy": 143,
    "V1_02_medium": 84,
    "V1_03_difficult": 105,
    "V2_01_easy": 112,
    "V2_02_medium": 115,
    "V2_03_difficult": 115,
}


# ----------------------------------------------------------------------------------------------------------------------


for feature in FeaturePool:

    feature_str = str(feature)
    result_1st_dir = os.path.join(RESULT_ROOT, feature_str)

    rmse_table = np.full((len(ResultFile), len(SpeedPool), len(SeqNameList), NumRepeating), -1.0)

    mean_timing_table = np.full((len(SpeedPool), len(SeqNameList), NumRepeating), -1.0)
    median_timing_table = np.full_like(mean_timing_table, -1.0)

    # loop over play speed
    for j, speed in enumerate(SpeedPool):

        speed_str = str(speed)
        result_dir = os.path.join(result_1st_dir, "Fast" + speed_str)

        # loop over num of repeating
        for l in range(NumRepeating):

            experiment_dir = os.path.join(result_dir, "Round" + str(l + 1))

            # loop over sequence
            for k, sname in enumerate(SeqNameList):

                # collect rmse
                for i, result_name in enumerate(ResultFile):
                    result = os.path.join(experiment_dir, sname + "_" + result_name + ".zip")
                    if not os.path.exists(result):
                        continue
                    # read
                    # print(result)
                    with zipfile.ZipFile(result, "r") as z:
                        with z.open("stats.json") as f:
                            data = f.read()
                            error = json.loads(data)["rmse"]
                            if error < RMSE_THRESH:
                                rmse_table[i, j, k, l] = error
                            else:
                                print(
                                    f"tracking failed: Feature {feature}, Speed {speed}, Seq {sname}, Round {l+1}, TR {error}"
                                )
                file_est = os.path.join(experiment_dir, sname + "_AllFrameTrajectory.txt")
                est_poses = np.loadtxt(file_est)
                if True:
                    tracking_ratio = est_poses.shape[0] / (SeqDuration[sname] * DataFreq)
                else:
                    tracking_ratio = (est_poses[-1, 0] - est_poses[0, 0]) / SeqDuration[sname]
                if tracking_ratio < TRACKING_THRESH:
                    rmse_table[i, j, k, l] = -1.0
                    print(
                        f"tracking failed: Feature {feature}, Speed {speed}, Seq {sname}, Round {l+1}, TR {tracking_ratio}"
                    )
                # collect tracking failure info
                file_latency = os.path.join(experiment_dir, sname + "_Log_Latency.txt")
                latency = np.sum(np.loadtxt(file_latency)[:, 1:], axis=-1)
                # record latency
                mean_timing_table[j, k, l] = np.mean(latency)
                median_timing_table[j, k, l] = np.median(latency)

    mean_timing_table = mean_timing_table.reshape(-1, NumRepeating)
    median_timing_table = median_timing_table.reshape(-1, NumRepeating)
    # save rmse
    for i, result_name in enumerate(ResultFile):
        output = os.path.join(result_1st_dir, result_name + ".txt")
        mn, nn, pn, qn = rmse_table.shape
        # the extra two column stores mean and median rmse
        cur_table = np.full((nn * pn, qn + 4), -1.0)
        cur_table[:, 0:qn] = rmse_table[i, :, :, :].reshape(nn * pn, qn)
        for row in range(cur_table.shape[0]):
            indices = cur_table[row, :] > 0.0
            if np.sum(indices) == qn:  # make sure every sequence succeeds
                cur_table[row, qn] = np.mean(cur_table[row, indices])
                cur_table[row, qn + 1] = np.median(cur_table[row, indices])
                cur_table[row, qn + 2] = np.mean(mean_timing_table[row, indices[:-4]])
                cur_table[row, qn + 3] = np.median(median_timing_table[row, indices[:-4]])
            # else:
            # cur_table[row, qn] = -1
            # cur_table[row, qn + 1] = -1
        np.savetxt(output, cur_table, fmt="%.6f", delimiter=",")

        # for visualization
        output = os.path.join(result_1st_dir, result_name + "_vis.txt")
        final_table = np.full((pn + 4, nn), -1.0)
        for col in range(final_table.shape[1]):
            start_ind = 0 + pn * col
            end_ind = start_ind + pn
            final_table[0:pn, col] = cur_table[start_ind:end_ind, qn]
            temp_timing = cur_table[start_ind:end_ind, qn + 2 :]
            indices = final_table[:, col] > 0.0
            if np.sum(indices) == 0:
                continue
            final_table[pn, col] = np.mean(final_table[indices, col])
            final_table[pn + 1, col] = np.sum(indices) / pn
            final_table[pn + 2, col] = np.mean(temp_timing[indices[:-4], 0])
            final_table[pn + 3, col] = np.median(temp_timing[indices[:-4], 1])
        np.savetxt(output, np.transpose(final_table), fmt="%.6f", delimiter=",")
