/**
 * @file undistorter.cpp
 * @author Yanwei Du (yanwei.du@gatech.edu)
 * @brief None
 * @version 0.1
 * @date 02-07-2023
 * @copyright Copyright (c) 2023
 */

#include "sv/util/undistorter.h"

namespace sv {

EuRoCUndistorter::Parameters EuRoCUndistorter::Parameters::LoadFromYaml(
    const std::string& file) {
  Parameters p;
  cv::FileStorage fs(file, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cerr << "failed to open file: " << file << std::endl;
    return p;
  }

  // read extrinsic
  fs["T_BS"] >> p.bTc;
  // read intrinsics
  fs["intrinsics"] >> p.K;
  // // read distortion
  fs["distortion_coefficients"] >> p.D;

  p.camera_model = (std::string)fs["camera_model"];
  p.distortion_model = (std::string)fs["distortion_model"];
  // read resolution
  {
    cv::FileNodeIterator it = fs["resolution"].begin();
    int32_t width = (int32_t)*it;
    ++it;
    int32_t height = (int32_t)*it;
    p.resolution = cv::Size2i(width, height);
  }
  p.rate = (float)fs["rate_hz"];

  return p;
}

void EuRoCUndistorter::ReadIntrinsics(const std::string& calib0_file,
                                      const std::string& calib1_file) {
  auto calib0 = Parameters::LoadFromYaml(calib0_file);
  auto calib1 = Parameters::LoadFromYaml(calib1_file);

  // construct extrinsic of left_T_right
  const cv::Mat c1Tc0 = calib1.bTc.inv() * calib0.bTc;

  // call rectification
  cv::Mat R0, P0, R1, P1, Q;
  cv::stereoRectify(calib0.K,
                    calib0.D,
                    calib1.K,
                    calib1.D,
                    calib0.resolution,
                    c1Tc0.colRange(0, 3).rowRange(0, 3),
                    c1Tc0.rowRange(0, 3).col(3),
                    R0,
                    R1,
                    P0,
                    P1,
                    Q,
                    cv::CALIB_ZERO_DISPARITY,
                    0);  // 0 means valid pixels only

  // 1. Init undistortion map.
  cv::initUndistortRectifyMap(calib0.K,
                              calib0.D,
                              R0,
                              P0.colRange(0, 3).rowRange(0, 3),
                              calib0.resolution,
                              CV_32F,
                              calib0_M1_,
                              calib0_M2_);
  cv::initUndistortRectifyMap(calib1.K,
                              calib1.D,
                              R1,
                              P1.colRange(0, 3).rowRange(0, 3),
                              calib1.resolution,
                              CV_32F,
                              calib1_M1_,
                              calib1_M2_);

  // update rectified parameters
  cv::Mat K = P0.colRange(0, 3).rowRange(0, 3);

  // 2. Read baseline.
  baseline_ = std::fabs(P1.at<double>(0, 3) / P1.at<double>(0, 0));
  std::vector<double> intrin{K.at<double>(0, 0),
                             K.at<double>(1, 1),
                             K.at<double>(0, 2),
                             K.at<double>(1, 2),
                             baseline_};

  // 3. Construct intrinsic mat.
  data_ = cv::Mat(intrin, true);

  // 4. Read image size.
  image_size_ = calib0.resolution;
}

cv::Mat EuRoCUndistorter::RunImpl(const cv::Mat& image, int cam) const {
  cv::Mat rectified;
  if (cam == 0) {
    cv::remap(image, rectified, calib0_M1_, calib0_M2_, cv::INTER_LINEAR);
  } else {
    cv::remap(image, rectified, calib1_M1_, calib1_M2_, cv::INTER_LINEAR);
  }
  return rectified;
}

}  // namespace sv
