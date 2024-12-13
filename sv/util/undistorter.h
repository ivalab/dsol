/**
 * @file undistorter.h
 * @author Yanwei Du (yanwei.du@gatech.edu)
 * @brief None
 * @version 0.1
 * @date 02-07-2023
 * @copyright Copyright (c) 2023
 */

#ifndef DSOL_UTIL_UNDISTORTER_H_
#define DSOL_UTIL_UNDISTORTER_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "sv/util/logging.h"

namespace sv {
/**
 * @brief
 *
 */
class Undistorter {
 public:
  using Ptr = std::shared_ptr<Undistorter>;
  using ConstPtr = std::shared_ptr<const Undistorter>;

  /**
   * @brief Construct a new Undistorter object
   *
   */
  Undistorter() = default;

  /**
   * @brief Destroy the Undistorter object
   *
   */
  virtual ~Undistorter() {}

  /**
   * @brief
   *
   * @param image
   * @param cam (0, 1)
   * @return cv::Mat
   */
  cv::Mat Run(const cv::Mat& image, int cam = 0) const {
    return RunImpl(image, cam);
  }

  /**
   * @brief Get the Intrin object
   *
   * @return const cv::Mat&
   */
  const cv::Mat& GetIntrin() const { return data_; }

  const cv::Size2i& GetImageSize() const { return image_size_; }

 protected:
  /**
   * @brief
   *
   * @param image
   * @param cam
   * @return cv::Mat
   */
  virtual cv::Mat RunImpl(const cv::Mat& image, int cam) const = 0;
  cv::Mat data_;
  cv::Size2i image_size_;
};

/**
 * @brief
 *
 */
class EuRoCUndistorter : public Undistorter {
 public:
  /**
   * @brief Construct a new Eu Ro C Undistorter object
   *
   * @param calib_dir
   */
  EuRoCUndistorter(const std::string& calib_dir) : Undistorter() {
    // all private/protected variables are assigned in the method
    ReadIntrinsics(calib_dir + "/calib0.yaml", calib_dir + "/calib1.yaml");
  }

  /**
   * @brief Destroy the Eu Ro C Undistorter object
   *
   */
  virtual ~EuRoCUndistorter() {}

 private:
  /**
   * @brief
   *
   * @param image
   * @param cam
   * @return cv::Mat
   */
  virtual cv::Mat RunImpl(const cv::Mat& image, int cam) const override;

  /**
   * @brief
   *
   * @param calib_file0
   * @param calib_file1
   */
  void ReadIntrinsics(const std::string& calib_file0,
                      const std::string& calib_file1);

  struct Parameters {
    cv::Mat bTc;  ///< extrinsic of the left camera(0) w.r.t body
    cv::Mat D;    ///< distortion
    cv::Mat K;    ///< intrinsic
    std::string camera_model = "UNKNOWN";      ///< "pinhole", e.t.c
    std::string distortion_model = "UNKNOWN";  ///< "radial-tangential", e.t.c
    cv::Size2i resolution = {0, 0};            ///< image size
    float rate = 0;                            ///< hz

    /**
     * @brief
     *
     * @param os
     * @param parameters
     * @return sdt::ostream&
     */
    friend std::ostream& operator<<(std::ostream& os, const Parameters& p) {
      os << "Parameters \n"
         << "bTc: \n"
         << p.bTc << "\n"
         << "D: " << p.D << "\n"
         << "K: \n"
         << p.K << "\n"
         << "camera_model: " << p.camera_model << "\n"
         << "distortion_model: " << p.distortion_model << "\n"
         << "resolution: " << p.resolution << "\n"
         << "rate: " << p.rate << std::endl;
      return os;
    }

    /**
     * @brief
     *
     * @param file
     * @return Parameters
     */
    static Parameters LoadFromYaml(const std::string& file);
  };

  cv::Mat calib0_M1_, calib0_M2_;
  cv::Mat calib1_M1_, calib1_M2_;
  double baseline_;
};

/**
 * @brief Create a Undistorter object
 *
 * @param dir
 * @param name
 * @return Undistorter::Ptr
 */
Undistorter::Ptr CreateUndistorter(const std::string& dir,
                                   const std::string& name) {
  std::string lowercase_name(name);
  std::transform(name.begin(), name.end(), lowercase_name.begin(), ::tolower);
  LOG(INFO) << lowercase_name;

  if (lowercase_name.find("euroc") != std::string::npos) {
    return std::make_shared<EuRoCUndistorter>(dir);
  }

  LOG(FATAL) << "Dataset (" << name << ") is not supported!";
  return nullptr;
}
}  // namespace sv

#endif  // DSOL_UTIL_UNDISTORTER_H_