#include "sv/util/dataset.h"

#include <absl/strings/match.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_replace.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <glog/logging.h>

#include <Eigen/Geometry>
#include <filesystem>
#include <fstream>
#include <istream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>  // imread
#include <opencv2/imgproc.hpp>    // threashold

#ifndef XTENSOR_FOUND
#define XTENSOR_FOUND 0
#endif

// Conditionallly include xtensor if package found
#if XTENSOR_FOUND
#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>
#endif

namespace sv {

namespace fs = std::filesystem;

using SO3d = Sophus::SO3d;
using SE3d = Sophus::SE3d;
using Matrix3d = Eigen::Matrix3d;
using Vector3d = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;
using RowMat44d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;

namespace {

constexpr int kPoseDim = 7;
constexpr int kIntrinDim = 4;
constexpr int kTransformDim = 16;

template <typename T>
void VectorExtend(std::vector<T>& v1, const std::vector<T>& v2) {
  v1.reserve(v1.size() + v2.size());
  v1.insert(v1.end(), v2.begin(), v2.end());
}

void SortStrsAsInts(std::vector<std::string>& strings) {
  std::sort(strings.begin(), strings.end(), [](const auto& s1, const auto& s2) {
    if (s1.size() != s2.size()) return (s1.size() < s2.size());
    return s1 < s2;
  });
}

/// @brief Read stereo data
bool ReadStereoData(const fs::path& left_path,
                    const fs::path& right_path,
                    const std::string& ext,
                    std::vector<std::string>& files) {
  // Read left images
  const auto left_files = GetFiles(left_path, ext, true);

  if (left_files.empty()) {
    LOG(WARNING) << "No files found in right_path: " << left_path;
  } else {
    VectorExtend(files, left_files);
  }

  // Read right images
  const auto right_files = GetFiles(right_path, ext, true);

  if (right_files.empty()) {
    LOG(WARNING) << "No files found in right_path: " << right_path;
  } else {
    // Otherwise must have the same size as left files
    CHECK_EQ(left_files.size(), right_files.size());
    VectorExtend(files, right_files);
  }

  return left_files.size() == right_files.size();
}

cv::Mat MakeStereoIntrins(const cv::Mat& intrin) {
  CHECK_EQ(intrin.cols, 5);
  CHECK_EQ(intrin.type(), CV_64FC1);
  cv::Mat intrins = cv::Mat::zeros(2, 5, CV_64FC1);
  intrin.copyTo(intrins.row(0));
  intrin.copyTo(intrins.row(1));
  intrins.at<double>(1, 4) *= -1;
  return intrins;
}

}  // namespace

void MatToSE3d(const cv::Mat& mat, SE3d& tf) {
  CHECK(mat.isContinuous());
  CHECK_EQ(mat.type(), CV_64FC1);
  CHECK_EQ(mat.total(), kPoseDim);

  tf.translation() = Eigen::Map<const Vector3d>(mat.ptr<double>(0));
  // 0  1  2  3  4  5  6
  // tx ty tz qx qy qz qw
  const Quaterniond q(mat.at<double>(6),
                      mat.at<double>(3),
                      mat.at<double>(4),
                      mat.at<double>(5));
  tf.setQuaternion(q.normalized());
}

SE3d SE3dFromMat(const cv::Mat& mat) {
  if (mat.empty()) return {};

  CHECK(mat.isContinuous());
  CHECK_EQ(mat.type(), CV_64FC1);
  CHECK_EQ(mat.total(), kTransformDim);

  Eigen::Map<const RowMat44d> tf(mat.ptr<double>(0));
  return {tf.topLeftCorner<3, 3>(), tf.topRightCorner<3, 1>()};
}

std::vector<std::string> GetFiles(const std::string& dir,
                                  std::string_view ext,
                                  bool sort) {
  std::vector<fs::path> paths;
  std::vector<std::string> files;

  const fs::path pdir{dir};

  if (!fs::exists(pdir)) {
    LOG(WARNING) << fmt::format("[{}] does not exist", pdir.string());
    return files;
  }

  if (!fs::is_directory(pdir)) {
    LOG(WARNING) << fmt::format("[{}] is not a directory", pdir.string());
    return files;
  }

  for (const auto& entry : fs::directory_iterator(pdir)) {
    if (!fs::is_regular_file(entry.path())) continue;
    if (!ext.empty() && entry.path().extension() != ext) continue;
    paths.push_back(entry.path());
  }

  // sort by filename only
  if (sort) {
    std::sort(
        paths.begin(), paths.end(), [](const fs::path& p1, const fs::path& p2) {
          return p1.filename() < p2.filename();
        });
  }

  // convert to string
  files.reserve(paths.size());
  std::transform(paths.cbegin(),
                 paths.cend(),
                 std::back_inserter(files),
                 [](const fs::path& p) { return p.string(); });

  return files;
}

cv::Mat CvReadImage(const std::string& file) {
  auto image = cv::imread(file, cv::IMREAD_UNCHANGED);
  if (image.empty()) {
    LOG(WARNING) << "Failed to read color image: " << file;
  }
  return image;
}

cv::Mat CvReadDepth(const std::string& file, double div_factor) {
  auto depth = cv::imread(file, cv::IMREAD_ANYDEPTH);

  if (depth.empty()) {
    LOG(WARNING) << "Failed to read depth image: " << file;
    return depth;
  }

  if (div_factor != 1.0) {
    CHECK_GT(div_factor, 0);
    depth.convertTo(depth, CV_32FC1, 1.0 / div_factor);
  } else {
    CHECK_EQ(depth.type(), CV_32FC1);
  }

  return depth;
}

cv::Mat ThresholdDepth(const cv::Mat& depth,
                       double min_depth,
                       double max_depth) {
  // If there's nothing to threshold, just return
  if (min_depth < 0 && max_depth < 0) return depth;

  cv::Mat out = depth.clone();

  if (0 <= min_depth) {
    cv::threshold(out, out, min_depth, 0, cv::THRESH_TOZERO);
  }

  if (0 < max_depth && std::isfinite(max_depth)) {
    cv::threshold(out, out, max_depth, 0, cv::THRESH_TOZERO_INV);
  }

  return out;
}

/// ============================================================================
DatasetBase::DatasetBase(const std::string& name,
                         const std::string& data_dir,
                         const std::vector<std::string>& dtypes)
    : name_{name}, data_dir_{data_dir}, dtypes_{dtypes} {
  CHECK(fs::exists(data_dir_)) << data_dir_;
}

cv::Mat DatasetBase::Get(std::string_view dtype, int i, int cam) const {
  CHECK_GE(i, 0);
  CHECK_GE(cam, 0);
  return GetImpl(dtype, i, cam);
}

std::string DatasetBase::Repr() const {
  return fmt::format("{}(dir={}, size={}, dtypes=[{}])",
                     name_,
                     data_dir_,
                     size_,
                     absl::StrJoin(dtypes_, ", "));
}

bool DatasetBase::is_stereo() const {
  return static_cast<int>(files_.at(DataType::kImage).size()) == (2 * size());
}

/// ============================================================================
IclNuim::IclNuim(const std::string& data_dir)
    : DatasetBase{"icl_nuim", data_dir, kDtypes} {
  // Check directories exist
  const fs::path data_dir_path{data_dir_};

  // Read image files
  files_[DataType::kImage] = GetFiles(data_dir_path / "rgb", ".png", false);
  SortStrsAsInts(files_[DataType::kImage]);
  size_ = static_cast<int>(files_[DataType::kImage].size());

  // Read depth files
  files_[DataType::kDepth] = GetFiles(data_dir_path / "depth", ".png", false);
  SortStrsAsInts(files_[DataType::kDepth]);

  // Read intrinsics
  data_[DataType::kIntrin] = cv::Mat{481.20, -480.00, 319.50, 239.50, 0.0};

  // Read poses, convert to homogeneous transform
  const auto poses = ReadPoses(data_dir_path / "poses.txt");
  data_[DataType::kPose] = ConvertPoses(poses);
}

cv::Mat IclNuim::GetImpl(std::string_view dtype, int i, int /*cam*/) const {
  const auto ind = static_cast<size_t>(i);
  if (dtype == DataType::kImage) {
    const auto& files = files_.at(DataType::kImage);
    if (ind >= files.size()) return {};
    return CvReadImage(files.at(i));
  }

  if (dtype == DataType::kDepth) {
    const auto& files = files_.at(DataType::kDepth);
    if (ind >= files.size()) return {};
    return CvReadDepth(files.at(i), kDepthDivFactor);
  }

  if (dtype == DataType::kIntrin) {
    return data_.at(DataType::kIntrin).clone();
  }

  if (dtype == DataType::kPose) {
    return data_.at(DataType::kPose).row(i).clone();
  }

  return {};
}

cv::Mat IclNuim::ReadPoses(const std::string& file) const {
  std::ifstream ifs{file};
  CHECK(ifs.good()) << "Failed to open file: " << file;
  std::string header;
  std::getline(ifs, header);  // skip the first line

  int i{};
  std::vector<double> data;
  data.reserve(static_cast<size_t>(size_ * kPoseDim));

  // timestamp tx ty tz qx qy qz qw
  while (ifs >> i) {
    std::copy_n(
        std::istream_iterator<double>(ifs), kPoseDim, std::back_inserter(data));
  }

  CHECK_EQ(data.size() % kPoseDim, 0);
  return cv::Mat(data, true)
      .reshape(0, static_cast<int>(data.size()) / kPoseDim);
}

cv::Mat IclNuim::ConvertPoses(const cv::Mat& poses) const {
  CHECK_EQ(poses.cols, kPoseDim);

  SE3d tf;
  cv::Mat transforms;
  transforms.create(poses.rows, kTransformDim, CV_64FC1);

  for (int i = 0; i < poses.rows; ++i) {
    MatToSE3d(poses.row(i), tf);
    Eigen::Map<RowMat44d> tf_map(transforms.ptr<double>(i));
    tf_map = tf.matrix();
  }

  return transforms;
}

/// ============================================================================
Vkitti2::Vkitti2(const std::string& data_dir)
    : DatasetBase{"vkitti2", data_dir, kDtypes} {
  const fs::path data_path{data_dir_};
  const auto image_stereo = ReadStereoData(data_path / "frames/rgb/Camera_0",
                                           data_path / "frames/rgb/Camera_1",
                                           ".jpg",
                                           files_[DataType::kImage]);
  const auto depth_stereo = ReadStereoData(data_path / "frames/depth/Camera_0",
                                           data_path / "frames/depth/Camera_1",
                                           ".png",
                                           files_[DataType::kDepth]);
  const auto image_size = static_cast<int>(files_.at(DataType::kImage).size());
  const auto depth_size = static_cast<int>(files_.at(DataType::kDepth).size());
  size_ = image_stereo ? image_size / 2 : image_size;
  if (depth_size > 0) {
    CHECK_EQ(depth_size, depth_stereo ? size_ * 2 : size_);
  }

  // Read intrinsics
  data_[DataType::kIntrin] = ReadIntrinsics(data_path / "intrinsic.txt");
  CHECK_EQ(data_.at(DataType::kIntrin).rows, size_ * 2);

  // Read extrinsics
  const auto extrins = ReadExtrinsics(data_path / "extrinsic.txt");
  CHECK_EQ(extrins.rows, size_ * 2);
  data_[DataType::kPose] = ConvertExtrins(extrins);
}

Vkitti2 Vkitti2::Create(const std::string& base_dir,
                        int seq,
                        const std::string& var) {
  return Vkitti2{fmt::format("{}/Scene{:02d}/{}", base_dir, seq, var)};
}

cv::Mat Vkitti2::GetImpl(std::string_view dtype, int i, int cam) const {
  if (dtype == DataType::kImage) {
    const size_t ind = ToFileInd(i, cam);
    const auto& files = files_.at(DataType::kImage);
    if (ind >= files.size()) return {};
    return CvReadImage(files.at(ind));
  }

  if (dtype == DataType::kDepth) {
    const size_t ind = ToFileInd(i, cam);
    const auto& files = files_.at(DataType::kDepth);
    if (ind >= files.size()) return {};
    return CvReadDepth(files.at(ind), kDepthDivFactor);
  }

  if (dtype == DataType::kIntrin) {
    const auto ind = ToDataInd(i, cam);
    cv::Mat intrin = cv::Mat::zeros(1, 5, CV_64FC1);
    cv::Mat fxycxy = data_.at(DataType::kIntrin).row(ind);
    fxycxy.copyTo(intrin.colRange(0, 4));
    // Left camera has positive baseline while right has negative
    intrin.at<double>(4) = cam == 0 ? kBaseline : -kBaseline;
    return intrin;
  }

  if (dtype == DataType::kPose) {
    const int ind = ToDataInd(i, cam);
    return data_.at(DataType::kPose).row(ind).clone();
  }

  return {};
}

cv::Mat Vkitti2::ReadExtrinsics(const std::string& file) const {
  std::ifstream ifs{file};
  CHECK(ifs.good()) << "Failed to open file: " << file;
  std::string header;
  std::getline(ifs, header);  // skip the first line

  int i{};
  int cam{};
  std::vector<double> data;
  data.reserve(static_cast<size_t>(size_ * kTransformDim * 2));

  while (ifs >> i >> cam) {
    std::copy_n(std::istream_iterator<double>(ifs),
                kTransformDim,
                std::back_inserter(data));
  }

  CHECK_EQ(data.size() % kTransformDim, 0);
  return cv::Mat(data, true)
      .reshape(0, static_cast<int>(data.size()) / kTransformDim);
}

cv::Mat Vkitti2::ReadIntrinsics(const std::string& file) const {
  std::ifstream ifs{file};
  CHECK(ifs.good()) << "Unable to open file: " << file;
  std::string header;
  std::getline(ifs, header);  // skip the first line

  int i{};
  int cam{};
  std::vector<double> data;
  data.reserve(static_cast<size_t>(size_ * kIntrinDim * 2));

  while (ifs >> i >> cam) {
    std::copy_n(std::istream_iterator<double>(ifs),
                kIntrinDim,
                std::back_inserter(data));
  }

  CHECK_EQ(data.size() % kIntrinDim, 0);
  return cv::Mat(data, true)
      .reshape(0, static_cast<int>(data.size()) / kIntrinDim);
}

cv::Mat Vkitti2::ConvertExtrins(const cv::Mat& extrins) const {
  CHECK_EQ(extrins.cols, kTransformDim);

  cv::Mat transforms;
  transforms.create(extrins.rows, kTransformDim, CV_64FC1);

  for (int i = 0; i < extrins.rows; ++i) {
    Eigen::Map<const RowMat44d> extrin_map(extrins.ptr<double>(i));
    const SE3d extrin(SO3d::fitToSO3(extrin_map.topLeftCorner<3, 3>()),
                      extrin_map.topRightCorner<3, 1>());
    Eigen::Map<RowMat44d> tf_map(transforms.ptr<double>(i));
    tf_map = extrin.inverse().matrix();
  }

  return transforms;
}

/// ============================================================================
TartanAir::TartanAir(const std::string& data_dir)
    : DatasetBase{"tartan_air", data_dir, kDtypes} {
  const fs::path data_path{data_dir_};

  const auto image_stereo = ReadStereoData(data_path / "image_left",
                                           data_path / "image_right",
                                           ".png",
                                           files_[DataType::kImage]);

  // Allow depth to be optional
  const auto depth_stereo = ReadStereoData(data_path / "depth_left",
                                           data_path / "depth_right",
                                           ".npy",
                                           files_[DataType::kDepth]);

  const auto image_size = static_cast<int>(files_.at(DataType::kImage).size());
  const auto depth_size = static_cast<int>(files_.at(DataType::kDepth).size());
  size_ = image_stereo ? image_size / 2 : image_size;
  if (depth_size > 0) {
    CHECK_EQ(depth_size, depth_stereo ? size_ * 2 : size_);
  }

  // left poses on top, right poses bottom
  cv::Mat poses;
  cv::vconcat(ReadPoses(data_path / "pose_left.txt"),
              ReadPoses(data_path / "pose_right.txt"),
              poses);
  CHECK_EQ(poses.rows, size_ * 2);
  data_[DataType::kPose] = ConvertPoses(poses);

  // Intrins
  const cv::Mat intrin({1, 5}, {320.0, 320.0, 320.0, 240.0, 0.25});
  data_[DataType::kIntrin] = MakeStereoIntrins(intrin);
}

TartanAir TartanAir::Create(const std::string& base_dir,
                            const std::string& scene,
                            const std::string& mode,
                            int seq) {
  return TartanAir{fmt::format("{}/{}/{}/P{:03d}", base_dir, scene, mode, seq)};
}

cv::Mat TartanAir::ReadPoses(const std::string& file) const {
  constexpr int kPoseSize = 7;
  std::ifstream ifs{file};
  CHECK(ifs.good()) << "Failed to open file: " << file;

  // tx ty tz qx qy qz qw
  std::vector<double> data;
  data.reserve(static_cast<size_t>(size_ * kPoseSize));

  std::copy(std::istream_iterator<double>(ifs),
            std::istream_iterator<double>(),
            std::back_inserter(data));

  CHECK_EQ(data.size() % kPoseSize, 0);
  return cv::Mat(data, true)
      .reshape(0, static_cast<int>(data.size()) / kPoseSize);
}

cv::Mat TartanAir::GetImpl(std::string_view dtype, int i, int cam) const {
  const size_t ind = ToInd(i, cam);

  cv::Mat data;

  if (dtype == DataType::kImage) {
    const auto& files = files_.at(DataType::kImage);
    if (ind >= files.size()) return {};
    return CvReadImage(files.at(ind));
  }

  if (dtype == DataType::kDepth) {
#if XTENSOR_FOUND
    const auto& files = files_.at(DataType::kDepth);
    if (ind >= files.size()) return {};
    const auto raw = xt::load_npy<float>(files.at(ind));

    // Copy to Mat, maybe there is a faster way
    cv::Mat depth;
    depth.create(static_cast<int>(raw.shape()[0]),
                 static_cast<int>(raw.shape()[1]),
                 CV_32FC1);
    for (int r = 0; r < depth.rows; ++r) {
      for (int c = 0; c < depth.cols; ++c) {
        depth.at<float>(r, c) = raw(r, c);
      }
    }
    return depth;
#else
    LOG(WARNING) << "xtensor not found, unable to read npy, return empty depth";
    return {};
#endif
  }

  if (dtype == DataType::kIntrin) {
    return data_.at(DataType::kIntrin).row(cam).clone();
  }

  if (dtype == DataType::kPose) {
    return data_.at(DataType::kPose).row(static_cast<int>(ind)).clone();
  }

  return {};
}

cv::Mat TartanAir::ConvertPoses(const cv::Mat& poses) const {
  CHECK_EQ(poses.cols, kPoseDim);

  SE3d tf_w_ned;
  cv::Mat transforms;
  transforms.create(poses.rows, kTransformDim, CV_64FC1);

  // motion is defined in ned frame, but what we really need is motion of camera
  // T_w_c = T_w_ned @ R_ned_c
  Matrix3d R_ned_c;
  // clang-format off
  R_ned_c << 0, 0, 1,
             1, 0, 0,
             0, 1, 0;
  // clang-format on
  const SE3d tf_ned_c(R_ned_c, Vector3d::Zero());

  for (int i = 0; i < poses.rows; ++i) {
    MatToSE3d(poses.row(i), tf_w_ned);
    Eigen::Map<RowMat44d> tf_map(transforms.ptr<double>(i));
    tf_map = (tf_w_ned * tf_ned_c).matrix();
  }

  return transforms;
}

/// ============================================================================
KittiOdom::KittiOdom(const std::string& data_dir)
    : DatasetBase{"kitti", data_dir, kDtypes} {
  const fs::path data_path{data_dir_};
  const auto image_stereo = ReadStereoData(data_path / "image_0",
                                           data_path / "image_1",
                                           ".png",
                                           files_[DataType::kImage]);
  const auto image_size = static_cast<int>(files_.at(DataType::kImage).size());
  size_ = image_stereo ? image_size / 2 : image_size;

  // Intrinsics
  data_[DataType::kIntrin] = ReadIntrinsics(data_path / "calib.txt");

  // Extrinsics (GT poses)
  const cv::Mat raw_poses = ReadPoses(data_path / "../../poses" /
                                      (data_path.stem().string() + ".txt"));
  CHECK_EQ(size_, raw_poses.rows);
  data_[DataType::kPose] = ConvertPoses(raw_poses);
}

KittiOdom KittiOdom::Create(const std::string& base_dir, int seq) {
  return KittiOdom{fmt::format("{}/sequences/{:02d}", base_dir, seq)};
}

cv::Mat KittiOdom::GetImpl(std::string_view dtype, int i, int cam) const {
  if (dtype == DataType::kImage) {
    const size_t ind = ToInd(i, cam);
    const auto& files = files_.at(DataType::kImage);
    if (ind >= files.size()) return {};
    return CvReadImage(files.at(ind));
  }

  if (dtype == DataType::kIntrin) {
    cv::Mat intrin = data_.at(DataType::kIntrin).clone();
    if (cam == 1) {
      intrin.at<double>(4) *= -1;
    }
    return intrin;
  }

  if (dtype == DataType::kPose) {
    auto pose = data_.at(DataType::kPose).row(i).clone();
    if (cam == 1) {
      // Need pose of right camera
      // T_w_r = T_w_l * T_l_r
      //       = [R t] * [I b] = [R R*b + t]
      //       = [R R.col(0) * b + t]
      Eigen::Map<RowMat44d> T_w_l(pose.ptr<double>());
      T_w_l.topRightCorner<3, 1>() += T_w_l.topLeftCorner<3, 1>() * baseline_;
    }
    return pose;
  }

  return {};
}

cv::Mat KittiOdom::ReadIntrinsics(const std::string& file) {
  constexpr int kProjSize = 12;
  std::ifstream ifs{file};
  CHECK(ifs.good()) << "Unable to open file: " << file;
  std::string buf;
  // First line is left camera, but matrix is same for right camera
  // and right also contains baseline information
  std::getline(ifs, buf);

  std::vector<double> data;
  data.reserve(kProjSize);

  // Ignore first number (index) in line
  ifs >> buf;
  std::copy_n(
      std::istream_iterator<double>(ifs), kProjSize, std::back_inserter(data));

  // Projection matrix
  // [0, 1,  2,  3]
  // [4, 5,  6,  7]
  // [8, 9, 10, 11]

  baseline_ = -data[3] / data[0];
  std::vector<double> intrin{data[0], data[5], data[2], data[6], baseline_};
  return cv::Mat(intrin, true);
}

cv::Mat KittiOdom::ReadPoses(const std::string& file) const {
  constexpr int kTransSize = 12;
  std::ifstream ifs{file};
  CHECK(ifs.good()) << "Unable to open file: " << file;

  std::vector<double> data;
  data.reserve(static_cast<size_t>(size_ * kTransformDim));

  std::string line;
  while (std::getline(ifs, line)) {
    std::istringstream iss(line);
    std::copy_n(std::istream_iterator<double>(iss),
                kTransSize,
                std::back_inserter(data));
    // kitti pose file only stores 3x4 matrix, so in order to get 4x4 matrix we
    // need to push back bottom row manually
    data.push_back(0);
    data.push_back(0);
    data.push_back(0);
    data.push_back(1);
  }

  CHECK_EQ(data.size() % kTransformDim, 0);
  return cv::Mat(data, true)
      .reshape(0, static_cast<int>(data.size()) / kTransformDim);
}

cv::Mat KittiOdom::ConvertPoses(const cv::Mat& poses) const {
  CHECK_EQ(poses.cols, kTransformDim);

  cv::Mat transforms;
  transforms.create(poses.rows, kTransformDim, CV_64FC1);

  for (int i = 0; i < poses.rows; ++i) {
    Eigen::Map<const RowMat44d> extrin_map(poses.ptr<double>(i));
    const SE3d extrin(SO3d::fitToSO3(extrin_map.topLeftCorner<3, 3>()),
                      extrin_map.topRightCorner<3, 1>());
    Eigen::Map<RowMat44d> tf_map(transforms.ptr<double>(i));
    tf_map = extrin.matrix();
  }

  return transforms;
}

/// ============================================================================
EuRoC::EuRoC(const std::string& data_dir)
    : DatasetBase{"euroc", data_dir, kDtypes},
      prefix_{data_dir + "/mav0/cam0/data", data_dir + "/mav0/cam1/data"} {
  const fs::path data_path{data_dir_};
  // ReadTimestamps(data_path / "times.txt", files_[DataType::kImage]);
  const cv::Mat raw_poses = ReadAlignedTimestampsAndPoses(
      data_path / "times.txt",
      data_path / "../gt_pose_aligned/" /
          (data_path.stem().string() + "_cam0_aligned.txt"),
      files_[DataType::kImage]);
  size_ = static_cast<int>(files_.at(DataType::kImage).size()) / 2;

  // Intrinsics
  data_[DataType::kIntrin] = ReadIntrinsics(data_path / "../calib0.yaml",
                                            data_path / "../calib1.yaml");

  // Extrinsics (GT poses)
  // const cv::Mat raw_poses =
  // ReadPoses(data_path / "../gt_pose_aligned/" /
  // (data_path.stem().string() + "_cam0_aligned.txt"),
  // files_[DataType::kImage]);

  CHECK_EQ(size_, raw_poses.rows);
  data_[DataType::kPose] = ConvertPoses(raw_poses);
}

cv::Mat EuRoC::GetImpl(std::string_view dtype, int i, int cam) const {
  if (dtype == DataType::kImage) {
    const size_t ind = ToInd(i, cam);
    const auto& files = files_.at(DataType::kImage);
    if (ind >= files.size()) return {};
    cv::Mat raw = CvReadImage(prefix_[cam] + "/" + files.at(ind) + ".png");
    cv::Mat rectified;
    if (cam == 0) {
      cv::remap(raw, rectified, calib0_M1_, calib0_M2_, cv::INTER_LINEAR);
    } else {
      cv::remap(raw, rectified, calib1_M1_, calib1_M2_, cv::INTER_LINEAR);
    }
    return rectified;
  }

  if (dtype == DataType::kIntrin) {
    cv::Mat intrin = data_.at(DataType::kIntrin).clone();
    if (cam == 1) {
      intrin.at<double>(4) *= -1;
    }
    return intrin;
  }

  if (dtype == DataType::kPose) {
    auto pose = data_.at(DataType::kPose).row(i).clone();
    if (cam == 1) {
      // Need pose of right camera
      // T_w_r = T_w_l * T_l_r
      //       = [R t] * [I b] = [R R*b + t]
      //       = [R R.col(0) * b + t]
      Eigen::Map<RowMat44d> T_w_l(pose.ptr<double>());
      T_w_l.topRightCorner<3, 1>() += T_w_l.topLeftCorner<3, 1>() * baseline_;
    }
    return pose;
  }
  if (dtype == DataType::kTimestamp) {
    const size_t ind = ToInd(i, cam);
    const double timestamp =
        std::stod(files_.at(DataType::kImage).at(ind)) / 1e9;
    return cv::Mat(std::vector<double>{timestamp}, true);
  }

  return {};
}

void EuRoC::ReadTimestamps(const std::string& name,
                           std::vector<std::string>& files) const {
  std::ifstream ifs(name);
  CHECK(ifs.good()) << "Unable to open file: " << name;

  std::string line;
  while (std::getline(ifs, line)) {
    files.emplace_back(line);
  }

  const size_t num = files.size();
  for (size_t i = 0; i < num; i++) {
    files.emplace_back(files.at(i));  // add stereo
  }
}

cv::Mat EuRoC::ReadIntrinsics(const std::string& calib0_file,
                              const std::string& calib1_file) {
  auto calib0 = Parameters::loadFromYaml(calib0_file);
  auto calib1 = Parameters::loadFromYaml(calib1_file);
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

  // init undistortion map
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
  baseline_ = std::fabs(P1.at<double>(0, 3) / P1.at<double>(0, 0));
  std::vector<double> intrin{K.at<double>(0, 0),
                             K.at<double>(1, 1),
                             K.at<double>(0, 2),
                             K.at<double>(1, 2),
                             baseline_};
  return cv::Mat(intrin, true);
}

cv::Mat EuRoC::ReadPoses(const std::string& file,
                         const std::vector<std::string>& timestamps) const {
  // constexpr int kTransSize = 12;
  std::ifstream ifs{file};
  CHECK(ifs.good()) << "Unable to open file: " << file;

  std::vector<double> data;
  data.reserve(timestamps.size() * 16);

  int index = 0;
  double timestamp = 0.0, tx = 0.0, ty = 0.0, tz = 0.0, qx = 0.0, qy = 0.0,
         qz = 0.0, qw = 1.0;
  while (true) {
    // read line
    ifs >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    ++index;

    if (ifs.eof()) {
      break;
    }

    Matrix3d R = Quaterniond(qw, qx, qy, qz).toRotationMatrix();
    std::vector<double> pose{R(0, 0),
                             R(0, 1),
                             R(0, 2),
                             tx,
                             R(1, 0),
                             R(1, 1),
                             R(1, 2),
                             ty,
                             R(2, 0),
                             R(2, 1),
                             R(2, 2),
                             tz,
                             0,
                             0,
                             0,
                             1};
    std::copy_n(pose.begin(), 16, std::back_inserter(data));
  }

  CHECK_EQ(data.size() % kTransformDim, 0);
  return cv::Mat(data, true)
      .reshape(0, static_cast<int>(data.size()) / kTransformDim);
}

cv::Mat EuRoC::ConvertPoses(const cv::Mat& poses) const {
  CHECK_EQ(poses.cols, kTransformDim);

  cv::Mat transforms;
  transforms.create(poses.rows, kTransformDim, CV_64FC1);

  for (int i = 0; i < poses.rows; ++i) {
    Eigen::Map<const RowMat44d> extrin_map(poses.ptr<double>(i));
    const SE3d extrin(SO3d::fitToSO3(extrin_map.topLeftCorner<3, 3>()),
                      extrin_map.topRightCorner<3, 1>());
    Eigen::Map<RowMat44d> tf_map(transforms.ptr<double>(i));
    tf_map = extrin.matrix();
  }

  return transforms;
}

////////////////////////////////////////////////////////////////////////////////
EuRoC::Parameters EuRoC::Parameters::loadFromYaml(const std::string& file) {
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

cv::Mat EuRoC::ReadAlignedTimestampsAndPoses(
    const std::string& timestamp_file_name,
    const std::string& pose_file_name,
    std::vector<std::string>& files) {
  // 1. read image timestamps
  std::vector<std::string> image_timestamps_str;
  std::vector<double> image_timestamps;
  {
    std::ifstream ifs(timestamp_file_name);
    CHECK(ifs.good()) << "Unable to open file: " << timestamp_file_name;
    std::string line;
    while (std::getline(ifs, line)) {
      image_timestamps_str.emplace_back(line);
      image_timestamps.emplace_back(
          std::round(std::stod(line) / 1e7) /
          1e2);  // nsec to sec, round to second decimal
    }
  }

  // 2. read pose timestamps
  std::vector<double> data;
  data.reserve(image_timestamps.size() * 16);
  std::vector<double> pose_timestamps;
  {
    std::ifstream ifs{pose_file_name};
    CHECK(ifs.good()) << "Unable to open file: " << pose_file_name;

    int index = 0;
    double timestamp = 0.0, tx = 0.0, ty = 0.0, tz = 0.0, qx = 0.0, qy = 0.0,
           qz = 0.0, qw = 1.0;
    while (true) {
      // read line
      ifs >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
      ++index;

      if (ifs.eof()) {
        break;
      }
      Matrix3d R = Quaterniond(qw, qx, qy, qz).toRotationMatrix();
      std::vector<double> pose{R(0, 0),
                               R(0, 1),
                               R(0, 2),
                               tx,
                               R(1, 0),
                               R(1, 1),
                               R(1, 2),
                               ty,
                               R(2, 0),
                               R(2, 1),
                               R(2, 2),
                               tz,
                               0,
                               0,
                               0,
                               1};
      std::copy_n(pose.begin(), 16, std::back_inserter(data));
      pose_timestamps.emplace_back(std::round(timestamp * 1e2) / 1e2);
    }
    CHECK_EQ(data.size() % kTransformDim, 0);
  }

  // 3. align image and pose timestamps
  size_t si = 0u, sj = 0u;
  while (si < image_timestamps.size() && sj < pose_timestamps.size()) {
    if (image_timestamps[si] < pose_timestamps[sj]) {
      ++si;
    } else if (image_timestamps[si] > pose_timestamps[sj]) {
      ++sj;
    } else {
      break;
    }
  }
  int64_t ei = (int64_t)image_timestamps.size() - 1,
          ej = (int64_t)pose_timestamps.size() - 1;
  while (ei >= 0 && ej >= 0) {
    if (image_timestamps[ei] < pose_timestamps[ej]) {
      --ej;
    } else if (image_timestamps[ei] > pose_timestamps[ej]) {
      --ei;
    } else {
      break;
    }
  }

  // index check
  CHECK_GT(ei, si);
  CHECK_GT(ej, sj);
  CHECK_EQ(ei - si, ej - sj);

  // copy timestamsp
  std::copy_n(image_timestamps_str.begin() + si,
              ei - si + 1,
              std::back_inserter(files));
  const size_t num = files.size();
  for (size_t i = 0; i < num; i++) {
    files.emplace_back(files.at(i));  // add stereo
  }

  // copy pose
  std::vector<double> aligned_data(data.begin() + sj * kTransformDim,
                                   data.begin() + (ej + 1) * kTransformDim);
  return cv::Mat(aligned_data, true)
      .reshape(0, static_cast<int>(aligned_data.size()) / kTransformDim);
}

std::string ExtractDatasetName(const std::string& data_dir) {
  std::string name;
  // Extract name from base_dir
  if (absl::StrContains(data_dir, "vkitti")) {
    name = "vkitti";
  } else if (absl::StrContains(data_dir, "kitti")) {
    name = "kitti";
  } else if (absl::StrContains(data_dir, "tartan_air")) {
    name = "tartan_air";
  } else if (absl::StrContains(data_dir, "realsense")) {
    name = "realsense";
  } else if (absl::StrContains(data_dir, "EuRoC")) {
    name = "EuRoC";
  }

  return name;
}

Dataset CreateDataset(const std::string& data_dir, std::string name) {
  if (name.empty()) {
    name = ExtractDatasetName(data_dir);
  }

  Dataset ds;

  if (name.empty()) {
    LOG(WARNING) << fmt::format("Could not infer dataset name from dir '{}'",
                                data_dir);
    return ds;
  }

  if (name == "vkitti") {
    ds = Vkitti2(data_dir);
  } else if (name == "kitti") {
    ds = KittiOdom(data_dir);
  } else if (name == "tartan_air") {
    ds = TartanAir(data_dir);
  } else if (name == "realsense") {
    ds = StereoFolder::Create(name, data_dir, "infra1", "infra2", "calib.txt");
  } else if (name == "EuRoC") {
    ds = EuRoC(data_dir);
  } else {
    LOG(WARNING) << fmt::format("Invalid dataset name: {}", name);
  }

  return ds;
}

StereoFolder::StereoFolder(const std::string& name,
                           const std::string& left_dir,
                           const std::string& right_dir,
                           const std::string& calib_file)
    : DatasetBase{name, left_dir, kDtypes} {
  const auto image_stereo =
      ReadStereoData(left_dir, right_dir, ".png", files_[DataType::kImage]);
  const auto image_size = static_cast<int>(files_.at(DataType::kImage).size());
  size_ = image_stereo ? image_size / 2 : image_size;

  // Intrinsics fx fy cx cy b
  data_[DataType::kIntrin] = ReadIntrinsics(calib_file);
}

StereoFolder StereoFolder::Create(const std::string& data_name,
                                  const std::string& data_dir,
                                  const std::string& left_name,
                                  const std::string& right_name,
                                  const std::string& calib_name) {
  const auto data_path = fs::path(data_dir);
  return StereoFolder(data_name,
                      data_path / left_name,
                      data_path / right_name,
                      data_path / calib_name);
}

cv::Mat StereoFolder::GetImpl(std::string_view dtype, int i, int cam) const {
  if (dtype == DataType::kImage) {
    const size_t ind = ToInd(i, cam);
    const auto& files = files_.at(DataType::kImage);
    if (ind >= files.size()) return {};
    return CvReadImage(files.at(ind));
  }

  if (dtype == DataType::kIntrin) {
    return data_.at(DataType::kIntrin).row(cam).clone();
  }

  return {};
}

cv::Mat StereoFolder::ReadIntrinsics(const std::string& file) {
  std::ifstream ifs{file};
  CHECK(ifs.good()) << "Unable to open file: " << file;
  // Get first line and parse [fx, fy, cx, cy, b]
  std::string buf;
  std::getline(ifs, buf);

  std::vector<double> data;
  std::istringstream iss(buf);
  std::copy_n(std::istream_iterator<double>(iss), 5, std::back_inserter(data));
  return MakeStereoIntrins(cv::Mat(data).reshape(0, 1));
}

}  // namespace sv
