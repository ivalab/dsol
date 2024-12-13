#include "sv/dsol/extra.h"

#include <opencv2/imgproc.hpp>

#include "sv/util/logging.h"

namespace sv::dsol {

void MotionModel::Init(const Sophus::SE3d& T_w_c,
                       const Eigen::Vector3d& vel,
                       const Eigen::Vector3d& omg) {
  T_last_ = T_w_c;
  vel_ = vel;
  omg_ = omg;
  init_ = true;
}

void MotionModel::Correct(const Sophus::SE3d& T_w_c, double dt) {
  // Only update velocity when dt is > 0
  if (dt > 0) {
    const auto tf_delta = T_last_.inverse() * T_w_c;
    const auto w = alpha_ / dt;
    omg_ = (1 - alpha_) * omg_ + (alpha_ / dt) * tf_delta.so3().log();
    vel_ = (1 - alpha_) * vel_ + (alpha_ / dt) * tf_delta.translation();
  }

  T_last_ = T_w_c;
}

/// ============================================================================
TumFormatWriter::TumFormatWriter(const std::string& prefix) : prefix_{prefix} {
  if (!prefix_.empty()) {
    named_ofs_["est"] = std::ofstream{prefix_ + "_AllFrameTrajectory.txt"};
    named_ofs_["est"] << "# timestamp tx ty tz qx qy qz qw\n";
    // named_ofs_["prior"] =
    //     std::ofstream{prefix_ + "_AllFrameTrajectoryPrior.txt"};
    // named_ofs_["prior"] << "# timestamp tx ty tz qx qy qz qw\n";
  }
}

void TumFormatWriter::Write(double timestamp,
                            const Sophus::SE3d& Twc_est,
                            const Sophus::SE3d& Twc_prior) {
  if (named_ofs_.count("est") && named_ofs_["est"].good()) {
    const auto& t = Twc_est.translation();
    const auto& q = Twc_est.unit_quaternion();
    const auto line =
        fmt::format("{:.4f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                    timestamp,
                    t.x(),
                    t.y(),
                    t.z(),
                    q.x(),
                    q.y(),
                    q.z(),
                    q.w());
    named_ofs_["est"] << line << std::endl;
  }

  if (named_ofs_.count("prior") && named_ofs_["prior"].good()) {
    const auto& t = Twc_prior.translation();
    const auto& q = Twc_prior.unit_quaternion();
    const auto line =
        fmt::format("{:.4f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}",
                    timestamp,
                    t.x(),
                    t.y(),
                    t.z(),
                    q.x(),
                    q.y(),
                    q.z(),
                    q.w());
    named_ofs_["prior"] << line << std::endl;
  }
}

void TumFormatWriter::Write(const std::string& filename,
                            const Sophus::SE3d& base_to_cam) {
  std::ofstream ofs{prefix_ + "_" + filename + ".txt"};
  ofs << "# base_to_cam: [R, t; 0 0 0 1]\n";
  ofs << base_to_cam.matrix();
  ofs.close();
}

StatsWriter::StatsWriter(const std::string& prefix) : prefix_(prefix) {
  if (!prefix_.empty()) {
    named_ofs_["configs"] = std::ofstream(prefix_ + "_configs.txt");
    named_ofs_["timings"] = std::ofstream(prefix_ + "_timings.txt");
    named_ofs_["timings"]
        << "# timestamp tracking_time mapping_time (seconds)\n";
    named_ofs_["tracking_stats"] =
        std::ofstream(prefix_ + "_tracking_stats.txt");
    named_ofs_["mapping_stats"] = std::ofstream(prefix_ + "_mapping_stats.txt");
  }
}

void StatsWriter::WriteConfigs(const std::string& config_str) {
  if (!named_ofs_.count("configs") || !named_ofs_["configs"].good()) {
    return;
  }
  named_ofs_["configs"] << config_str;
}

void StatsWriter::WriteTimings(double timestamp,
                               double tracking_time,
                               double mapping_time) {
  if (!named_ofs_.count("timings") || !named_ofs_["timings"].good()) {
    return;
  }
  named_ofs_["timings"] << fmt::format("{:.4f} {:.6f} {:.6f}",
                                       timestamp,
                                       tracking_time,
                                       mapping_time)
                        << std::endl;
}

void StatsWriter::WriteTrackingStatsHeader(const std::string& header) {
  named_ofs_["tracking_stats"] << "# timestamp " << header << "\n";
}

void StatsWriter::WriteTrackingStats(double timestamp,
                                     const std::string& stats) {
  named_ofs_["tracking_stats"] << timestamp << " " << stats << "\n";
}

void StatsWriter::WriteMappingStatsHeader(const std::string& header) {
  named_ofs_["mapping_stats"] << "# timestamp " << header << "\n";
}

void StatsWriter::WriteMappingStats(double timestamp,
                                    const std::string& stats) {
  named_ofs_["mapping_stats"] << timestamp << " " << stats << "\n";
}

/// ============================================================================
PlayData::PlayData(const Dataset& dataset, const PlayCfg& cfg)
    : frames(cfg.nframes), depths(cfg.nframes), poses(cfg.nframes) {
  const auto tf_c0_w = SE3dFromMat(dataset.Get(DataType::kPose, 0)).inverse();

  for (int k = 0; k < cfg.nframes; ++k) {
    const int i = cfg.index + k * (cfg.skip + 1);
    LOG(INFO) << "Reading index: " << i;

    // pose
    const auto pose = dataset.Get(DataType::kPose, i);
    const auto tf_w_c = SE3dFromMat(pose);
    const auto tf_c0_c = tf_c0_w * tf_w_c;

    int aff_val = cfg.affine ? k : 0;

    // stereo images
    ImagePyramid grays_l;
    {
      auto image = dataset.Get(DataType::kImage, i, 0);
      if (image.type() == CV_8UC3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
      }

      if (aff_val != 0) image += aff_val;
      MakeImagePyramid(image, cfg.nlevels, grays_l);
    }

    ImagePyramid grays_r;
    {
      auto image = dataset.Get(DataType::kImage, i, 1);
      if (image.type() == CV_8UC3) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
      }
      if (aff_val != 0) image += aff_val;
      MakeImagePyramid(image, cfg.nlevels, grays_r);
    }

    AffineModel affm{0, static_cast<double>(aff_val)};
    frames.at(k) = Frame(grays_l, grays_r, tf_c0_c, affm, affm);
    depths.at(k) = dataset.Get(DataType::kDepth, i);
    poses.at(k) = tf_c0_c;  // gt

    LOG(INFO) << fmt::format("frame {}: {}", k, frames.at(k));
  }

  const auto intrin = dataset.Get(DataType::kIntrin, 0);
  camera = Camera::FromMat(frames.front().image_size(), intrin);
}

void InitKfWithDepth(Keyframe& kf,
                     const Camera& camera,
                     PixelSelector& selector,
                     const cv::Mat& depth,
                     TimerSummary& tm,
                     int gsize) {
  {
    auto t = tm.Scoped("SelectPixels");
    selector.Select(kf.grays_l(), gsize);
  }

  {
    auto t = tm.Scoped("InitPoints");
    kf.InitPoints(selector.pixels(), camera);
  }

  {
    auto t = tm.Scoped("InitDepths");
    kf.InitFromDepth(depth);
  }

  {
    auto t = tm.Scoped("InitPatches");
    kf.InitPatches(gsize);
  }

  kf.UpdateStatusInfo();
}

void PlayCfg::Check() const {
  CHECK_GE(index, 0);
  CHECK_GT(nframes, 0);
  CHECK_GE(skip, 0);
  CHECK_GT(nlevels, 0);
}

std::string PlayCfg::Repr() const {
  return fmt::format(
      "PlayCfg(index={}, nframes={}, skip={}, nlevels={}, affine={})",
      index,
      nframes,
      skip,
      nlevels,
      affine);
}

}  // namespace sv::dsol
