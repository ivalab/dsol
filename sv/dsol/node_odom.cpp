#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wconversion"
#include <cv_bridge/cv_bridge.h>
#include <fmt/format.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <boost/circular_buffer.hpp>

#pragma GCC diagnostic pop

#include "sv/dsol/extra.h"
#include "sv/dsol/node_util.h"
#include "sv/dsol/odom.h"
#include "sv/ros1/msg_conv.h"
#include "sv/util/undistorter.h"

namespace sv::dsol {

namespace cb = cv_bridge;
namespace sm = sensor_msgs;
namespace gm = geometry_msgs;
namespace mf = message_filters;

struct NodeOdom {
  explicit NodeOdom(const ros::NodeHandle& pnh);

  void InitOdom();
  void InitRosIO();

  void Cinfo0Cb(const sm::CameraInfo& cinfo0_msg);
  void Cinfo1Cb(const sm::CameraInfo& cinfo1_msg);
  void MonoDepthCb(const sm::ImageConstPtr& image0_ptr,
                   const sm::ImageConstPtr& depth0_ptr);
  void StereoCb(const sm::ImageConstPtr& image0_ptr,
                const sm::ImageConstPtr& image1_ptr);
  void StereoDepthCb(const sm::ImageConstPtr& image0_ptr,
                     const sm::ImageConstPtr& image1_ptr,
                     const sm::ImageConstPtr& depth0_ptr);

  void TfCamCb(const gm::Transform& tf_cam_msg);
  void TfImuCb(const gm::Transform& tf_imu_msg);

  void AccCb(const sm::Imu& acc_msg);
  void GyrCb(const sm::Imu& gyr_msg);

  void PublishOdom(const std_msgs::Header& header, const Sophus::SE3d& tf);
  void PublishCloud(const std_msgs::Header& header);
  void PublishDisplayImage(const std_msgs::Header& header,
                           const cv::Mat& image);

  using SyncStereo = mf::TimeSynchronizer<sm::Image, sm::Image>;
  using SyncStereoDepth = mf::TimeSynchronizer<sm::Image, sm::Image, sm::Image>;
  using MySyncPolicy = mf::sync_policies::ApproximateTime<sm::Image, sm::Image>;
  using AdaptSynchronizer = mf::Synchronizer<MySyncPolicy>;

  ros::NodeHandle pnh_;

  boost::circular_buffer<sm::Imu> gyrs_;
  mf::Subscriber<sm::Image> sub_image0_;
  mf::Subscriber<sm::Image> sub_image1_;
  mf::Subscriber<sm::Image> sub_depth0_;

  std::optional<SyncStereo> sync_stereo_;
  std::optional<SyncStereoDepth> sync_stereo_depth_;
  std::optional<AdaptSynchronizer> sync_mono_depth_;
  sv::Undistorter::Ptr undistorter_ptr_;

  ros::Subscriber sub_cinfo0_;
  ros::Subscriber sub_cinfo1_;
  //  ros::Subscriber sub_acc_;
  ros::Subscriber sub_gyr_;

  ros::Publisher pub_points_;
  ros::Publisher pub_parray_;
  ros::Publisher pub_camera_pose_in_imu_;
  PosePathPublisher pub_odom_;
  ros::Publisher pub_disp_frame_;

  // Publish robot wheel odometry info as reference only.
  void SendTransform(const ros::Time& time, const Sophus::SE3d& tf);
  void RobotWheelOdomCb(const nav_msgs::OdometryConstPtr& msg);
  ros::Subscriber sub_robot_wheel_odom_;
  ros::Publisher pub_robot_wheel_path_;
  nav_msgs::Path robot_wheel_path_msg_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tfli_;
  tf2_ros::TransformBroadcaster tfbr_;
  tf2_ros::StaticTransformBroadcaster static_br_;

  MotionModel motion_;
  TumFormatWriter writer_;
  DirectOdometry odom_;
  bool is_gazebo_{false};

  std::string fixed_frame_{"fixed"};  // Defined by the first camera frame.
  std::string map_frame_{"map"};
  std::string odom_frame_{"odom"};
  std::string camera_frame_{"camera"};
  std::string base_frame_{"base_footprint"};
  bool publish_map_to_odom_tf_{false};
  bool publish_tf_{false};

  sm::PointCloud2 cloud_;
};

NodeOdom::NodeOdom(const ros::NodeHandle& pnh)
    : pnh_(pnh),
      gyrs_(50),
      sub_image0_(pnh_, "image0", 5),
      sub_image1_(pnh_, "image1", 5),
      sub_depth0_(pnh_, "depth0", 5),
      tfli_(tf_buffer_) {
  InitOdom();
  InitRosIO();

  const auto save = pnh_.param<std::string>("save", "");
  writer_ = TumFormatWriter(save);
  if (!writer_.IsDummy()) {
    ROS_WARN_STREAM("Writing results to: " << writer_.prefix());
  }
}

void NodeOdom::InitOdom() {
  {
    auto cfg = ReadOdomCfg({pnh_, "odom"});
    pnh_.getParam("tbb", cfg.tbb);
    pnh_.getParam("log", cfg.log);
    pnh_.getParam("vis", cfg.vis);
    odom_.Init(cfg);
  }
  if (pnh_.hasParam("cell_size")) {
    auto cfg = ReadSelectCfg({pnh_, "select"});
    pnh_.getParam("cell_size", cfg.cell_size);
    odom_.selector.Init(cfg);
    LOG(INFO) << "selector cell_size = " << cfg.cell_size;
  } else {
    odom_.selector = PixelSelector(ReadSelectCfg({pnh_, "select"}));
  }
  // (GP)
  {
    const auto save = pnh_.param<std::string>("save", "");
    if (save.empty()) {
      odom_.aligner = FrameAligner(ReadDirectCfg({pnh_, "align"}));
    } else {
      auto cfg = ReadDirectCfg({pnh_, "align"});
      cfg.save_prefix = save;
      odom_.aligner = FrameAligner(cfg);
      odom_.stats_writer_ptr = StatsWriter::Create(save);
      odom_.stats_writer_ptr->WriteMappingStatsHeader(AlignStatus::Header());
      odom_.stats_writer_ptr->WriteTrackingStatsHeader(AlignStatus::Header());
    }
  }
  odom_.matcher = StereoMatcher(ReadStereoCfg({pnh_, "stereo"}));
  odom_.adjuster = BundleAdjuster(ReadDirectCfg({pnh_, "adjust"}));
  odom_.cmap = GetColorMap(pnh_.param<std::string>("cm", "jet"));
  ROS_INFO_STREAM(odom_.Repr());
  if (odom_.stats_writer_ptr) {
    odom_.stats_writer_ptr->WriteConfigs(odom_.Repr());
  }

  // Init motion model
  motion_.Init();
}

void NodeOdom::InitRosIO() {
  bool use_depth = pnh_.param<bool>("use_depth", false);
  bool use_stereo = pnh_.param<bool>("use_stereo", false);
  CHECK(use_depth || use_stereo);  // Either one should be enabled.
  if (!use_stereo) {
    // sync_stereo_.emplace(sub_image0_, sub_depth0_, 5);
    // sync_stereo_->registerCallback(
    //     boost::bind(&NodeOdom::MonoDepthCb, this, _1, _2));
    // @NOTE (yanwei) to deal with not exact matched timestamp (tum-rgbd)
    sync_mono_depth_.emplace(MySyncPolicy(5), sub_image0_, sub_depth0_);
    sync_mono_depth_->registerCallback(
        boost::bind(&NodeOdom::MonoDepthCb, this, _1, _2));
  } else {
    if (use_depth) {
      sync_stereo_depth_.emplace(sub_image0_, sub_image1_, sub_depth0_, 5);
      sync_stereo_depth_->registerCallback(
          boost::bind(&NodeOdom::StereoDepthCb, this, _1, _2, _3));
    } else {
      sync_stereo_.emplace(sub_image0_, sub_image1_, 5);
      sync_stereo_->registerCallback(
          boost::bind(&NodeOdom::StereoCb, this, _1, _2));
    }
  }

  // Init undistorter if any.
  const bool do_rectification = pnh_.param<bool>("do_rectification", false);
  if (do_rectification) {
    const std::string data_name = pnh_.param<std::string>("data", "");
    const std::string calib_dir = pnh_.param<std::string>("calib_dir", "");

    CHECK(!data_name.empty());
    CHECK(!calib_dir.empty());

    undistorter_ptr_ = sv::CreateUndistorter(calib_dir, data_name);
    odom_.SetCamera(Camera::FromMat(undistorter_ptr_->GetImageSize(),
                                    undistorter_ptr_->GetIntrin()));
  } else {
    // subscribe to left camera (no baseline is available)
    sub_cinfo0_ = pnh_.subscribe("cinfo0", 1, &NodeOdom::Cinfo0Cb, this);
    if (use_stereo) {
      // read baseline
      sub_cinfo1_ = pnh_.subscribe("cinfo1", 1, &NodeOdom::Cinfo1Cb, this);
    }
  }
  sub_gyr_ = pnh_.subscribe("gyr", 200, &NodeOdom::GyrCb, this);
  // sub_acc_ = pnh_.subscribe("acc", 100, &NodeOdom::AccCb, this);

  pub_points_ = pnh_.advertise<sm::PointCloud2>("points", 1);
  pub_parray_ = pnh_.advertise<gm::PoseArray>("parray", 1);
  pub_odom_ = PosePathPublisher(pnh_, "odom", fixed_frame_);
  pub_disp_frame_ = pnh_.advertise<sm::Image>("disp_frame", 1);

  // Check if using data from gazebo.
  {
    std::string data_name = pnh_.param<std::string>("data", "");
    std::transform(
        data_name.begin(), data_name.end(), data_name.begin(), ::tolower);
    if (data_name.find("gazebo") == std::string::npos) {
      is_gazebo_ = false;
    } else {
      is_gazebo_ = true;
    }
  }

  pub_camera_pose_in_imu_ =
      pnh_.advertise<geometry_msgs::PoseWithCovarianceStamped>(
          "/slam_pose_topic", 10);

  // Visualize robot on-board odometry.
  const bool vis_wheel_odom = pnh_.param<bool>("vis_wheel_odom", false);
  if (vis_wheel_odom) {
    std::string robot_wheel_odom_topic =
        pnh_.param<std::string>("robot_wheel_odom_topic", "/odom");

    if (!robot_wheel_odom_topic.empty()) {
      sub_robot_wheel_odom_ = pnh_.subscribe(
          robot_wheel_odom_topic, 1, &NodeOdom::RobotWheelOdomCb, this);
      pub_robot_wheel_path_ =
          pnh_.advertise<nav_msgs::Path>("/robot_wheel_path", 1);
    }
  }

  // Read frames.
  {
    map_frame_ = pnh_.param<std::string>("map_frame", "map");
    odom_frame_ = pnh_.param<std::string>("odom_frame", "odom");
    base_frame_ = pnh_.param<std::string>("base_frame", "base_footprint");
    camera_frame_ = pnh_.param<std::string>("camera_frame", "camera");
    publish_map_to_odom_tf_ = pnh_.param<bool>("publish_map_to_odom_tf", false);
    publish_tf_ = pnh_.param<bool>("publish_tf", false);
  }
}

void NodeOdom::Cinfo0Cb(const sensor_msgs::CameraInfo& cinfo0_msg) {
  // no stereo cam is available, read camera parameters from left.
  if (sub_cinfo1_.getTopic().empty()) {
    odom_.camera = MakeCamera(cinfo0_msg);
    ROS_INFO_STREAM(odom_.camera.Repr());
  }
  sub_cinfo0_.shutdown();
}

void NodeOdom::Cinfo1Cb(const sensor_msgs::CameraInfo& cinfo1_msg) {
  odom_.camera = MakeCamera(cinfo1_msg);
  ROS_INFO_STREAM(odom_.camera.Repr());
  sub_cinfo1_.shutdown();
}

void NodeOdom::AccCb(const sensor_msgs::Imu& acc_msg) {}

void NodeOdom::GyrCb(const sensor_msgs::Imu& gyr_msg) {
  // Normally there is a transform from imu to camera, but in realsense, imu and
  // left infrared camera are aligned (only small translation, so we skip
  // reading the tf)

  gyrs_.push_back(gyr_msg);
}

void NodeOdom::MonoDepthCb(const sensor_msgs::ImageConstPtr& image0_ptr,
                           const sensor_msgs::ImageConstPtr& depth0_ptr) {
  return StereoDepthCb(image0_ptr, nullptr, depth0_ptr);
}

void NodeOdom::StereoCb(const sensor_msgs::ImageConstPtr& image0_ptr,
                        const sensor_msgs::ImageConstPtr& image1_ptr) {
  StereoDepthCb(image0_ptr, image1_ptr, nullptr);
}

void NodeOdom::StereoDepthCb(const sensor_msgs::ImageConstPtr& image0_ptr,
                             const sensor_msgs::ImageConstPtr& image1_ptr,
                             const sensor_msgs::ImageConstPtr& depth0_ptr) {
  static size_t skip_img = 0;
  // @NOTE (yanwei) skip the first few images to avoid garbage data
  if (is_gazebo_ && skip_img < 10) {
    ++skip_img;
    return;
  }

  // Make sure camera is initialized.
  if (!odom_.camera.Ok()) {
    ROS_WARN_STREAM("camera not ready!");
    return;
  }

  // Start processing.
  const auto curr_header = image0_ptr->header;
  camera_frame_ = curr_header.frame_id;
  const auto raw_image0 = cb::toCvShare(image0_ptr)->image;
  cv::Mat raw_image1;
  if (image1_ptr) {
    raw_image1 = cb::toCvShare(image1_ptr)->image;
  }

  // Check if online undistortion is needed.
  // @TOOD (yanwei) do we need to care about depth image if available?
  cv::Mat image0, image1;
  if (undistorter_ptr_) {
    image0 = undistorter_ptr_->Run(raw_image0, 0);
    if (!raw_image1.empty()) {
      image1 = undistorter_ptr_->Run(raw_image1, 1);
    }
  } else {
    // shallow copy
    image0 = raw_image0;
    image1 = raw_image1;
  }

  // depth
  cv::Mat depth0;
  if (depth0_ptr) {
    depth0 = cb::toCvCopy(depth0_ptr)->image;
    depth0.convertTo(depth0,
                     CV_32FC1,
                     1.0 / odom_.cfg().depth_factor);  // convert to meters
  }

  // Get delta time
  static ros::Time prev_stamp;
  const auto delta_duration =
      prev_stamp.isZero() ? ros::Duration{} : curr_header.stamp - prev_stamp;
  const auto dt = delta_duration.toSec();
  ROS_INFO_STREAM("dt: " << dt * 1000 << " ms");

  // Motion model
  Sophus::SE3d dtf_pred;
  if (dt > 0) {
    // Do a const vel prediction first
    dtf_pred = motion_.PredictDelta(dt);

    // Then overwrite rotation part if we have imu
    // TODO(dsol): Use 0th order integration, maybe switch to 1st order later
    ROS_INFO_STREAM(
        fmt::format("prev: {}, curr: {}, first_imu: {}, last_imu: {}",
                    prev_stamp.toSec(),
                    curr_header.stamp.toSec(),
                    gyrs_.front().header.stamp.toSec(),
                    gyrs_.back().header.stamp.toSec()));
    Sophus::SO3d dR{};
    int n_imus = 0;
    for (size_t i = 0; i < gyrs_.size(); ++i) {
      const auto& imu = gyrs_[i];
      // Skip imu msg that is earlier than the previous odom
      if (imu.header.stamp <= prev_stamp) continue;
      if (imu.header.stamp > curr_header.stamp) continue;

      const auto prev_imu_stamp =
          i == 0 ? prev_stamp : gyrs_.at(i - 1).header.stamp;
      const double dt_imu = (imu.header.stamp - prev_imu_stamp).toSec();
      CHECK_GT(dt_imu, 0);
      Eigen::Map<const Eigen::Vector3d> w(&imu.angular_velocity.x);
      dR *= Sophus::SO3d::exp(w * dt_imu);
      ++n_imus;
    }
    ROS_INFO_STREAM("n_imus: " << n_imus);
    // We just replace const vel prediction
    if (n_imus > 0) dtf_pred.so3() = dR;
  }

  // TODO (yanwei) Temporary use odom for rotation.
  // if (false && dt > 0) {
  //   try {
  //     geometry_msgs::TransformStamped odom_to_cam = tf_buffer_.lookupTransform(
  //         odom_frame_, camera_frame_, prev_stamp, ros::Duration(1.0));
  //     const Sophus::SE3d last_T = Ros2Sophus(odom_to_cam.transform);

  //     odom_to_cam = tf_buffer_.lookupTransform(
  //         odom_frame_, camera_frame_, curr_header.stamp, ros::Duration(1.0));
  //     const Sophus::SE3d cur_T = Ros2Sophus(odom_to_cam.transform);
  //     dtf_pred = last_T.inverse() * cur_T;
  //     std::cout << dtf_pred.matrix() << std::endl;
  //   } catch (tf2::TransformException e) {
  //     ROS_WARN_STREAM(
  //         "tf execption caught when looking for odom to camera "
  //         "transformation.\n"
  //         << e.what());
  //   }
  // }

  const auto status = odom_.Estimate(
      curr_header.stamp.toSec(), image0, image1, dtf_pred, depth0);
  ROS_INFO_STREAM(status.Repr());

  // Motion model correct if tracking is ok and not first frame
  if (status.track.ok) {
    motion_.Correct(status.Twc(), dt);
  } else {
    ROS_WARN_STREAM("Tracking failed (or 1st frame), slow motion model");
  }

  // Write to output
  writer_.Write(
      curr_header.stamp.toSec(), status.Twc(), status.track.Twc_prior);

  // publish stuff
  std_msgs::Header header;
  header.frame_id = fixed_frame_;
  header.stamp = curr_header.stamp;

  PublishOdom(header, status.Twc());
  if (status.map.remove_kf) {
    PublishCloud(header);
  }
  PublishDisplayImage(header, status.disp_frame);

  prev_stamp = curr_header.stamp;
}

void NodeOdom::PublishOdom(const std_msgs::Header& header,
                           const Sophus::SE3d& tf) {
  // Publish odom poses
  const auto pose_msg = pub_odom_.Publish(header.stamp, tf);

  if (publish_tf_) {
    SendTransform(header.stamp, tf);
  }

  // Publish camera pose in imu frame
  {
    Sophus::SE3d Tic(
        (Eigen::Matrix3d() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished(),
        Eigen::Vector3d::Zero());
    geometry_msgs::PoseWithCovarianceStamped camera_pose_in_imu;
    camera_pose_in_imu.header.frame_id = map_frame_;
    camera_pose_in_imu.header.stamp = header.stamp;
    Sophus2Ros(Tic * tf, camera_pose_in_imu.pose.pose);
    pub_camera_pose_in_imu_.publish(camera_pose_in_imu);

    // nav_msgs::Odometry camera_odom_in_imu;
    // camera_odom_in_imu.header.frame_id = "map";
    // camera_odom_in_imu.header.stamp = header.stamp;
    // Sophus2Ros(Tci * tf, camera_odom_in_imu.pose.pose);
    // mpCameraPoseInIMUOdometryPublisher_.publish(camera_odom_in_imu);
  }

  // Publish keyframe poses
  const auto poses = odom_.window.GetAllPoses();
  gm::PoseArray parray_msg;
  parray_msg.header = header;
  parray_msg.poses.resize(poses.size());
  for (size_t i = 0; i < poses.size(); ++i) {
    Sophus2Ros(poses.at(i), parray_msg.poses.at(i));
  }
  pub_parray_.publish(parray_msg);
}

void NodeOdom::PublishCloud(const std_msgs::Header& header) {
  if (pub_points_.getNumSubscribers() == 0) return;

  cloud_.header = header;
  cloud_.point_step = 16;
  cloud_.fields = MakePointFields("xyzi");

  ROS_DEBUG_STREAM(odom_.window.MargKf().status().Repr());
  Keyframe2Cloud(odom_.window.MargKf(), cloud_, 50.0);
  pub_points_.publish(cloud_);
}

// void NodeOdom::TfCamCb(const geometry_msgs::Transform& tf_cam_msg) {
//   odom_.camera.baseline_ = -tf_cam_msg.translation.x;
//   ROS_INFO_STREAM(odom_.camera.Repr());
// }

// void NodeOdom::TfImuCb(const geometry_msgs::Transform& tf_imu_msg) {}

void NodeOdom::PublishDisplayImage(const std_msgs::Header& header,
                                   const cv::Mat& image) {
  if (pub_disp_frame_.getNumSubscribers() == 0) {
    return;
  }
  sm::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
  pub_disp_frame_.publish(msg);
}

void NodeOdom::RobotWheelOdomCb(const nav_msgs::OdometryConstPtr& msg) {
  geometry_msgs::PoseStamped pose_stamped;
  pose_stamped.header = msg->header;
  pose_stamped.pose = msg->pose.pose;
  robot_wheel_path_msg_.poses.push_back(pose_stamped);
  robot_wheel_path_msg_.header = msg->header;
  pub_robot_wheel_path_.publish(robot_wheel_path_msg_);
}

void NodeOdom::SendTransform(const ros::Time& time, const Sophus::SE3d& Twc) {
  if (base_frame_.empty()) {
    // Compose fixed_frame to camera.
    // fixed_frame aligns with camera optical frame. (z points forward)
    geometry_msgs::TransformStamped fixed_to_cam;
    fixed_to_cam.header.frame_id = fixed_frame_;
    fixed_to_cam.header.stamp = time;
    fixed_to_cam.child_frame_id = camera_frame_;
    Sophus2Ros(Twc, fixed_to_cam.transform);
    tfbr_.sendTransform(fixed_to_cam);
  } else {
    // Define map that aligns with base_frame
    static bool is_map_defined = false;
    static Sophus::SE3d Tmw;
    if (!is_map_defined) {
      geometry_msgs::TransformStamped base_to_cam = tf_buffer_.lookupTransform(
          base_frame_, camera_frame_, time, ros::Duration(0.2));
      Tmw = Ros2Sophus(base_to_cam.transform);
      writer_.Write("camera_extrinsic", Tmw);

      base_to_cam.header.frame_id = map_frame_;
      base_to_cam.header.stamp = time;
      base_to_cam.child_frame_id = fixed_frame_;
      static_br_.sendTransform(base_to_cam);
      is_map_defined = true;
    }
    if (!publish_map_to_odom_tf_) {
      return;
    }
    // Publish map to odom.
    const Sophus::SE3d Tmc = Tmw * Twc;
    try {
      const geometry_msgs::TransformStamped cam_to_odom =
          tf_buffer_.lookupTransform(
              camera_frame_, odom_frame_, time, ros::Duration(1.0));
      geometry_msgs::TransformStamped map_to_odom;
      map_to_odom.header.frame_id = map_frame_;
      map_to_odom.header.stamp = time + ros::Duration(0.5);
      map_to_odom.child_frame_id = odom_frame_;
      Sophus2Ros(Tmc * Ros2Sophus(cam_to_odom.transform),
                 map_to_odom.transform);
      tfbr_.sendTransform(map_to_odom);
    } catch (tf2::TransformException e) {
      ROS_WARN_STREAM(
          "tf execption caught when looking for odom to camera "
          "transformation.\n"
          << e.what());
    }
  }
}

}  // namespace sv::dsol

int main(int argc, char** argv) {
  ros::init(argc, argv, "dsol_odom");
  cv::setNumThreads(4);
  // glog settings
  {
    // Init google logging.
    google::InitGoogleLogging(argv[0]);

    // Init ros.
    ros::NodeHandle nh("~");

    // Loading paramters.
    nh.param<int>("min_log_level", FLAGS_minloglevel, 0);
    // int also_log_to_console = 1;
    // nh.param<int>("also_log_to_console", also_log_to_console, 1);
    nh.param<bool>("also_log_to_console", FLAGS_alsologtostderr, true);
    // FLAGS_alsologtostderr = also_log_to_console;
    nh.param<int>("verbose_logging", FLAGS_v, 1);

    // Set log directory.
    std::string log_dir = "/tmp/dsol_logging";
    nh.getParam("save", log_dir);
    FLAGS_log_dir = log_dir;
    google::SetLogDestination(google::FATAL, (log_dir + ".FATAL.").c_str());
    google::SetLogDestination(google::ERROR, (log_dir + ".ERROR.").c_str());
    google::SetLogDestination(google::WARNING, (log_dir + ".WARNING.").c_str());
    google::SetLogDestination(google::INFO, (log_dir + ".INFO.").c_str());

    // Output stream.
    ROS_INFO(
        "glog-logging: (\n min_log_level = %d, \n also_log_to_console = "
        "%d, \n verbose_logging = %d, \n log_dir_prefix = %s).",
        FLAGS_minloglevel,
        FLAGS_alsologtostderr,
        FLAGS_v,
        log_dir.c_str());
  }
  sv::dsol::NodeOdom node{ros::NodeHandle{"~"}};
  ros::spin();
}
