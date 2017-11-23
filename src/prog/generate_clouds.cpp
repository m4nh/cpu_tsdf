/*
 * Copyright (c) 2013-, Stephen Miller
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, 
 * with or without modification, are permitted provided 
 * that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the 
 * above copyright notice, this list of conditions 
 * and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the 
 * above copyright notice, this list of conditions and 
 * the following disclaimer in the documentation and/or 
 * other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the 
 * names of its contributors may be used to endorse or
 * promote products derived from this software without 
 * specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS 
 * AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
 * THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF 
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER 
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 * THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <pcl/console/print.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/pcl_macros.h>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <opencv2/opencv.hpp>
#include <args.hxx>

#include <string>
#include <vector>
#include <iomanip>

/**
 */
struct CameraParameters
{
  double fx, fy, cx, cy;
  int cols, rows;
  double min_distance;
  double max_distance;
  int point_cloud_downscale;
} camera;

typedef pcl::PointXYZRGB PointType;

void extractPointCloud(cv::Mat &rgb, cv::Mat &depth, CameraParameters camera, pcl::PointCloud<PointType>::Ptr &cloud)
{

  cloud->width = rgb.cols;
  cloud->height = rgb.rows;
  cloud->resize(rgb.cols * rgb.rows);

  for (float i = 0; i < depth.rows; i += 1)
  {
    for (float j = 0; j < depth.cols; j += 1)
    {
      float d = float(depth.at<short>(i, j)) / 1000.0;
      double x = (d / camera.fx) * (j - camera.cx);
      double y = (d / camera.fy) * (i - camera.cy);
      double z = d;
      if (z != z)
        continue;

      cv::Vec3b color = rgb.at<cv::Vec3b>(i, j);
      cloud->points[j + i * rgb.cols].x = x;
      cloud->points[j + i * rgb.cols].y = y;
      cloud->points[j + i * rgb.cols].z = z;
      cloud->points[j + i * rgb.cols].r = color[2];
      cloud->points[j + i * rgb.cols].g = color[1];
      cloud->points[j + i * rgb.cols].b = color[0];

      //std::cout << x << "," << y << "," << z << "\n";
      //if (cp.point.z >= camera.min_distance && cp.point.z <= camera.max_distance)
      //{

      //}
    }
  }
}

using namespace std;

void extractFromFolder(string folder_name, string tag, vector<string> &output_files)
{
  bool reverse = false;
  if (boost::starts_with(tag, "*"))
  {
    tag.erase(0, 1);
    reverse = true;
  }

  boost::filesystem::directory_iterator end_itr;
  for (boost::filesystem::directory_iterator itr(folder_name); itr != end_itr; ++itr)
  {
    std::string extension = boost::filesystem::extension(itr->path());
    std::string pathname = itr->path().string();

    if (!reverse)
    {
      if (pathname.find(tag) != std::string::npos)
      {
        output_files.push_back(pathname);
      }
    }
    else
    {
      if (pathname.find(tag) == std::string::npos)
      {
        output_files.push_back(pathname);
      }
    }
  }
  std::sort(output_files.begin(), output_files.end());
}

struct RGBDDataset
{
  vector<string> rgb_files;
  vector<string> depth_files;
  vector<string> pose_files;
  void clear()
  {
    rgb_files.clear();
    depth_files.clear();
  }

  bool isValid()
  {
    return rgb_files.size() == depth_files.size();
  }

  pcl::PointCloud<PointType>::Ptr generatePointCloud(int index, CameraParameters camera)
  {
    cv::Mat depth = cv::imread(depth_files[index], CV_LOAD_IMAGE_ANYDEPTH);
    cv::Mat rgb = cv::imread(rgb_files[index]);

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    extractPointCloud(rgb, depth, camera, cloud);
    return cloud;
  }

  static void generateRGBDataset(string path, string rgb_tag, string depth_tag, string pose_tag, RGBDDataset &output_dataset)
  {
    output_dataset.clear();
    extractFromFolder(path, rgb_tag, output_dataset.rgb_files);
    extractFromFolder(path, depth_tag, output_dataset.depth_files);
    extractFromFolder(path, pose_tag, output_dataset.pose_files);
  }
};

int main(int argc, const char **argv)
{
  args::ArgumentParser parser("This is a test program.", "This goes after the options.");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<string> dataset_path(parser, "path", "Dataset path", {"path"});
  args::ValueFlag<string> output_path(parser, "output path", "Output path", {"output_path"});
  args::ValueFlag<string> color_tag(parser, "rgb tag", "Tags for color information", {"rgb_tag"});
  args::ValueFlag<string> depth_tag(parser, "depth tag", "Tags for depth information", {"depth_tag"}, "depth");
  args::ValueFlag<string> pose_tag(parser, "pose tag", "Tags for pose information", {"pose_tag"}, "pose");
  args::ValueFlag<int> zfill(parser, "zfill", "Zero fills", {"zfill"}, 5);
  args::ValueFlag<int> downsample(parser, "downsample", "Downsample frames number", {"downsample"}, 1);
  args::ValueFlag<double> fx(parser, "fx", "Camera Focal x", {"fx"}, 538.399364);
  args::ValueFlag<double> fy(parser, "fy", "Camera Focal y", {"fy"}, 538.719401);
  args::ValueFlag<double> cx(parser, "cx", "Camera Cx", {"cx"}, 316.192117);
  args::ValueFlag<double> cy(parser, "cy", "Camera Cy", {"cy"}, 240.080174);

  camera.fx = args::get(fx);
  camera.fy = args::get(fy);
  camera.cx = args::get(cx);
  camera.cy = args::get(cy);

  try
  {
    parser.ParseCLI(argc, argv);
  }
  catch (args::Help)
  {
    std::cout << parser;
    return 0;
  }

  //INput path
  std::string path = args::get(dataset_path);

  //Output path
  boost::filesystem::path out_path(string(args::get(output_path)));
  boost::system::error_code returnedError;
  boost::filesystem::create_directories(out_path, returnedError);

  //Tags
  std::string label_tag = args::get(color_tag);

  int store_counter = 0;
  int jumps = args::get(downsample);

  RGBDDataset dataset;
  RGBDDataset::generateRGBDataset(path, label_tag, args::get(depth_tag), args::get(pose_tag), dataset);

  if (!dataset.isValid())
  {
    cout << "Dataset invalid! rgb(" << dataset.rgb_files.size() << "," << dataset.depth_files.size() << ")\n";
    return 0;
  }
  for (int i = 0; i < dataset.rgb_files.size(); i++)
  {
    if (i % jumps != 0)
      continue;
    cout << dataset.rgb_files[i] << "-> " << dataset.depth_files[i] << "->" << dataset.pose_files[i] << "\n";

    std::stringstream ss;
    ss << setfill('0') << setw(int(args::get(zfill))) << store_counter;

    string current_name = ss.str();
    pcl::PointCloud<PointType>::Ptr cloud = dataset.generatePointCloud(i, camera);

    string cloud_name = current_name + ".pcd";
    string pose_name = current_name + ".txt";

    boost::filesystem::copy_file(dataset.pose_files[i], out_path / pose_name);

    boost::filesystem::path cloud_path = out_path / cloud_name;
    pcl::io::savePCDFileBinary(cloud_path.string(), *cloud);

    store_counter++;
  }

  // camera.fx = 538.399364;
  // camera.cx = 316.192117;
  // camera.fy = 538.719401;
  // camera.cy = 240.080174;

  // int jumps = 5;

  // std::string depths_path = "/home/daniele/Desktop/datasets/roars_2017/indust/indust_scene_4_dome/camera_depth_image_raw/";
  // std::string predictions_path = "/tmp/indust_scene_4_dome_raw_pix/images/";
  // std::string robot_poses_path = "/home/daniele/Desktop/datasets/roars_2017/indust/indust_scene_4_dome/tf#_comau_smart_six_link6.txt";

  // std::vector<std::string> depth_files;
  // std::vector<std::string> rgb_files;
  // std::vector<std::string> predictions_files;
  // std::vector<std::string> gt_files;

  // {
  //   boost::filesystem::directory_iterator end_itr;
  //   for (boost::filesystem::directory_iterator itr(depths_path); itr != end_itr; ++itr)
  //   {
  //     std::string extension = boost::filesystem::extension(itr->path());
  //     std::string pathname = itr->path().string();
  //     if (extension == ".png" || extension == ".png")
  //     {
  //       depth_files.push_back(pathname);
  //     }
  //   }
  // }

  // {
  //   boost::filesystem::directory_iterator end_itr;
  //   for (boost::filesystem::directory_iterator itr(predictions_path); itr != end_itr; ++itr)
  //   {
  //     std::string extension = boost::filesystem::extension(itr->path());
  //     std::string pathname = itr->path().string();
  //     if (extension == ".png" || extension == ".png")
  //     {
  //       if (pathname.find("inputs") != std::string::npos)
  //       {
  //         rgb_files.push_back(pathname);
  //       }
  //       if (pathname.find("outputs") != std::string::npos)
  //       {
  //         predictions_files.push_back(pathname);
  //       }
  //       if (pathname.find("targets") != std::string::npos)
  //       {
  //         gt_files.push_back(pathname);
  //       }
  //     }
  //   }
  // }

  // std::sort(depth_files.begin(), depth_files.end());
  // std::sort(rgb_files.begin(), rgb_files.end());
  // std::sort(gt_files.begin(), gt_files.end());
  // std::sort(predictions_files.begin(), predictions_files.end());

  // for (int i = 0; i < gt_files.size(); i++)
  // {
  //   if (i % jumps != 0)
  //     continue;

  //   std::cout << i << "\n";
  //   cv::Mat depth = cv::imread(depth_files[i], CV_LOAD_IMAGE_ANYDEPTH);
  //   cv::Mat rgb = cv::imread(rgb_files[i]);
  //   cv::resize(rgb, rgb, depth.size());

  //   cv::Mat prediction = cv::imread(predictions_files[i]);
  //   cv::resize(prediction, prediction, depth.size());

  //   cv::Mat gt = cv::imread(gt_files[i]);
  //   cv::resize(gt, gt, depth.size());

  //   pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
  //   extractPointCloud(rgb, depth, camera, cloud);

  //   std::string dest = std::string(5, '0').append(".pcd");

  //   std::stringstream ss;
  //   ss << setfill('0') << setw(5) << i;

  //   std::string output_name = "/tmp/clouds_export/output/" + ss.str() + ".pcd";
  //   std::cout << output_name << "\n";

  //   pcl::io::savePCDFileBinary(output_name, *cloud);
  //   // double min;
  //   // double max;
  //   // cv::minMaxIdx(depth, &min, &max);
  //   // std::cout << "Min: " << min << " Max:" << max << "\n";
  //   // cv::Mat adjMap;
  //   // cv::convertScaleAbs(depth, adjMap, 255 / max);

  //   // cv::imshow("depth", adjMap);
  //   // cv::imshow("rgb", rgb);
  //   // cv::imshow("pred", prediction);
  //   // cv::waitKey(0);
  // }

  // return (0);
}
