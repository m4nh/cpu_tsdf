/* 
 * Copyright (C) 2017 daniele de gregorio, University of Bologna - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the GNU GPLv3 license.
 *
 * please write to: d.degregorio@unibo.it
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cfloat>

#include <Eigen/Core>
//PCL

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>

//OPENCV
#include <opencv2/opencv.hpp>

#include <Eigen/Core>

//Skimap
#include <skimap/SkiMap.hpp>
#include <skimap/SkipListMapV2.hpp>

#include <skimap/voxels/VoxelDataRGBW.hpp>
#include <skimap/voxels/VoxelDataMultiLabel.hpp>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <args.hxx>

#define MAX_RANDOM_COORD 10.0
#define MIN_RANDOM_COORD -10.0
#define MAX_RANDOM_COLOR 1.0
#define MIN_RANDOM_COLOR 0.0

typedef pcl::PointXYZRGB PointType;
using namespace std;

/** 
 * skimap definition
 */
typedef float CoordinateType;
typedef int16_t IndexType;
typedef int8_t LabelType;
typedef float WeightType;
//typedef skimap::VoxelDataRGBW<WeightType, WeightType> VoxelDataColor;
typedef skimap::VoxelDataMultiLabel<LabelType, WeightType> VoxelData;

typedef skimap::SkipListMapV2<VoxelData, IndexType, CoordinateType> SKIMAP;
typedef skimap::SkipListMapV2<VoxelData, IndexType, CoordinateType>::Voxel3D Voxel3D;
SKIMAP *voxel_grid;

int convertLabel(int l_rgbd)
{

    std::vector<int> label_maps(11);
    label_maps[0] = -1;
    label_maps[1] = 0;
    label_maps[2] = 2;
    label_maps[3] = 3;
    label_maps[4] = 4;
    label_maps[5] = -1;
    label_maps[6] = -1;
    label_maps[7] = 6;
    label_maps[8] = -1;
    label_maps[9] = -1;
    label_maps[10] = -1;

    // std::cout << "Converting: " << l_rgbd << " -> " << label_maps[l_rgbd] << " \n";
    return label_maps[l_rgbd];
}
int main(int argc, const char **argv)
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<string> cloud_path(parser, "cloud_path", "Cloud path", {"cloud_path"});
    args::ValueFlag<string> labels_path(parser, "labels_path", "Labels path", {"labels_path"}, "");
    args::ValueFlag<string> output_path(parser, "output_path", "Output path", {"output_path"}, "");
    // args::ValueFlag<string> output_path(parser, "output path", "Output path", {"output_path"});
    // args::ValueFlag<string> color_tag(parser, "rgb tag", "Tags for color information", {"rgb_tag"});
    // args::ValueFlag<string> depth_tag(parser, "depth tag", "Tags for depth information", {"depth_tag"}, "depth");
    // args::ValueFlag<string> pose_tag(parser, "pose tag", "Tags for pose information", {"pose_tag"}, "pose");
    // args::ValueFlag<int> zfill(parser, "zfill", "Zero fills", {"zfill"}, 5);
    // args::ValueFlag<int> downsample(parser, "downsample", "Downsample frames number", {"downsample"}, 1);
    // args::ValueFlag<double> max_distance(parser, "max distance", "Camera Max Distance", {"max_z"}, 1.0);
    // args::ValueFlag<double> min_distance(parser, "min distance", "Camera Min Distance", {"min_z"}, 0.3);
    args::ValueFlag<double> resolution(parser, "resolution", "Map Resolution", {"resolution"}, 0.005);
    // args::ValueFlag<double> fx(parser, "fx", "Camera Focal x", {"fx"}, 538.399364);
    // args::ValueFlag<double> fy(parser, "fy", "Camera Focal y", {"fy"}, 538.719401);
    // args::ValueFlag<double> cx(parser, "cx", "Camera Cx", {"cx"}, 316.192117);
    // args::ValueFlag<double> cy(parser, "cy", "Camera Cy", {"cy"}, 240.080174);
    // args::ValueFlag<double> depth_rescale(parser, "depth rescale", "Depth Rescale", {"depth_rescale"}, 1.0);

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }

    std::string label_filename = args::get(labels_path);
    std::cout << "Loading labels: " << label_filename << "\n";
    std::fstream in(label_filename.c_str());
    std::string line;
    std::vector<int> labels;
    int i = 0;
    int labels_size = -1;
    while (std::getline(in, line))
    {
        int value;
        std::stringstream ss(line);
        ss >> value;
        if (i == 0)
        {
            labels_size = value;
        }
        else
        {
            labels.push_back(convertLabel(value));
        }
        i++;
    }
    std::cout << "Loaded: " << labels_size << "/" << int(labels.size()) << "\n";

    //Skimap
    voxel_grid = new SKIMAP(args::get(resolution));

    pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
    std::string filename = args::get(cloud_path);
    pcl::io::loadPLYFile(filename, *cloud);

    for (int i = 0; i < cloud->size(); i++)
    {
        PointType &point = cloud->points[i];
        if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
        {
            continue;
        }

        LabelType label = labels[i];

        VoxelData voxel(label, 1.0);
        voxel_grid->integrateVoxel(
            CoordinateType(point.x),
            CoordinateType(point.y),
            CoordinateType(point.z),
            &voxel);
    }
    std::cout << "DONE!\n";
    voxel_grid->saveToFile(args::get(output_path));
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
