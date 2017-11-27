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

std::vector<Eigen::Affine3d> loadPoses(std::vector<std::string> &pose_files)
{
    using namespace std;

    std::vector<Eigen::Affine3d> poses(pose_files.size());
    for (size_t i = 0; i < pose_files.size(); i++)
    {
        ifstream f(pose_files[i].c_str());
        float v;
        Eigen::Matrix4d mat;
        mat(3, 0) = 0;
        mat(3, 1) = 0;
        mat(3, 2) = 0;
        mat(3, 3) = 1;
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                f >> v;
                mat(y, x) = static_cast<double>(v);
            }
        }
        f.close();
        poses[i] = mat;

        // Update units
        poses[i].matrix().topRightCorner<3, 1>() *= 1;
    }
    return poses;
}

int convertColorLabel(int r, int g, int b)
{
    typedef std::map<int, cv::Scalar> ColorMap;

    ColorMap colors;
    colors[0] = cv::Scalar(244, 67, 54);
    colors[1] = cv::Scalar(33, 150, 243);
    colors[2] = cv::Scalar(0, 150, 136);
    colors[3] = cv::Scalar(205, 220, 57);
    colors[4] = cv::Scalar(156, 39, 176);
    colors[5] = cv::Scalar(76, 175, 80);
    colors[6] = cv::Scalar(0, 188, 212);
    colors[-1] = cv::Scalar(255, 255, 255);

    for (ColorMap::iterator it = colors.begin(); it != colors.end(); ++it)
    {
        int label = it->first;
        cv::Scalar color = it->second;
        if (r == color[0] && g == color[1] && b == color[2])
        {
            return label;
        }
    }
    printf("Color %d,%d,%d not recognized!\n", r, g, b);
    return -1;

    // if (r == 244 && g == 67 & b == 54)
    // {
    //     return 0;
    // }
    // else if (r == 76 && g == 175 & b == 80)
    // {
    //     return 1;
    // }
    // else if (r == 0 && g == 188 & b == 212)
    // {
    //     return 2;
    // }
    // else if (r == 0 && g == 150 & b == 136)
    // {
    //     return 3;
    // }
    // else if (r == 156 && g == 39 & b == 176)
    // {
    //     return 4;
    // }
    // else if (r == 255 && g == 255 & b == 255)
    // {
    //     return -1;
    // }
    // else if (r == 205 && g == 220 && b == 57)
    // {
    //     return 5;
    // }
    // else
    // {
    //     printf("Color %d,%d,%d not recognized!\n", r, g, b);
    //     exit(0);
    // }
}

void extractPointCloud(cv::Mat &rgb, cv::Mat &depth, CameraParameters camera, pcl::PointCloud<PointType>::Ptr &cloud, double depth_rescale = 1.0)
{

    cloud->width = rgb.cols;
    cloud->height = rgb.rows;
    cloud->resize(rgb.cols * rgb.rows);

    // double min, max;
    // cv::minMaxLoc(depth, &min, &max);
    // std::cout << min << "," << max << "\n";
    // exit(0);

    for (float i = 0; i < depth.rows; i += 1)
    {
        for (float j = 0; j < depth.cols; j += 1)
        {
            float d = float(depth.at<short>(i, j)) / (1000.0 * depth_rescale);
            double x = (d / camera.fx) * (j - camera.cx);
            double y = (d / camera.fy) * (i - camera.cy);
            double z = d;
            if (z != z || z > camera.max_distance || z < camera.min_distance)
            {
                cloud->points[j + i * rgb.cols].x = std::numeric_limits<float>::quiet_NaN();
                cloud->points[j + i * rgb.cols].y = std::numeric_limits<float>::quiet_NaN();
                cloud->points[j + i * rgb.cols].z = std::numeric_limits<float>::quiet_NaN();
                continue;
            }
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
    std::vector<Eigen::Affine3d> poses;

    void clear()
    {
        rgb_files.clear();
        depth_files.clear();
    }

    void initPoses()
    {
        poses = loadPoses(pose_files);
    }

    bool isValid()
    {
        return rgb_files.size() == depth_files.size();
    }

    pcl::PointCloud<PointType>::Ptr generatePointCloud(int index, CameraParameters camera, double depth_rescale = 1)
    {
        cv::Mat depth = cv::imread(depth_files[index], CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat rgb = cv::imread(rgb_files[index]);

        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
        extractPointCloud(rgb, depth, camera, cloud, depth_rescale);
        return cloud;
    }

    void integrate(SKIMAP *&grid, int index, CameraParameters camera, double depth_rescale = 1.0)
    {

        cv::Mat depth = cv::imread(depth_files[index], CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat rgb = cv::imread(rgb_files[index]);

        pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>());
        extractPointCloud(rgb, depth, camera, cloud, depth_rescale);

        pcl::transformPointCloud(*cloud, *cloud, poses[index]);

        int w = rgb.cols;
        int h = rgb.rows;

        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
            {
                PointType &point = (*cloud)(j, i);
                if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
                {
                    continue;
                }

                LabelType label = convertColorLabel(
                    point.r,
                    point.g,
                    point.b);

                VoxelData voxel(label, 1.0);
                grid->integrateVoxel(
                    CoordinateType(point.x),
                    CoordinateType(point.y),
                    CoordinateType(point.z),
                    &voxel);
            }
        }
    }

    static void generateRGBDataset(string path, string rgb_tag, string depth_tag, string pose_tag, RGBDDataset &output_dataset)
    {
        output_dataset.clear();
        extractFromFolder(path, rgb_tag, output_dataset.rgb_files);
        extractFromFolder(path, depth_tag, output_dataset.depth_files);
        extractFromFolder(path, pose_tag, output_dataset.pose_files);
        output_dataset.initPoses();
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
    args::ValueFlag<double> max_distance(parser, "max distance", "Camera Max Distance", {"max_z"}, 1.0);
    args::ValueFlag<double> min_distance(parser, "min distance", "Camera Min Distance", {"min_z"}, 0.3);
    args::ValueFlag<double> resolution(parser, "resolution", "Map Resolution", {"resolution"}, 0.005);
    args::ValueFlag<double> fx(parser, "fx", "Camera Focal x", {"fx"}, 538.399364);
    args::ValueFlag<double> fy(parser, "fy", "Camera Focal y", {"fy"}, 538.719401);
    args::ValueFlag<double> cx(parser, "cx", "Camera Cx", {"cx"}, 316.192117);
    args::ValueFlag<double> cy(parser, "cy", "Camera Cy", {"cy"}, 240.080174);
    args::ValueFlag<double> depth_rescale(parser, "depth rescale", "Depth Rescale", {"depth_rescale"}, 1.0);

    camera.fx = args::get(fx);
    camera.fy = args::get(fy);
    camera.cx = args::get(cx);
    camera.cy = args::get(cy);
    camera.max_distance = args::get(max_distance);
    camera.min_distance = args::get(min_distance);

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }

    //Skimap
    voxel_grid = new SKIMAP(args::get(resolution));

    //INput path
    std::string path = args::get(dataset_path);

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
        // pcl::PointCloud<PointType>::Ptr cloud = dataset.generatePointCloud(i, camera, args::get(depth_rescale));

        dataset.integrate(voxel_grid, i, camera, args::get(depth_rescale));

        // string cloud_name = current_name + ".pcd";
        // string pose_name = current_name + ".txt";

        // boost::filesystem::copy_file(dataset.pose_files[i], out_path / pose_name);

        // boost::filesystem::path cloud_path = out_path / cloud_name;
        // pcl::io::savePCDFileBinary(cloud_path.string(), *cloud);

        store_counter++;
    }

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
