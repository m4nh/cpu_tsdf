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

//PCL

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

//OPENCV
#include <opencv2/opencv.hpp>

#include <Eigen/Core>

//Skimap
#include <skimap/SkiMap.hpp>
#include <skimap/SkipListMapV2.hpp>

#include <skimap/voxels/VoxelDataRGBW.hpp>

#include <boost/filesystem/convenience.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/predicate.hpp>

#define MAX_RANDOM_COORD 10.0
#define MIN_RANDOM_COORD -10.0
#define MAX_RANDOM_COLOR 1.0
#define MIN_RANDOM_COLOR 0.0

/** 
 * skimap definition
 */
typedef float CoordinateType;
typedef int16_t IndexType;
typedef uint16_t WeightType;
typedef skimap::VoxelDataRGBW<IndexType, WeightType> VoxelDataColor;
typedef skimap::SkipListMapV2<VoxelDataColor, IndexType, CoordinateType> SKIMAP;
typedef skimap::SkipListMapV2<VoxelDataColor, IndexType, CoordinateType>::Voxel3D Voxel3D;
SKIMAP *map;

/**
 * Helper function to obtain a random double in range.
 */
double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
using namespace std;

std::vector<Eigen::Affine3d> loadPoses(std::vector<std::string> &pose_files)
{
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

int main(int argc, char **argv)
{

    std::string scene_path = "/media/daniele/data/daniele/datasets/indust_maks/scene1_clouds_rgb";

    srand(time(NULL));

    std::vector<std::string> pcd_files;
    std::vector<std::string> pose_files;

    extractFromFolder(scene_path, "pcd", pcd_files);
    extractFromFolder(scene_path, "txt", pose_files);

    std::vector<Eigen::Affine3d> poses = loadPoses(pose_files);
    std::cout << "Poses: " << poses.size() << "/" << pose_files.size() << "\n";
    for (int i = 0; i < pcd_files.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_files[i], *cloud) == 0) //* load the file
        {
            //std::cout << "OK" << std::endl;
        }
    }

    BOOST_FOREACH (std::string filename, pose_files)
    {
        std::cout << filename << "\n";
    }

    //     //Builds the map
    //     float map_resolution = 0.05;
    //     map = new SKIMAP(map_resolution);

    //     /**
    //      * This command enables the Concurrent Access Self Management. If it is enabled
    //      * you can use OPENMP to call the method 'integrateVoxel' in a concurrent manner safely.
    //      */
    //     map->enableConcurrencyAccess();

    //     /**
    //      * With this two parameters we can simulate N_MEASUREMENTS sensor measurements
    //      * each of them with N_POINTS points
    //      */
    //     int N_MEASUREMENTS = 100;
    //     int N_POINTS = 640 * 480 / 2;

    //     for (int m = 0; m < N_MEASUREMENTS; m++)
    //     {
    // /**
    //          * Integration Timer
    //          */

    // #pragma omp parallel for
    //         for (int i = 0; i < N_POINTS; i++)
    //         {

    //             /**
    //              * Generates a random 3D Point
    //              */
    //             double x = fRand(MIN_RANDOM_COORD, MAX_RANDOM_COORD);
    //             double y = fRand(MIN_RANDOM_COORD, MAX_RANDOM_COORD);
    //             double z = fRand(MIN_RANDOM_COORD, MAX_RANDOM_COORD);

    //             /**
    //              * Creates a Voxel. In this case the voxel is a VoxelDataColor data
    //              * structure with r,g,b color information and a w weight. In general
    //              * the 'weight' is used to fuse voxel togheter: a positive weight 1.0
    //              * means an addiction, a negative weight -1.0 means a subtraction
    //              */
    //             VoxelDataColor voxel;
    //             voxel.r = fRand(MIN_RANDOM_COLOR, MAX_RANDOM_COLOR);
    //             voxel.g = fRand(MIN_RANDOM_COLOR, MAX_RANDOM_COLOR);
    //             voxel.b = fRand(MIN_RANDOM_COLOR, MAX_RANDOM_COLOR);
    //             voxel.w = 1.0;

    //             /**
    //              * Integration of custom voxel in the SkiMap data structure.
    //              */
    //             map->integrateVoxel(
    //                 CoordinateType(x),
    //                 CoordinateType(y),
    //                 CoordinateType(z),
    //                 &voxel);
    //         }

    //         /**
    //          * Map Visiting. With this command you can extract all voxels in SkiMap.
    //          * You can iterate on results to display Voxels in your viewer
    //          */
    //         std::vector<Voxel3D> voxels;
    //         map->fetchVoxels(voxels);

    //         printf("Map voxels: %d\n", int(voxels.size()));
    //     }
}