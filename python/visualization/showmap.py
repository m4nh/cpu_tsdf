#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import PyKDL
import glob
import os
import pprint
import cv2
import sys
import numpy as np
import math
from roars.rosutils.rosnode import RosNode
import roars.geometry.transformations as transformations
import roars.vision.colors as colors
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import random
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, centroid, fclusterdata
# from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


def quaternion_from_matrix(matrix, isprecise=False):
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0,         0.0,         0.0],
                      [m01 + m10,     m11 - m00 - m22, 0.0,         0.0],
                      [m02 + m20,     m12 + m21,     m22 - m00 - m11, 0.0],
                      [m21 - m12,     m02 - m20,     m10 - m01,     m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def createBoxMarker(name, box, parent):
    #⬢⬢⬢⬢⬢➤ Map Marker
    marker = Marker()
    marker.header.frame_id = parent
    marker.type = Marker.CUBE
    marker.id = 0  # random.randint(0, 100000)
    marker.action = marker.ADD
    marker.ns = name
    marker.pose.orientation.w = 1

    # sx = math.fabs(box[0, 0] - box[0, 1])
    # sy = math.fabs(box[1, 0] - box[1, 1])
    # sz = math.fabs(box[2, 0] - box[2, 1])

    # px = box[0, 1] - sx * 0.5
    # py = box[1, 1] - sy * 0.5
    # pz = box[2, 1] - sz * 0.5

    # marker.pose.position.x = px
    # marker.pose.position.y = py
    # marker.pose.position.z = pz

    marker.scale.x = math.fabs(box[0, 0] - box[0, 1])
    marker.scale.y = math.fabs(box[1, 0] - box[1, 1])
    marker.scale.z = math.fabs(box[2, 0] - box[2, 1])
    marker.color.r = 1
    marker.color.g = 1
    marker.color.b = 1
    marker.color.a = 0.3
    return marker


class Instance(object):

    def __init__(self, points, voxels, label):
        self.label = label
        self.points = points
        self.voxels = voxels
        self.center = Clusters.computeCentroid(self.points)
        _, _, self.rf = self.svd()
        self.transform = np.eye(4, dtype=float)
        self.heavier_label = int(self.getHeavierLabel())

    def getHeavierLabel(self):
        label_map = {}
        for v in self.voxels:
            for label, w in v.label_map.iteritems():
                if label not in label_map:
                    label_map[label] = 0.0
                label_map[label] += w
        max_l = -1
        max_w = -1
        for label, w in label_map.iteritems():
            if w > max_w:
                max_w = w
                max_l = label
        return max_l

    def getRF(self):

        self.rf[:, 0] = self.rf[:, 0] / np.linalg.norm(self.rf[:, 0])
        self.rf[:, 1] = self.rf[:, 1] / np.linalg.norm(self.rf[:, 1])
        self.rf[:, 2] = self.rf[:, 2] / np.linalg.norm(self.rf[:, 2])

        frame = PyKDL.Frame(PyKDL.Vector(
            self.center[0], self.center[1], self.center[2]
        ))
        q = quaternion_from_matrix(np.array([
            self.rf[0, 0], self.rf[0, 1], self.rf[0, 2], 0,
            self.rf[1, 0], self.rf[1, 1], self.rf[1, 2], 0,
            self.rf[2, 0], self.rf[2, 1], self.rf[2, 2], 0,
            0, 0, 0, 1]).reshape((4, 4)), isprecise=True
        )
        # self.transform = np.array([
        #     self.rf[0, 0], self.rf[0, 1], self.rf[0, 2], self.center[0],
        #     self.rf[1, 0], self.rf[1, 1], self.rf[1, 2], self.center[1],
        #     self.rf[2, 0], self.rf[2, 1], self.rf[2, 2], self.center[2],
        #     0, 0, 0, 1
        # ]).reshape((4, 4))
        q = q / np.linalg.norm(q)
        frame.M = PyKDL.Rotation.Quaternion(q[3], q[0], q[1], q[2])
        self.transform = transformations.KLDtoNumpyMatrix(frame)
        return frame  # transformations.NumpyMatrixToKDL(self.transform)

    def getBBox(self):
        hpoints = np.array(self.points).reshape(len(self.points), 3)
        ones = np.ones((len(self.points), 1))
        hpoints = np.hstack((hpoints, ones))
        hpoints = np.matmul(np.linalg.inv(self.transform), hpoints.T)
        hpoints = hpoints.T

        transformed_points = hpoints[:, 0:3]

        mins = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
        maxs = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
        for i in range(0, 3):
            col = transformed_points[:, i]
            mins[i] = np.amin(col)
            maxs[i] = np.amax(col)

        box = np.hstack((mins, maxs))
        return box

    def getBoxCorrection(self):
        self.getRF()
        box = self.getBBox()

        sx = math.fabs(box[0, 0] - box[0, 1])
        sy = math.fabs(box[1, 0] - box[1, 1])
        sz = math.fabs(box[2, 0] - box[2, 1])

        px = box[0, 1] - sx * 0.5
        py = box[1, 1] - sy * 0.5
        pz = box[2, 1] - sz * 0.5
        return PyKDL.Frame(PyKDL.Vector(
            px, py, pz
            #0, 0, 0
        ))

    def getBoxRF(self):
        rf = self.getRF()
        box = self.getBBox().ravel()
        corr = self.getBoxCorrection()
        return rf * corr

    def size(self):
        return len(self.points)

    def svd(self):
        # pca = PCA(n_components=3)
        # pca.fit(np.array(self.points))
        # base = np.eye(3, dtype=float)
        shifted = np.subtract(self.points, self.center)
        svd = np.linalg.svd(np.array(shifted))
        return svd

    def shortRepresentation(self):
        box = self.getBBox()
        sx = math.fabs(box[0, 0] - box[0, 1])
        sy = math.fabs(box[1, 0] - box[1, 1])
        sz = math.fabs(box[2, 0] - box[2, 1])

        size = np.array([sx, sy, sz]).reshape((1, 3))

        box_rf = transformations.KDLtoNumpyVector(self.getBoxRF()).reshape((1, 7))
        whole = np.concatenate((np.array([self.heavier_label]), box_rf.ravel(), size.ravel()))
        return whole

    def toString(self):
        return ' '.join(map(str, self.shortRepresentation()))


class Clusters(object):

    def __init__(self):
        self.points = []
        self.voxels = []
        self.labels = []
        self.objects_map = {}
        self.points_map = {}
        self.voxels_map = {}

    def appendVoxel(self, voxel):
        self.voxels.append(voxel)
        self.appendPoint(voxel.pos)

    def appendPoint(self, p):
        self.points.append(p)

    @staticmethod
    def computeCentroid(point_set):
        center = np.array([0.0, 0.0, 0.0])
        for p in point_set:
            center += p
        center = center * (1.0 / float(len(point_set)))
        return center

    def buildClusters(self, th=0.05, min_cluster_size=0):
        self.labels = fclusterdata(self.points, th, criterion='distance')
        for i in range(0, len(self.labels)):
            label = self.labels[i]
            if label not in self.points_map:
                self.points_map[label] = []
                self.voxels_map[label] = []
            self.points_map[label].append(self.points[i])
            self.voxels_map[label].append(self.voxels[i])

        for l, points in self.points_map.iteritems():
            if len(points) >= min_cluster_size:
                self.objects_map[l] = Instance(points, self.voxels_map[l], l)

        return self.labels


class MultiLabelVoxel(object):
    BACKGROUND_LABEL = -1
    CLASSES_COLOR_MAP = {
        -1: 'white',
        0: 'red',
        1: 'blue',
        2: 'teal',
        3: 'lime',
        4: 'purple',
        5: 'green',
        6: 'cyan'
    }

    def __init__(self, data):
        self.data = data
        self.pos = np.array(data[0: 3])
        self.size = int(data[3])
        self.label_map = {}

        for i in range(4, 4 + self.size * 2, 2):
            l = data[i]
            w = data[i + 1]
            self.label_map[l] = w

    def heavierLabel(self):
        max_w = 0
        max_l = -1
        for l, w in self.label_map.iteritems():
            if w > max_w:
                max_l = l
                max_w = w
        return (max_l, max_w)

    def isBackground(self):
        return self.containsLabel(MultiLabelVoxel.BACKGROUND_LABEL)

    def containsLabel(self, label):
        return label in self.label_map

    def getColor(self):
        if self.size == 1:
            l, w = self.heavierLabel()
            return colors.getColor(MultiLabelVoxel.colorFromLabel(l))
        else:
            if self.containsLabel(-1):
                # l, w = self.heavierLabel()
                # if l == -1:
                return colors.getColor(MultiLabelVoxel.colorFromLabel(-1))
                # else:
                #     return colors.getColor(MultiLabelVoxel.colorFromLabel(l))
            else:
                return (0, 0, 0)

    @staticmethod
    def colorFromLabel(label):
        try:
            return MultiLabelVoxel.CLASSES_COLOR_MAP[label]
        except:
            return 'black'


node = RosNode("ciao", disable_signals=True)

map_file = node.setupParameter("map_file", "")
min_w = node.setupParameter("min_w", 0)
min_cluster_size = node.setupParameter("min_cluster_size", 0)
debug = node.setupParameter("debug", True)
is_gt = node.setupParameter("is_gt", False)
output_path = node.setupParameter("output_path", "/tmp/daniele/")
output_name = node.setupParameter("output_name", "")
#⬢⬢⬢⬢⬢➤ Map Rototranslation
roll = node.setupParameter("roll", 0.0)
pitch = node.setupParameter("pitch", 0.0)
yaw = node.setupParameter("yaw", 0.0)
map_transform = PyKDL.Frame()
map_transform.M = PyKDL.Rotation.RPY(roll, pitch, yaw)


#⬢⬢⬢⬢⬢➤ Visualization topic
vis_topic = node.createPublisher('/showmap/map', MarkerArray)


#⬢⬢⬢⬢⬢➤ Read Map file
ff = open(map_file, 'r')
lines = ff.readlines()
voxels = []
skip_counter = -1
map_resolution = 0.01

for l in lines:
    skip_counter += 1
    if skip_counter == 1:
        data = np.fromstring(l, sep=' ')
        map_resolution = float(data[2])
        print "Set resolution", data, data[2]
    if skip_counter > 1:
        data = np.fromstring(l, sep=' ')
        vox = MultiLabelVoxel(data.ravel())
        voxels.append(vox)


#⬢⬢⬢⬢⬢➤ Map Marker
marker = Marker()
marker.header.frame_id = "world"
marker.type = Marker.CUBE_LIST
marker.id = 100
marker.action = marker.ADD
marker.ns = "map"

marker.scale.x = map_resolution
marker.scale.y = map_resolution
marker.scale.z = map_resolution
print "Resolutiion", map_resolution

#⬢⬢⬢⬢⬢➤ Centroid
center = np.array([0.0, 0.0, 0.0])
center_counter = 0.0
max_global_weight = 0.0
for voxel in voxels:
    center = center + voxel.pos
    center_counter += 1.0
    l, w = voxel.heavierLabel()
    if w > max_global_weight:
        max_global_weight = w
center = center * (1.0 / center_counter)

if float(min_w) != int(min_w):
    min_w = max_global_weight * min_w
    print("USING PERCENTAGE WEIGHT", min_w)


if debug:
    print "Max global weight:", max_global_weight

#⬢⬢⬢⬢⬢➤ Clusters
clusters = Clusters()

#⬢⬢⬢⬢⬢➤ Marker Fill
for voxel in voxels:

    (l, w) = voxel.heavierLabel()
    is_probable_background = voxel.containsLabel(-1)

    label_color = voxel.getColor()

    if w > min_w:

        if not voxel.isBackground():
            clusters.appendVoxel(voxel)

        p = PyKDL.Vector(voxel.pos[0] - center[0], voxel.pos[1] - center[1], voxel.pos[2] - center[2])
        # p = map_transform * p

        point = Point()
        point.x = voxel.pos[0]
        point.y = voxel.pos[1]
        point.z = voxel.pos[2]
        color = ColorRGBA()
        color.r = label_color[0] / 255.0
        color.g = label_color[1] / 255.0
        color.b = label_color[2] / 255.0
        color.a = 1  # float(w) / max_global_weight

        marker.points.append(point)
        marker.colors.append(color)

clusters.buildClusters(th=0.025, min_cluster_size=min_cluster_size)

if debug:
    while node.isActive():
        marker_array = MarkerArray()
        marker_array.markers.append(marker)

        for l, instance in clusters.objects_map.iteritems():
            cluster_name = "Cluster_{}_{}".format(l, instance.heavier_label)
            frame = instance.getBoxRF()
            #frame.M = PyKDL.Rotation()

            box = instance.getBBox()
            marker_array.markers.append(createBoxMarker(
                "Box_{}".format(l),
                box,
                cluster_name
            ))
            node.broadcastTransform(frame, cluster_name, "world", node.getCurrentTime())
            print "INstance"
            print instance.toString()

        for m in marker_array.markers:
            m.header.stamp = node.getCurrentTime()
        vis_topic.publish(marker_array)
        node.tick()
else:
    try:
        os.mkdir(os.path.join(output_path, output_name))
    except:
        pass

    instance_list = []
    counter = 0
    for l, instance in clusters.objects_map.iteritems():
        instance_list.append(instance.shortRepresentation())
        short = instance.shortRepresentation()
        name = "instance_{}_{}.txt".format(str(counter).zfill(3), int(short[0]))
        name = os.path.join(output_path, output_name, name)
        np.savetxt(name, instance.points)
        counter += 1

    instance_list = np.array(instance_list)

    #out_name = "instances_" + os.path.basename(map_file) + "_{}{}.txt".format(min_w, '_GT' if is_gt else '')
    instances_output_path = os.path.join(output_path, output_name, "boxes.txt")
    np.savetxt(instances_output_path, instance_list)
