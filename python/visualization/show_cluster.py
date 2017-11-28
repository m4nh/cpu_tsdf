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
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, centroid, fclusterdata
# from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

from scipy.optimize import minimize, rosen, rosen_der


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


def createBoxMarker(name, box, parent, color=(1.0, 1.0, 1.0)):
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
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = 0.3
    return marker


class Instance(object):

    def __init__(self, points, label, voxels=[]):
        self.label = label
        self.points = points
        self.voxels = voxels
        self.center = Clusters.computeCentroid(self.points)
        _, _, self.rf = self.svd()
        self.transform = np.eye(4, dtype=float)
        if len(voxels) > 0:
            self.heavier_label = int(self.getHeavierLabel())
        else:
            self.heavier_label = self.label

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

    def getBoxArea(self):
        self.getRF()
        box = self.getBBox()
        sx = math.fabs(box[0, 0] - box[0, 1])
        sy = math.fabs(box[1, 0] - box[1, 1])
        sz = math.fabs(box[2, 0] - box[2, 1])
        return sx * sy * sz

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

        box_rf = transformations.KDLtoNumpyVector(
            self.getBoxRF()).reshape((1, 7))
        whole = np.concatenate(
            (np.array([self.heavier_label]), box_rf.ravel(), size.ravel()))
        return whole

    def toString(self):
        return ' '.join(map(str, self.shortRepresentation()))

    def updateSVD(self, x):
        x = np.array(x).reshape(4)
        x = x / np.linalg.norm(x)
        frame = PyKDL.Frame()
        frame.M = PyKDL.Rotation.Quaternion(x[0], x[1], x[2], x[3])
        mat = transformations.KLDtoNumpyMatrix(frame)
        self.rf = mat[0:3, 0:3]
        area = self.getBoxArea()
        print("Area:", area)
        return area

    def optimizeSVD(self):
        frame = self.getRF()
        x0 = frame.M.GetQuaternion()
        x0 = np.array(x0)
        minimize(self.updateSVD, x0, method='L-BFGS-B')


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


def createMarkerVolume(name, points=[], color=[1.0, 1.0, 1.0], map_resolution=0.01, base_frame='world'):
    marker = Marker()
    marker.header.frame_id = base_frame
    marker.type = Marker.CUBE_LIST
    marker.id = 0
    marker.action = marker.ADD
    marker.ns = name

    marker.scale.x = map_resolution
    marker.scale.y = map_resolution
    marker.scale.z = map_resolution
    for p in points:
        pp = Point()
        pp.x = p[0]
        pp.y = p[1]
        pp.z = p[2]
        marker.points.append(pp)

        cc = ColorRGBA()
        cc.r = color[0]
        cc.g = color[1]
        cc.b = color[2]
        cc.a = 1.0
        marker.colors.append(cc)
    return marker


node = RosNode("ciao", disable_signals=True)
node.setHz(10)

#⬢⬢⬢⬢⬢➤ Visualization topic
vis_topic = node.createPublisher('/showmap/clusters', MarkerArray)


map_file = node.setupParameter("map_file", "")

instance_folder = node.setupParameter("instance_folder", "")

test = '/media/psf/Home/Desktop/remote_Temp/rgb-dataset-masks-squared-maps/benchs/scene_01_w0.01/instance_000_0.txt'

points = np.loadtxt(test)

volume_marker = createMarkerVolume(
    'volume', points=points, map_resolution=0.005)

instance = Instance(points=points, label=0)

instance_original_frame = instance.getRF()
instance_original_box = createBoxMarker(
    "instance_original_box", instance.getBBox(), "instance")

instance.optimizeSVD()

instance_new_frame = instance.getRF()
instance_new_box = createBoxMarker(
    "instance_new_box", instance.getBBox(), "instance_opt", color=(0.0, 1.0, 0.0))


while node.isActive():

    node.broadcastTransform(instance_original_frame,
                            "instance", "world", node.getCurrentTime())
    node.broadcastTransform(instance_new_frame,
                            "instance_opt", "world", node.getCurrentTime())

    print "Box area:", instance.getBoxArea()

    marker_array = MarkerArray()
    marker_array.markers.append(volume_marker)
    marker_array.markers.append(instance_original_box)
    marker_array.markers.append(instance_new_box)
    for m in marker_array.markers:
        m.header.stamp = node.getCurrentTime()

    vis_topic.publish(marker_array)
    node.tick()
