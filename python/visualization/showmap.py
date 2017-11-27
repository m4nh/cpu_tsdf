#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import PyKDL
import glob
import os
import pprint
import cv2
import sys
import numpy as np
from roars.rosutils.rosnode import RosNode
import roars.vision.colors as colors
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class MultiLabelVoxel(object):
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
        self.pos = np.array(data[0:3])
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
                #     # print self.label_map
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


# for k, v in MultiLabelVoxel.CLASSES_COLOR_MAP.iteritems():
#     print k, colors.getColor(v)

# sys.exit(0)

node = RosNode("ciao", disable_signals=True)

map_file = node.setupParameter("map_file", "")
min_w = node.setupParameter("min_w", 0)


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
        voxels.append(MultiLabelVoxel(data.ravel()))


#⬢⬢⬢⬢⬢➤ Map Marker
marker = Marker()
marker.header.frame_id = "world"
marker.type = Marker.CUBE_LIST
marker.id = 100
marker.action = marker.ADD
marker.ns = "map"
marker_array = MarkerArray()
marker_array.markers.append(marker)
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

print "Max global weight:", max_global_weight
#⬢⬢⬢⬢⬢➤ Marker Fill
for voxel in voxels:

    (l, w) = voxel.heavierLabel()
    is_probable_background = voxel.containsLabel(-1)

    label_color = voxel.getColor()

    if w > min_w:

        p = PyKDL.Vector(voxel.pos[0] - center[0], voxel.pos[1] - center[1], voxel.pos[2] - center[2])
        p = map_transform * p

        point = Point()
        point.x = p.x()
        point.y = p.y()
        point.z = p.z()
        color = ColorRGBA()
        color.r = label_color[0] / 255.0
        color.g = label_color[1] / 255.0
        color.b = label_color[2] / 255.0
        color.a = 1  # float(w) / max_global_weight

        marker.points.append(point)
        marker.colors.append(color)

while node.isActive():

    for m in marker_array.markers:
        m.header.stamp = node.getCurrentTime()
    vis_topic.publish(marker_array)
    node.tick()
