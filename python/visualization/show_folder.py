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
import tools


node = RosNode("ciao", disable_signals=True)
node.setHz(10)

#⬢⬢⬢⬢⬢➤ Visualization topic
vis_topic = node.createPublisher('/showmap/clusters', MarkerArray)


map_file = node.setupParameter("map_file", "")

instance_folder = node.setupParameter("instance_folder", "")


files = glob.glob(os.path.join(instance_folder, "*.txt"))

box_file = ''
box_opt_file = ''
instance_files = []
instances = []
for f in files:
    if 'boxes.txt' in f:
        box_file = f
    elif 'boxes_opt' in f:
        box_opt_file = f
    else:
        instance_files.append(f)
        instances.append(tools.Instance.fromFile(f))


boxes_data = np.loadtxt(box_file)
boxes_opt_data = np.loadtxt(box_opt_file)

boxes = []
boxes_opt = []

for i in range(0, len(boxes_data)):
    box = tools.VBox(boxes_data[i])
    boxes.append(box)

for i in range(0, len(boxes_opt_data)):
    box = tools.VBox(boxes_opt_data[i])
    boxes_opt.append(box)


while node.isActive():

    mar = MarkerArray()

    for i in range(0, len(instances)):
        name = 'instance_{}'.format(i)
        volume = tools.createMarkerVolume(name, instances[i].points)
        mar.markers.append(volume)

    for i in range(0, len(boxes)):
        name = 'box_{}'.format(i)
        node.broadcastTransform(
            boxes[i].frame, name, 'world', node.getCurrentTime())
        mar.markers.append(
            tools.createBoxMarker(name, boxes[i], name, color=(1.0, 0.0, 0.0))
        )

    for i in range(0, len(boxes_opt)):
        name = 'box_opt_{}'.format(i)
        node.broadcastTransform(
            boxes_opt[i].frame, name, 'world', node.getCurrentTime())
        mar.markers.append(
            tools.createBoxMarker(
                name, boxes_opt[i], name, color=(0.0, 1.0, 0.0))
        )

    vis_topic.publish(mar)
    node.tick()
