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

import pymesh

node = RosNode("match_boxes", disable_signals=True)

file1 = node.setupParameter("file1", "/tmp/daniele/instances_02.map_0_GT.txt")
file2 = node.setupParameter("file2", "/tmp/daniele/instances_scene_02.map_200.txt")

d1 = np.loadtxt(file1)
d2 = np.loadtxt(file2)


mesh = pymesh.form_mesh([], [])
