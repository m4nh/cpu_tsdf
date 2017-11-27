import glob
import os
import pprint
import cv2
import sys
import numpy as np
import math

# path = '/media/daniele/data/daniele/datasets/indust_maks/scene1'
path = '/home/daniele/Scrivania/scene_02'
out_path = '/tmp/daniele/temp_images'
label = 5
filter_type = 'l2'
name = 'object_{}_mask_{}'.format(label, filter_type)


tag = 'mask_{}.'.format(label)
tag_depth = 'depth'
debug_screenshot = False

images = sorted(glob.glob(os.path.join(path, "*.png")))

rgbs = [x for x in images if tag in x]
depths = [x for x in images if tag_depth in x]

rgbs = sorted(rgbs)
depths = sorted(depths)


def getPoint(i, j, depth, ratio=10000):

    z = float(depth) / float(ratio)
    x = (z / 570.3) * (float(j) - 320.0)
    y = (z / 570.3) * (float(i) - 240.0)
    return np.array([x, y, z])


def distanceFunction(p1, p2):
    if filter_type == 'l1':
        return np.linalg.norm(p1 - p2, ord=1)
    if filter_type == 'l2':
        return np.linalg.norm(p1 - p2, ord=2)
    if filter_type == 'off':
        return 0  # np.linalg.norm(p1 - p2, ord=2)


for index in range(0, len(rgbs)):

    print rgbs[index], depths[index]

    img = cv2.imread(rgbs[index])
    colored = img.copy()
    depth = cv2.imread(depths[index], cv2.IMREAD_ANYDEPTH)

    color = img[np.where((img != np.array([255, 255, 255])).all(axis=2))]
    color = tuple(color[0, :].astype(int))

    # # cv2.imshow("img", img)
    # # cv2.waitKey(0)
    # # print color
    # # sys.exit(0)

    img[np.where((img == [255, 255, 255]).all(axis=2))] = np.array([0, 0, 0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero(gray)
    rect = cv2.boundingRect(points)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask_colored = np.zeros(img.shape, dtype=np.uint8)
    mask2 = np.ones(img.shape, dtype=float)
    mask3 = np.ones(img.shape, dtype=float)

    cv2.rectangle(mask, (rect[0], rect[1]),
                  (rect[0] + rect[2], rect[1] + rect[3]), 255, -1)
    cv2.rectangle(mask_colored, (rect[0], rect[1]),
                  (rect[0] + rect[2], rect[1] + rect[3]), color, -1)

    left = rect[0]
    right = left + rect[2]
    top = rect[1]
    down = top + rect[3]
    center_j = left + rect[2] * 0.5
    center_i = top + rect[3] * 0.5

    depth_masked = cv2.bitwise_and(depth, depth, mask=mask)
    print "center", center_i, center_j, depth_masked[center_i, center_j]

    middle_val = depth_masked[center_i, center_j]
    middle_point = getPoint(center_i, center_j, middle_val)

    for i in range(top, down + 1):
        for j in range(left, right + 1):
            current_point = getPoint(i, j, depth_masked[i, j])

            dist = distanceFunction(middle_point, current_point)
            dist_max = distanceFunction(
                np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.0, 0.0]))

            if dist_max == 0:
                dist_max = 100000000000000
            # print dist, dist_max
            dist_ratio = dist / dist_max
            val = 1.0 - dist_ratio
            if val <= 0.0:
                val = 0.0

            mask2[i, j] = np.array([val, val, val])
            mask3[i, j] = 1.0 - np.array([val, val, val])

    minD, maxD, _, _ = cv2.minMaxLoc(depth_masked)
    print minD, maxD

    # mask_colored[np.where((mask_colored == [0, 0, 0]).all(
    #     axis=2))] = np.array([255, 255, 255])

    mask_colored = np.multiply(
        mask_colored.astype(float), mask2).astype(np.uint8) + np.multiply(
        np.ones(img.shape).astype(float) * 255, mask3).astype(np.uint8)

    alpha = np.array(img.shape)
    r_channel, g_channel, b_channel = cv2.split(mask_colored)

    a_channel = (mask2[:, :, 0] * 255).astype(np.uint8)
    img_RGBA = cv2.merge((r_channel, g_channel, b_channel, a_channel))

    if debug_screenshot:
        try:
            os.mkdir(os.path.join(out_path, name))
        except:
            pass
        cv2.imwrite(os.path.join(out_path, name, "mask_alpha.png"), img_RGBA)
        cv2.imwrite(os.path.join(out_path, name, "mask.png"), mask)
        cv2.imwrite(os.path.join(out_path, name, "mask2.png"), mask2)
        cv2.imwrite(os.path.join(out_path, name, "rgb.png"), img)
        cv2.imwrite(os.path.join(out_path, name, "depth.png"), depth)
        cv2.imwrite(os.path.join(out_path, name,
                                 "depth_masked.png"), depth_masked)
        cv2.imwrite(os.path.join(out_path, name,
                                 "final_mask.png"), mask_colored)

        cv2.imshow("rgb", img)
        cv2.imshow("mask", mask)
        cv2.imshow("mask2", mask2)
        cv2.imshow("depth_masked", depth_masked)
        cv2.imshow("depth", depth)
        cv2.imshow("final_mask", mask_colored)
        cv2.imshow("colored", colored)
        cv2.waitKey(0)
        sys.exit(0)
    else:

        # cv2.rectangle(img, (rect[0], rect[1]),
        #               (rect[0] + rect[2], rect[1] + rect[3]), color, -1)
        # img[np.where((img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
        # print rect

        base_name = os.path.basename(rgbs[index]).split(".")[0]
        outpath = os.path.join(
            path, base_name + "_filtered_{}.".format(filter_type) + os.path.basename(rgbs[index]).split(".")[1])

        print outpath
        cv2.imwrite(outpath, mask_colored)
    # print outpath
