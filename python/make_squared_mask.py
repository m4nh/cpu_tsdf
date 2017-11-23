import glob
import os
import pprint
import cv2
import sys
import numpy as np

# path = '/media/daniele/data/daniele/datasets/indust_maks/scene1'
path = '/media/daniele/data/daniele/datasets/indust_maks/scene7'

tag = 'mask_6'

images = sorted(glob.glob(os.path.join(path, "*.png")))

images = [x for x in images if tag in x]

for image in images:

    img = cv2.imread(image)
    color = img[np.where((img != np.array([255, 255, 255])).all(axis=2))]
    color = tuple(color[0, :].astype(int))

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # print color
    # sys.exit(0)
    img[np.where((img == [255, 255, 255]).all(axis=2))] = np.array([0, 0, 0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    points = cv2.findNonZero(gray)
    rect = cv2.boundingRect(points)

    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, -1)
    img[np.where((img == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    print rect

    base_name = os.path.basename(image).split(".")[0]
    outpath = os.path.join(path, base_name + "_squared." + os.path.basename(image).split(".")[1])

    cv2.imwrite(outpath, img)
    print outpath
