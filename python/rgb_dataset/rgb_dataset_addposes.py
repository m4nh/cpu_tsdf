import os
import glob
import shutil
import numpy as np
import numpy
import PyKDL


def KLDtoNumpyMatrix(frame):
    M = frame.M
    R = numpy.array([
        [M[0, 0], M[0, 1], M[0, 2]],
        [M[1, 0], M[1, 1], M[1, 2]],
        [M[2, 0], M[2, 1], M[2, 2]],
    ])
    P = numpy.transpose(
        numpy.array([
            frame.p.x(),
            frame.p.y(),
            frame.p.z()
        ])
    )
    P = P.reshape(3, 1)
    T = numpy.concatenate([R, P], 1)
    T = numpy.concatenate([T, numpy.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    return T


def KDLFromArray(chunks, fmt='QXYZ'):
    if fmt == 'RPY':
        frame = PyKDL.Frame()
        frame.p = PyKDL.Vector(
            chunks[0], chunks[1], chunks[2]
        )
        frame.M = PyKDL.Rotation.RPY(
            chunks[3],
            chunks[4],
            chunks[5]
        )
    if fmt == 'XYZQ':
        frame = PyKDL.Frame()
        frame.p = PyKDL.Vector(
            chunks[0], chunks[1], chunks[2]
        )
        q = numpy.array([chunks[3],
                         chunks[4],
                         chunks[5],
                         chunks[6]])
        q = q / numpy.linalg.norm(q)
        frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
    if fmt == 'QXYZ':
        frame = PyKDL.Frame()
        frame.p = PyKDL.Vector(
            chunks[4], chunks[5], chunks[6]
        )
        q = numpy.array([chunks[1],
                         chunks[2],
                         chunks[3],
                         chunks[0]])
        q = q / numpy.linalg.norm(q)
        frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
    return frame


id = '15'
scene_path = '/home/daniele/Downloads/rgbd-scenes-v2/imgs/scene_{}'.format(id)
pose_file = '/home/daniele/Downloads/rgbd-scenes-v2/pc/{}.pose'.format(id)

zeros = 5


if os.path.exists(pose_file):
    ff = open(pose_file, 'r')
    lines = ff.readlines()

    counter = 0
    for line in lines:
        print "########"
        chunks = line.split()
        pose = np.array(map(float, chunks))
        raw_pose = KLDtoNumpyMatrix(KDLFromArray(pose))
        pose_name = str(counter).zfill(zeros) + "-pose.txt"

        counter += 1

        # pose_name = sub_scene + "_{}_pose.txt".format(counter)
        pose_path = os.path.join(scene_path, pose_name)
        print pose, raw_pose, pose_path
        np.savetxt(pose_path, raw_pose)
        # counter += 1
        # print pose_name
