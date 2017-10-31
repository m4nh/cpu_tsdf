import PyKDL
import numpy
import numpy as np
import os


def KDLFromArray(chunks, fmt='XYZQ'):
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
    return frame


source_file = "/home/daniele/Desktop/datasets/roars_2017/indust/indust_scene_7_dome/tf#_comau_smart_six_link6.txt"
output_folder = "output"

try:
    os.mkdir(output_folder)
except:
    pass

camera_extrinsics = KDLFromArray([0.09131591676464931, 0.023330268359173824, -0.19437327402734972, -
                                  0.7408449656427065, 0.7505081899194715, 0.01462135745716728, -0.01655531561119271])

poses = np.loadtxt(source_file)
frames = []
for p in poses:
    cp = KDLFromArray(p) * camera_extrinsics
    frames.append(cp)


counter = 0
for f in frames:
    counter_name = str(counter).zfill(5)
    print counter_name

    mat = np.eye(4)

    for i in range(0, 3):
        for j in range(0, 3):
            mat[i, j] = f.M[i, j]

    mat[0, 3] = f.p.x()
    mat[1, 3] = f.p.y()
    mat[2, 3] = f.p.z()

    out_file = os.path.join(output_folder, counter_name + ".txt")
    np.savetxt(out_file, mat)

    counter += 1
