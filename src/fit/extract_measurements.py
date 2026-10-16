import numpy as np
import sys
import os
from collections import OrderedDict

import utils

DATA_DIR = "data"


def convert_cp():
    """Read control points (CP) from data/customBodyPoints.txt."""
    cp = []
    tmplist = []
    with open(os.path.join(DATA_DIR, 'customBodyPoints.txt'), "r") as f:
        for line in f:
            if '#' in line:
                if len(tmplist) != 0:
                    cp.append(tmplist)
                    tmplist = []
            elif len(line.split()) == 1:
                continue
            else:
                tmplist.append(list(map(float, line.strip().split())))
    cp.append(tmplist)
    return cp


def calc_measure(cp, vertex, height):
    """Calculate measurements from the SMPL vertices using the control points."""
    measure_list = []

    for measure in cp:
        length = 0.0
        p2 = vertex[int(measure[0][1]), :]

        for i in range(0, len(measure)):
            p1 = p2
            if measure[i][0] == 1:
                p2 = vertex[int(measure[i][1]), :]
            elif measure[i][0] == 2:
                p2 = vertex[int(measure[i][1]), :] * measure[i][3] + \
                    vertex[int(measure[i][2]), :] * measure[i][4]
            else:
                p2 = vertex[int(measure[i][1]), :] * measure[i][4] + \
                    vertex[int(measure[i][2]), :] * measure[i][5] + \
                    vertex[int(measure[i][3]), :] * measure[i][6]
            length += np.sqrt(np.sum((p1 - p2) ** 2.0))

        measure_list.append(length * 100)

    # Scale everything so that measurement 0 (body height) equals the known height.
    measure_list = np.array(measure_list, dtype=np.float64)
    measure_list = float(height) * (measure_list / measure_list[0])

    # Empirical corrections from the original code
    measure_list[8] = measure_list[8] * 0.36
    measure_list[3] = measure_list[3] * 0.6927
    return measure_list.reshape(utils.M_NUM, 1)


def compute_measurements(height, vertices):
    """Return an ordered {name: value} dict. No printing, no files written."""
    cp = convert_cp()
    measure = calc_measure(cp, vertices, height)
    result = OrderedDict()
    for i in range(0, utils.M_NUM):
        result[str(utils.M_STR[i])] = float(measure[i, 0])
    return result


def save_obj(vertices, obj_mesh_name='test.obj',
             face_path='./src/tf_smpl/smpl_faces.npy'):
    faces = np.load(face_path)
    with open(obj_mesh_name, 'w') as fp:
        for v in vertices:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:  # Faces are 1-based in obj files
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))
    print("Model Saved...")


def extract_measurements(height, vertices):
    """Original entry point: print all measurements and save test.obj."""
    result = compute_measurements(height, vertices)
    for name, value in result.items():
        print("%s: %f" % (name, value))
    save_obj(vertices)
    return result