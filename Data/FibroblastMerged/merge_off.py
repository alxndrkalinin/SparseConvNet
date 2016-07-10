#!/usr/bin/env python
# coding: utf-8
# @author: Alexandr Kalinin <akalinin@umich.edu>

import argparse
import numpy as np

def read_off(file):
    if 'OFF' != file.readline().strip():
        print 'Not a valid OFF header'
        return
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = []
    for i_vert in range(n_verts):
        verts.append([float(s) for s in file.readline().strip().split(' ')])
    faces = []
    for i_face in range(n_faces):
        faces.append([int(s) for s in file.readline().strip().split(' ')][1:])
    return verts, faces


def write_off(filename, verts, faces):
    with open(filename, 'w') as out:
        out.write('OFF\n')
        out.write('%s %s %s\n' % (len(verts), len(faces), 0))
        for vert in verts:
            out.write('%s %s %s\n' % tuple(vert))
        for face in faces:
            out.write('3 %s %s %s\n' % tuple(face))
    return out.close()


def merge_off(file1, file2, output):
    with open(file1, 'r') as off1, open(file2, 'r') as off2:

        print('Merging %s mesh and %s mesh...' % (file1, file2))

        off1_v, off1_f = read_off(off1)
        off2_v, off2_f = read_off(off2)

        off_v = np.vstack((off1_v, off2_v))
        off2_f_renum = [[x + len(off1_v) for x in off2_f[i]] for i in xrange(len(off2_v))]
        off_f = np.vstack((off1_f, off2_f_renum))

        print('... into %s' % output)

        write_off(output, off_v, off_f)
        print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merging 2 meshes in .OFF format into one')
    parser.add_argument('off1', metavar='I1', nargs=1, help='first .OFF file')
    parser.add_argument('off2', metavar='I2', nargs=1, help='seconf .OFF file')
    parser.add_argument('out_off', metavar='O', nargs=1, help='output .OFF file')
    args = parser.parse_args()
    merge_off(args.off1[0], args.off2[0], args.out_off[0])
