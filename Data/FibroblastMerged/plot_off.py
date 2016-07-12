#!/usr/bin/env python
# coding: utf-8
# @author: Alexandr Kalinin <akalinin@umich.edu>

import argparse
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import numpy as np
from stl import mesh
from merge_off import read_off

def read_off2mesh(off_file):
	with open(off_file, 'r') as off:
	    off_v, off_f = read_off(off)
	off_v = np.array(off_v)
	off_f = np.array(off_f)
	off_mesh = mesh.Mesh(np.zeros(off_f.shape[0], dtype=mesh.Mesh.dtype))
	for i, f in enumerate(off_f):
	    for j in range(3):
	        off_mesh.vectors[i][j] = off_v[f[j],:]

	# calculate scaling
	max_dev = max(off_v.max(axis=0) - off_v.min(axis=0)) / 2
	scale = np.array([off_v.mean(axis=0) - max_dev, off_v.mean(axis=0) + max_dev]).transpose()
	return off_mesh, scale


def plot_mesh(off_file):
	off_mesh, scale = read_off2mesh(off_file)
	ax = mplot3d.Axes3D(plt.figure())
	collection = mplot3d.art3d.Poly3DCollection(off_mesh.vectors, linewidths=1, alpha=0.2)
	face_color = [0.5, 0.5, 1]
	collection.set_facecolor(face_color)
	ax.add_collection3d(collection)
	ax.auto_scale_xyz(scale[0], scale[1], scale[2])
	plt.show()
	# pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactive plot of 3D mesh from .OFF file')
    parser.add_argument('off', metavar='I1', nargs=1, help='.OFF file')
    args = parser.parse_args()
    plot_mesh(args.off[0])