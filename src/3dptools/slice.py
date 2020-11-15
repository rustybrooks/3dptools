#!/usr/bin/env python

import argparse
import meshcut

import os
import numpy as np
import mayavi.mlab as mlab
import itertools
import utils
import ply
# import scipy.spatial.distance as spdist
import scipy.spatial


def merge_close_vertices(verts, faces, close_epsilon=1e-5):
    kdtree = scipy.spatial.cKDTree(verts)
    print("made tree")
    nverts = len(verts)
    old2new = np.zeros(nverts, dtype=np.int)
    merged_verts = np.zeros(nverts, dtype=np.bool)
    new_verts = []

    for i in range(nverts):
        if merged_verts[i]:
            continue

        nearest = kdtree.query_ball_point(verts[i], r=close_epsilon)
        # print(nearest)
        merged_verts[nearest] = True
        old2new[nearest] = len(new_verts)
        # for qi in nearest[1]:
        #     merged_verts[qi] = True
        #     old2new[qi] = len(new_verts)
        #
        new_verts.append(verts[i])

    new_verts = np.array(new_verts)

    # Recompute face indices to index in new_verts
    new_faces = np.zeros((len(faces), 3), dtype=np.int)
    for i, f in enumerate(faces):
        new_faces[i] = (old2new[f[0]], old2new[f[1]], old2new[f[2]])

    # again, plot with utils.trimesh3d(new_verts, new_faces)
    print(faces[0:5], new_faces[0:5])
    return new_verts, new_faces


def show(plane, expected_n_contours):
    P = meshcut.cross_section_mesh(mesh, plane)
    colors = [
        (0, 1, 1),
        (1, 0, 1),
        (0, 0, 1)
    ]
    print("num contours : ", len(P), ' expected : ', expected_n_contours)

    if True:
        utils.trimesh3d(mesh.verts, mesh.tris, color=(1, 1, 1),
                        opacity=0.5)
        utils.show_plane(plane.orig, plane.n, scale=1, color=(1, 0, 0),
                         opacity=0.5)

        for p, color in zip(P, itertools.cycle(colors)):
            p = np.array(p)
            mlab.plot3d(p[:, 0], p[:, 1], p[:, 2], tube_radius=None,
                        line_width=3.0, color=color)
    return P


def load_stl(stl_fname):
    import stl
    m = stl.mesh.Mesh.from_file(stl_fname)

    # Flatten our vert array to Nx3 and generate corresponding faces array
    verts = m.vectors.reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)

    verts, faces = merge_close_vertices(verts, faces)
    # verts, faces = meshcut.merge_close_vertices(verts, faces)
    return verts, faces


def load_file(filename):
    v = f = None
    ext = os.path.splitext(filename)[-1].lower()
    if ext in ['.stl']:
        v, f = load_stl(filename)
    elif ext in ['.ply']:
        with open(filename) as f:
            v, f, _ = ply.load_ply(f)

    return v, f


if __name__ == '__main__':
    ##
    example_dir = os.path.join('.')
    # example_fname = os.path.join(example_dir, 'data', 'mesh.ply')
    example_fname = os.path.join(example_dir, 'data', 'YodaForce.stl')
    # example_fname = os.path.join(example_dir, 'data', 'sphere.stl')
#    with open(example_fname) as f:
#        verts, faces, _ = ply.load_ply(f)

    verts, faces = load_file(example_fname)
    # import sys; sys.exit(1)
    print(1)

    mesh = meshcut.TriangleMesh(verts, faces)
    print(2)

    # This will align the plane with some edges, so this is a good test
    # for vertices intersection handling
    plane_orig = (1.28380000591278076172, -0.12510000169277191162, 0)
    plane_norm = (1, 0, 0)

    plane = meshcut.Plane(plane_orig, plane_norm)
    P = show(plane, expected_n_contours=3)
    ##
    # This will align the plane with some edges, so this is a good test
    # for vertices intersection handling
    plane_orig = (0.93760002, -0.12909999, 0)
    plane_norm = (1, 0, 0)

    plane = meshcut.Plane(plane_orig, plane_norm)
    show(plane, expected_n_contours=1)

    ##
    plane_orig = (1, 0, 0)
    plane_norm = (1, 0, 0)

    plane = meshcut.Plane(plane_orig, plane_norm)
    show(plane, expected_n_contours=3)
    ##
    plane_orig = (0.7, 0, 0)
    plane_norm = (0.2, 0.5, 0.3)

    plane = meshcut.Plane(plane_orig, plane_norm)
    show(plane, expected_n_contours=2)
    mlab.show()
