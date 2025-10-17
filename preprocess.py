from mesh_to_sdf import mesh_to_voxels

import trimesh
import skimage
from icecream import ic
import numpy as np
from trimesh import Trimesh
from typing import Tuple
import os

import os
from pathlib import Path

os.environ['IGR_PATH'] = 'IGR'

import logging
import math
import os
import pickle
import sys
from pathlib import Path

import pyrender
import torch
from matplotlib import pyplot as plt

from icecream import ic
from data_reader import read_scene_zarr


from sdf_physics.physics3d.bodies import SDFBox, SDFCylinder, SDF3D, Mesh3D, SDFGrid3D
from sdf_physics.physics3d.constraints import TotalConstraint3D
from sdf_physics.physics3d.forces import Gravity3D
from sdf_physics.physics3d.utils import get_tensor, Rx, Ry, Recorder3D, Defaults3D, load_igrnet, decode_igr
from sdf_physics.physics3d.world import World3D, run_world

def compute_unit_sphere_transform(mesh: Trimesh) -> Tuple[np.ndarray, float]:
    """
    returns translation and scale, which is applied to meshes before computing their SDF cloud
    """
    # the transformation applied by mesh_to_sdf.scale_to_unit_sphere(mesh)
    translation = - mesh.bounding_box.centroid
    scale = 1 / np.max(np.linalg.norm(mesh.vertices + translation, axis=1))

    return translation, scale

def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    translation = - mesh.bounding_box.centroid
    scale =  2 / np.max(mesh.bounding_box.extents)

    return translation, scale


if __name__ == "__main__":

    dataset_dir = "./google3/"
    zarr_path = "./google3/scene.zarr"  # 修改成你的路径


    data = read_scene_zarr(zarr_path)

    width, height, cam_position, target_look_at, cam_up_direction, near, far = data.get("metadata_camera", (None,)*7)
    rgb, depth, seg_mask, intrinsic, extrinsic_world_to_cam, extrinsic_cam_to_world = data.get("camera", (None,)*6)
    bodies = []
    joints = []
    friction_coeff = 0.15

   
    # mesh = shrink_mesh('obj2_registered.obj', shrink_mode= "by_margin", margin_m=0.04, shrink_center="centroid",)

    mesh_raw = trimesh.load('obj1_registered.obj')
    translation, scale = scale_to_unit_cube(mesh_raw)

    voxels = mesh_to_voxels(mesh_raw, 64, pad=False)

    # mesh.show()
    ## apply translation and scale to mesh

    # mesh.apply_scale(float(1/scale))
    # mesh.apply_translation(-translation)


    body_grid = SDFGrid3D(pos = -np.array([translation[0],  translation[1], translation[2]]) , 
                        scale =  1/scale, 
                        sdf = voxels, 
                        vel=(0, 0, 0), 
                        mass=0.5,
                        thickness= 0.0,)
    ## to cpu
    vertices = body_grid.verts.cpu()
    faces = body_grid.faces.cpu()

    ### create mesh via vertices and faces
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)

    mesh_from_grid = trimesh.Trimesh(vertices=V, faces=F, process=True)
    mesh_from_grid.apply_translation(-np.array([translation[0],  translation[1], translation[2]]))
    out_path = "obj1_processed.obj"
    mesh_from_grid.export(out_path)
    print(f"Exported mesh to: {out_path}")
    ic(translation, scale)
    ic(voxels.shape)
    ic(np.min(voxels), np.max(voxels))

    ## save voxels as npy and save translation and scale
    np.save('voxels_obj1.npy', voxels)
    np.save('translation_obj1.npy', -translation)
    np.save('scale_obj1.npy', 1/scale)
