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

from sdf_physics.physics3d.bodies import SDFBox, SDFCylinder, SDF3D, Mesh3D, SDFGrid3D
from sdf_physics.physics3d.constraints import TotalConstraint3D
from sdf_physics.physics3d.forces import Gravity3D
from sdf_physics.physics3d.utils import get_tensor, Rx, Ry, Recorder3D, Defaults3D, load_igrnet, decode_igr
from sdf_physics.physics3d.world import World3D, run_world

from icecream import ic
from data_reader import read_scene_zarr
os.environ['PYOPENGL_PLATFORM'] = 'egl'

basedir = Path(__file__).resolve().parent
print(basedir)
shapespace_basedir = basedir.joinpath('shapespaces', 'IGR', 'models')
experiment_basedir = basedir.joinpath('mesh_demo')

TIME = 0.2

# vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()


def callback_vel(world):
    velocity = world.bodies[1].v
    return velocity

def mesh_sdf_demo(scene, output_dir):

    bodies = []
    joints = []
    friction_coeff = 0.15


    mesh = trimesh.load('obj1_registered.obj')
    voxels = np.load('voxels_obj1.npy')
    translation = np.load('translation_obj1.npy')
    scale = np.load('scale_obj1.npy')


    mesh_2 = trimesh.load('obj2_registered.obj')
    voxels_2 = np.load('voxels_obj2.npy')
    translation_2 = np.load('translation_obj2.npy')
    scale_2 = np.load('scale_obj2.npy')

    
    pose_1 =  torch.tensor(np.array([translation[0],  translation[2], translation[1]]), 
                            dtype=Defaults3D.DTYPE, 
                            device=Defaults3D.DEVICE, 
                            requires_grad=True)

    pose_2 = torch.tensor(np.array([translation_2[0],  translation_2[2], translation_2[1]]), 
                            dtype=Defaults3D.DTYPE, 
                            device=Defaults3D.DEVICE, 
                            requires_grad=True)

    optimizer = torch.optim.Adam([pose_1, pose_2], lr=4e-3)
    start_pose_1 = pose_1.clone()
    start_pose_2 = pose_2.clone()
    voxels = np.transpose(voxels, (0, 2, 1)).copy() 
    voxels_2 = np.transpose(voxels_2, (0, 2, 1)).copy() 


    for e in range(5):
        bodies = []
        joints = []
        
        optimizer.zero_grad()

        body_grid = SDFGrid3D(pos = pose_1 , 
                            scale =  scale, 
                            sdf = voxels, 
                            vel=(0, 0, 0), 
                            mass=0.5,
                            thickness= 0.0)
        # body_grid.add_force(Gravity3D())
        joints.append(TotalConstraint3D(body_grid))
        bodies.append(body_grid)


        body_grid_2 = SDFGrid3D(pos =  pose_2 , 
                            scale =  scale_2, 
                            sdf = voxels_2, 
                            vel=(0, 0, 0), 
                            mass=0.5,
                            thickness= 0.0)
        bodies.append(body_grid_2)
        body_grid_2.add_force(Gravity3D())
        # joints.append(TotalConstraint3D(body_grid_2))


        floor = SDFBox([0, -0.10, 0], [1, 0.2, 1], col=(255, 255, 255), fric_coeff=friction_coeff, restitution=0, thickness= 0.0)
        bodies.append(floor)
        joints.append(TotalConstraint3D(floor))



        world =  World3D(bodies, joints, 
                         strict_no_penetration= False, 
                         time_of_contact_diff= True, 
                         stop_contact_grad= False, 
                         stop_friction_grad= False) 
        recorder = Recorder3D(dt=Defaults3D.DT, scene=scene, path=os.path.join(output_dir, f'{e}'), save_to_disk=True)
        vel_traj = run_world(world, fixed_dt= True, scene=scene, run_time=TIME, recorder=recorder, on_step=callback_vel)
        ic(vel_traj)
        # ic(vel_traj[0][-3:])
        ic(len(vel_traj))
        loss = sum((v**2).sum() for v in vel_traj)
        ic(loss)
        loss.backward()
        ic(pose_2.grad)
        optimizer.step()
        ic(pose_1, pose_2)
        ic(start_pose_1, start_pose_2)

if __name__ == "__main__":

    dataset_dir = "./google3/"
    zarr_path = "./google3/scene.zarr"  # 修改成你的路径


    data = read_scene_zarr(zarr_path)

    width, height, cam_position, target_look_at, cam_up_direction, near, far = data.get("metadata_camera", (None,)*7)
    rgb, depth, seg_mask, intrinsic, extrinsic_world_to_cam, extrinsic_cam_to_world = data.get("camera", (None,)*6)

    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    cam = pyrender.PerspectiveCamera(yfov=math.pi / 3, aspectRatio=4 / 3)
    # cam = pyrender.OrthographicCamera(xmag=1, ymag=1, zfar=1500)
    cam_pose = get_tensor([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])
    theta = math.pi / 4
    cam_pose = Ry(theta) @ Rx(-theta) @ cam_pose
    scene.add(cam, pose=cam_pose.cpu())
    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
    light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
    scene.add(light1, pose=(Rx(-theta)).cpu())
    scene.add(light2, pose=(Ry(theta*2) @ Rx(-theta)).cpu())

    mesh_sdf_demo(scene, output_dir="./optim")
