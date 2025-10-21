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

from kal_sdf import kal_mesh_to_voxel

basedir = Path(__file__).resolve().parent
print(basedir)
shapespace_basedir = basedir.joinpath('shapespaces', 'IGR', 'models')
experiment_basedir = basedir.joinpath('mesh_demo')

TIME = 0.3
# vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
# mesh.show()


def callback_vel(world):
    v_list = []
    for b in world.bodies:
        if not isinstance(b, SDFGrid3D):
            # 只关心体素网格（你的两个物体）
            continue
            # 有些“固定”的物体速度可能恒为 0；留着也无妨，或在此再过滤
        v_list.append(b.v.reshape(-1))
    return torch.cat(v_list, dim=0)

def make_diff_world(num_bodies,  sdf_bodies, bodies_pos, bodies_scales, bodies_com):

    bodies = []
    joints = []
    friction_coeff = 0.15

    for idx in range(num_bodies):
        # Create SDF grid body
        body = SDFGrid3D(
            pos=bodies_pos[idx],
            scale=bodies_scales[idx],
            sdf=sdf_bodies[idx],
            vel=(0, 0, 0),
            COM=bodies_com[idx],
            mass=0.5,
            thickness=0.0,   
        )

        body.add_force(Gravity3D())
        bodies.append(body)

            ## to cpu
        vertices = body.verts.clone().detach().cpu()
        faces = body.faces.clone().detach().cpu()

        ### create mesh via vertices and faces
        V = np.asarray(vertices, dtype=np.float64)
        F = np.asarray(faces, dtype=np.int64)

        mesh_from_grid = trimesh.Trimesh(vertices=V, faces=F, process=True)
        mesh_from_grid.apply_translation( np.array(bodies_pos[idx].cpu().detach().numpy() ) )
        out_path = f"obj{idx}_pocessed.obj"
        mesh_from_grid.export(out_path)
        print(f"Exported mesh to: {out_path}")
    floor = SDFBox([0, -0.10, 0], [1, 0.2, 1], col=(255, 255, 255), fric_coeff=friction_coeff, restitution=0, thickness= 0.0)
    bodies.append(floor)
    joints.append(TotalConstraint3D(floor))


    world = World3D(bodies, 
                    joints, 
                    strict_no_penetration= True, 
                    time_of_contact_diff= True, 
                    stop_contact_grad= False, 
                    stop_friction_grad= False) 
    

    return world

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

    # mesh_sdf_demo(scene, output_dir="./optim")

    ## 
    mesh_dataset_path = "/home/iadc/diffsdfsim/google4/recon_mesh"

    objects_path = []
    for i in range(1,20):
        if not os.path.exists(f"{mesh_dataset_path}/obj{i}/obj{i}_registered.obj"):
            break
        print(f"Loading obj{i}_registered.glb from: {mesh_dataset_path}/obj{i}/obj{i}_registered.obj")
        transformed_reconstructed_mesh_path = f"{mesh_dataset_path}/obj{i}/obj{i}_registered.obj"
        objects_path.append(transformed_reconstructed_mesh_path)

    # objects_path.pop(3)
    ic(objects_path)

    sdf_bodies = []
    bodies_pos = []
    bodies_scales = []
    bodies_com = []
    for mesh_path in objects_path:
        sdf_field, translation, scale, unit_mesh = kal_mesh_to_voxel(mesh_path= mesh_path, voxel_resolution= 64)
        print(f"[mesh to sdf]: Done mesh to SDF by kaolin to mesh {mesh_path}")
        # ic(sdf_field.shape)
        voxels = sdf_field.squeeze(0).squeeze(0).permute(0, 2, 1).detach().to(dtype=Defaults3D.DTYPE, device=Defaults3D.DEVICE)
        voxels = torch.flip(voxels, dims=[0])

        scale = 1/scale

        # ic(voxels)
        pose = torch.tensor( np.array([translation[0], -1 *  translation[2],-1 *  translation[1]]), 
                            dtype=Defaults3D.DTYPE, 
                            device=Defaults3D.DEVICE, 
                            requires_grad=False)
        
        com = torch.tensor( np.array([0.0, 0.0, 0.0]), 
                            dtype=Defaults3D.DTYPE, 
                            device=Defaults3D.DEVICE,
                            requires_grad=True)
        
        sdf_bodies.append(voxels)
        bodies_pos.append(pose)
        bodies_scales.append(scale)
        bodies_com.append(com)
    
    start_poses = [p.clone() for p in bodies_pos]
    start_coms = [c.clone() for c in bodies_com]
    # optimizer = torch.optim.Adam(bodies_pos, lr=4e-3)

    COM_optimizer = torch.optim.Adam(bodies_com, lr=1e-2)

    for e in range(20):
        # optimizer.zero_grad()
        COM_optimizer.zero_grad()
        output_dir = "world"
        world  = make_diff_world(num_bodies = len(sdf_bodies), sdf_bodies = sdf_bodies, bodies_pos = bodies_pos, bodies_scales = bodies_scales, bodies_com = bodies_com)

        recorder = Recorder3D(dt=Defaults3D.DT, scene=scene, path=os.path.join(output_dir, f'{e}'), save_to_disk=True)

        vel_traj = run_world(world, fixed_dt= True, scene=scene, run_time=TIME, recorder=recorder, on_step= callback_vel)
        ic(vel_traj)
        # ic(vel_traj[0][-3:])
        ic(len(vel_traj))
        loss = sum((v ** 2).sum() for v in vel_traj)
        ic(loss)
        loss.backward()
        # optimizer.step()
        COM_optimizer.step()
        ic("start poses:", start_poses)
        ic("optimized poses", bodies_pos)
        ic("start COMs:", start_coms)
        ic("optimized COMs", bodies_com)