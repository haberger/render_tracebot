import argparse
import os
import numpy as np
import yaml
import sys
import open3d as o3d
import trimesh
import json

parser = argparse.ArgumentParser()
parser.add_argument('config_path', default="config.yaml", help="Path to config file")
args = parser.parse_args()

dirname = os.path.dirname(__file__) #TODO
#dirname = "/home/v4r/David/BlenderProc"

with open(os.path.join(dirname, args.config_path), "r") as stream:
    config = yaml.safe_load(stream)

mesh_dir = os.path.join(dirname, config["models_dir"])



target_bop_objs = []
for filename in sorted(os.listdir(mesh_dir)):
    if filename[-3:] == 'stl':
        path = os.path.join(mesh_dir, filename)
        print(path)
        #path = "/media/v4r/My Passport/BlenderProc1/bop_dataset/lm/models/obj_000001.ply"
        obj = trimesh.load(path)
        target_bop_objs.append(obj) 
        print(filename)
        #exit
# obj = trimesh.load("/home/v4r/David/models/obj_000008.stl")
# target_bop_objs = [obj]
models_info = dict()

for oi, obj in enumerate(target_bop_objs):
    min_x, min_y, min_z, max_x, max_y, max_z = obj.bounds.reshape(-1) * 1000.0
    size_x, size_y, size_z = obj.extents * 1000.0
    samples = trimesh.sample.sample_surface_even(obj, 15000)[0]
    diameter = 1000.0 * np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=-1).max()
    models_info[str(oi+1)] = {
        'diameter': diameter,
        'min_x': min_x, 'min_y': min_y, 'min_z': min_z,
        'max_x': max_x, 'max_y': max_y, 'max_z': max_z,
        'size_x': size_x, 'size_y': size_y, 'size_z': size_z,
        }

if not os.path.exists(f"models_infoss.json"):
    with open(f"models_infoss.json", 'w') as file:
        json.dump(models_info, file, indent=2)

#print(target_bop_objs)