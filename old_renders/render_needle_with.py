﻿import blenderproc as bproc
# from curses.panel import bottom_panel
# from logging import raiseExceptions
# from sre_parse import CATEGORIES
# from unicodedata import category
import argparse
import os
import numpy as np
import yaml
import sys
import imageio
import random
# from multiprocessing import Process, Queue

# from v4r.bproc_to_bop import complete_dataset_to_bop

# import debugpy

# debugpy.listen(5678)

# debugpy.wait_for_client()

# TODO fix bump and physics simulTION CENTER OF MASS
# TODO ALPHA
# TODO if bop Dataset dosesn nopt have info generate info


def get_bounding_box_diameter(obj):
    bound_box = obj.get_bound_box().real
    p1 = bound_box[0]
    p2 = bound_box[6]
    bounding_box_diameter = np.linalg.norm(p2-p1)
    return bounding_box_diameter

def set_material_properties(obj, cc_textures):
        random_cc_texture = np.random.choice(cc_textures)
        obj.replace_materials(random_cc_texture)
        mat = obj.get_materials()[0]      
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Alpha", 1.0)
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        if not obj.has_uv_mapping():
            obj.add_uv_mapping("smart")
        obj.hide(False)
        return obj

def load_bop_dataset_as_distractor(bop_datasets_path, dataset, max_size, category_ids, id_mapping): #TODO distractor or no distracotr
    id = max(231, max(category_ids)+1)
    category_ids.append(id)
    id_mapping[f"{dataset}_distractors"] = id

    if dataset in ["ycbv", "lm", "tyol", "hb", "icbin", "itodd", "tud1"]:
        bop_dataset_path = os.path.join(bop_datasets_path, dataset)
        bop_dataset = bproc.loader.load_bop_objs(
            bop_dataset_path = bop_dataset_path, 
            mm2m = True)
    elif dataset in ["tless"]:
        bop_dataset_path = os.path.join(bop_datasets_path, dataset)
        bop_dataset = bproc.loader.load_bop_objs(
            bop_dataset_path = bop_dataset_path, 
            model_type = 'cad', 
            mm2m = True)
    else:
        raise Exception(f"BOP Dataset \"{dataset}\" not supported")
    distractor_objs = []
    for bop_obj in bop_dataset:
        bop_obj.set_shading_mode('auto')
        bop_obj.hide(True)
        if max_size:
            if get_bounding_box_diameter(bop_obj) <= max_size:
                distractor_objs.append(bop_obj)
        bop_obj.set_cp("category_id", id)
    return distractor_objs   


def render(config):

    bproc.init()

    mesh_dir = os.path.join(dirname, config["models_dir"])

    category_ids = []
    id_mapping = {}

    target_objs = []
    distractor_objs = []
    
    ns1 = [31, 81]
    ns2 = [32, 82]
    ns3 = [33, 83]
    ns4 = [34, 84]
    ns0 = [3, 8]
    ns5 = [9, 10]
    ns0_list = []
    ns1_list = []
    ns2_list = []
    ns3_list = []
    ns4_list = []
    ns5_list = []

    cap = [4]
    with_cap = [81,82,83,84,8,9]
    without_cap = [31,32,33,34,3,10]

    dataset_name = config["dataset_name"]

    sets_path = os.path.join(config["output_dir"], 'bop_data', dataset_name, "train_pbr")
    if os.path.isdir(sets_path):
        set_path = sorted(os.listdir(sets_path))[-1]

        segmap_path = os.path.join(sets_path, set_path, "segmap")

        saved_maps = sorted(os.listdir(segmap_path))
        if len(saved_maps) != 0 and int(saved_maps[-1][:-4]) != 999:
            segmap_idx = int(saved_maps[-1][:-4])+1
            print(saved_maps)
        else:
            segmap_idx = 0
    else:
        segmap_idx = 0

    for filename in os.listdir(mesh_dir): #TODO mm2m stl/ply
        if filename[-3:] != "stl":
            continue
        obj = bproc.loader.load_obj(os.path.join(mesh_dir, filename))[0]
        id = int(filename[4:-4])
        if id >= 255:
            raise Exception("filename needs to be in format obj_xxxxxx.stl, where xxxxxx is 0 padded number smaller than 230")
        if id not in category_ids:
            obj.hide(True) 
            category_ids.append(id)
            obj.set_cp("category_id", id)
            id_mapping[filename] = id
            obj.set_shading_mode('auto')
            if id == 4:
                ns0_list.append(obj)
                ns1_list.append(obj)
                ns2_list.append(obj)
                ns3_list.append(obj)
                ns4_list.append(obj)
                ns5_list.append(obj)
            if id in ns1:
                ns1_list.append(obj)
            elif id in ns2:
                ns2_list.append(obj)
            elif id in ns3:
                ns3_list.append(obj)
            elif id in ns4:
                ns4_list.append(obj)
            elif id in ns5:
                ns5_list.append(obj)
            elif id in ns0:
                ns0_list.append(obj)
            else:
                distractor_objs.append(obj)
        else:
            raise Exception("filename needs to be in format obj_xxxxxx.stl, where xxxxxx is 0 padded number smaller than 255")

    needle_lists = [ns0_list, ns1_list, ns2_list, ns3_list, ns4_list, ns5_list]

    #load bop Datatsets
    bop_datasets = {}
    for bop_dataset in config["distractions"]["bop_datasets"]:
        dataset = load_bop_dataset_as_distractor(
            config["distractions"]["bop_datasets_path"], 
            bop_dataset, 
            config["distractions"]["max_size"], 
            category_ids, 
            id_mapping)
        bop_datasets[bop_dataset] = dataset

    # create room
    room_size = max(config["cam"]["radius_max"] * 1.1, 2)
    room_planes = [bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1]),
                bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[0, -room_size, room_size], rotation=[-1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[0, room_size, room_size], rotation=[1.570796, 0, 0]),
                bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[room_size, 0, room_size], rotation=[0, -1.570796, 0]),
                bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1], location=[-room_size, 0, room_size], rotation=[0, 1.570796, 0])]
    for plane in room_planes:
        plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[room_size * 1.5, room_size * 1.5, 1], location=[0, 0, room_size*5])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(200)

    # load cc_textures
    cc_textures = bproc.loader.load_ccmaterials(config["texture_dir"])

    # Define a function that samples 6-DoF poses
    def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
        max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())
        
    # activate depth rendering without antialiasing and set amount of samples for color rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

    bproc.camera.set_intrinsics_from_K_matrix(np.reshape(config["cam"]["K"], (3, 3)), 
                                                config["cam"]["width"], 
                                                config["cam"]["height"])
    
    for i in range(config["num_scenes"]):
        n = random.randint(0, 5)
        target_needle_parts = needle_lists[n]#.copy()

        for part in target_needle_parts:
            if part.get_cp("category_id") in with_cap:
                target_needle = part


        # Sample bop objects for a scene
        #sampled_target_objs = list(np.random.choice(target_objs, size=1, replace=False))
        sampled_distractor_bop_objs = []
        sampled_distractor_bop_objs += list(np.random.choice(distractor_objs, size=3, replace=False))
        for bop_dataset in bop_datasets.values():
            dist_per_datatset = min(config["distractions"]["num_distractions"], len(bop_dataset))
            sampled_distractor_bop_objs += list(np.random.choice(bop_dataset, size=dist_per_datatset, replace=False))

        # Randomize materials and set physics
        for obj in (sampled_distractor_bop_objs+[target_needle]):
            obj = set_material_properties(obj, cc_textures)
        
        # Sample two light sources
        light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
        light_plane.replace_materials(light_plane_material)
        light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
        location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                                elevation_min = 5, elevation_max = 89)
        light_point.set_location(location)

        # sample CC Texture and assign to room planes
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)
            mat = plane.get_materials()[0]      
            mat.set_principled_shader_value("Alpha", 1.0)


        # Sample object poses and check collisions 
        bproc.object.sample_poses(objects_to_sample = sampled_distractor_bop_objs + [target_needle],
                                sample_pose_func = sample_pose_func, 
                                max_tries = 1000)
                
        # Physics Positioning
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                        max_simulation_time=10,
                                        check_object_interval=1,
                                        substeps_per_frame = 20,
                                        solver_iters=25,
                                        origin_mode="CENTER_OF_MASS")


        pose_tmat = target_needle.get_local2world_mat()
        target_needle.disable_rigidbody()
        target_needle.hide(True)

        parts = []
        for part in target_needle_parts:
            if part.get_cp("category_id") not in with_cap:
                part.set_local2world_mat(pose_tmat)
                part = set_material_properties(part, cc_textures)   
                parts.append(part)        

        # BVH tree used for camera obstacle checks
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_distractor_bop_objs+parts)

        cam_poses = 0
        
        while cam_poses < config["img_per_scene"]:
            # Sample location
            location = bproc.sampler.shell(center = [0, 0, 0],
                                    radius_min = config["cam"]["radius_min"],
                                    radius_max = config["cam"]["radius_max"],
                                    elevation_min = config["cam"]["elevation_min"],
                                    elevation_max = config["cam"]["elevation_max"])
            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            poi = bproc.object.compute_poi(np.concatenate((np.random.choice(sampled_distractor_bop_objs, size=max(1, len(sampled_distractor_bop_objs)-1), replace=False), np.array(parts))))
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-np.pi/2.0, np.pi/2.0))
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            
            # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
                # Persist camera pose
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                cam_poses += 1
        # render the whole pipeline
        data = bproc.renderer.render()
        data.update(bproc.renderer.render_segmap(map_by=["class"]))
        
        # Write data in bop format
        
        bproc.writer.write_bop(os.path.join(config["output_dir"], 'bop_data'),
                            target_objects = parts,
                            dataset = dataset_name,
                            depth_scale = 0.1,
                            depths = data["depth"],
                            colors = data["colors"], 
                            color_file_format = "JPEG",
                            ignore_dist_thres = 10)


        sets_path = os.path.join(config["output_dir"], 'bop_data', dataset_name, "train_pbr")
        
        
        set_path = sorted(os.listdir(sets_path))[-1]

        segmap_path = os.path.join(sets_path, set_path, "segmap")

        if not os.path.exists(segmap_path):
            os.makedirs(segmap_path)
        for segmap in data["class_segmaps"]:
            imageio.imwrite(os.path.join(segmap_path, f"{segmap_idx:06}.png"),np.array(segmap))
            segmap_idx += 1 
        for obj in (parts + sampled_distractor_bop_objs):      
            obj.disable_rigidbody()
            obj.hide(True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__) #TODO
    #dirname = "/home/v4r/David/BlenderProc"

    #read config
    with open(os.path.join(dirname, args.config_path), "r") as stream:
        config = yaml.safe_load(stream)
    render(config)
    # q = Queue()
    # r = Process(target=render,  args =(config, q))  
    # post_processing = Process(target=complete_dataset_to_bop, args=(None, q))

    
    

    