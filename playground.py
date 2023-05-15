import blenderproc as bproc
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
import shutil

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

def load_bop_dataset_as_distractor(bop_datasets_path, dataset, max_size): #TODO distractor or no distracotr
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
    return distractor_objs   

def load_needle(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    needle = {}
    needle['parts'] = []
    needle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'needle':
            needle['whole'] = obj
        else:
            if name == 'needle_without':
                needle['annos'].append(obj)
                obj.set_cp("category_id", 5)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            elif name == 'needle_cap':
                needle['annos'].append(obj)
                needle['parts'].append(obj)
                obj.set_cp("category_id", 20)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            else:
                needle['parts'].append(obj)
                obj.set_cp("category_id", 5)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
    tracebot_objs["needle"] = needle
    return tracebot_objs

def load_white_clamp(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    white_clamp = {}
    white_clamp['parts'] = []
    white_clamp['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'clamp_w':
            white_clamp['whole'] = obj
            white_clamp['annos'].append(obj)
        else:
            white_clamp['parts'].append(obj)
        obj.set_cp("category_id", 6)
        model_path = os.path.join(config['models_dir'], 'clamp', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["white_clamp"] = white_clamp

    return tracebot_objs    

def load_red_clamp(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    red_clamp = {}
    red_clamp['parts'] = []
    red_clamp['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'clamp_r':
            red_clamp['whole'] = obj
            red_clamp['annos'].append(obj)
        else:
            red_clamp['parts'].append(obj)
        obj.set_cp("category_id", 7)
        model_path = os.path.join(config['models_dir'], 'clamp', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)
    tracebot_objs["red_clamp"] = red_clamp
    return tracebot_objs  

def load_red_cap(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    red_cap = {}
    red_cap['parts'] = []
    red_cap['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'red_cap':
            red_cap['whole'] = obj
            red_cap['annos'].append(obj)
        else:
            red_cap['parts'].append(obj)
        obj.set_cp("category_id", 8)
        model_path = os.path.join(config['models_dir'], 'red_cap', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)
    tracebot_objs["red_cap"] = red_cap
    return tracebot_objs  

def load_yellow_cap(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    yellow_cap = {}
    yellow_cap['parts'] = []
    yellow_cap['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'yellow_cap':
            yellow_cap['whole'] = obj
            yellow_cap['annos'].append(obj)
        else:
            yellow_cap['parts'].append(obj)
        obj.set_cp("category_id", 9)
        model_path = os.path.join(config['models_dir'], 'yellow_cap', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)
    tracebot_objs["yellow_cap"] = yellow_cap
    return tracebot_objs  

def load_canister(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    canister = {}
    canister['parts'] = []
    canister['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'canister':
            canister['whole'] = obj
            canister['annos'].append(obj)
        else:
            canister['parts'].append(obj)
        obj.set_cp("category_id", 10)
        model_path = os.path.join(config['models_dir'], 'canister', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["canister"] = canister

    return tracebot_objs    

def load_small_bottle(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    small_bottle = {}
    small_bottle['parts'] = []
    small_bottle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'small_bottle':
            small_bottle['whole'] = obj
            small_bottle['annos'].append(obj)
        else:
            small_bottle['parts'].append(obj)
        obj.set_cp("category_id", 12)
        model_path = os.path.join(config['models_dir'], 'small_bottle', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["small_bottle"] = small_bottle
    return tracebot_objs  

def load_medium_bottle(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    medium_bottle = {}
    medium_bottle['parts'] = []
    medium_bottle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'medium_bottle':
            medium_bottle['whole'] = obj
            medium_bottle['annos'].append(obj)
        else:
            medium_bottle['parts'].append(obj)
        obj.set_cp("category_id", 13)
        model_path = os.path.join(config['models_dir'], 'medium_bottle', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["medium_bottle"] = medium_bottle
    return tracebot_objs  

def load_large_bottle(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)

    large_bottle = {}
    large_bottle['parts'] = []
    large_bottle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == 'large_bottle':
            large_bottle['whole'] = obj
            large_bottle['annos'].append(obj)
        else:
            large_bottle['parts'].append(obj)
        obj.set_cp("category_id", 14)
        model_path = os.path.join(config['models_dir'], 'large_bottle', obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs["large_bottle"] = large_bottle
    return tracebot_objs 

def render(config):

    bproc.init()

    # mesh_dir = os.path.join(dirname, config["models_dir"])

    dataset_name = config["dataset_name"]

    target_path = os.path.join(config['output_dir'], 'bop_data', dataset_name, 'models')
    if not os.path.isdir(target_path):
        shutil.copytree(config['models_dir'], target_path)

    tracebot = {}
    tracebot = load_needle(tracebot, os.path.join(config["models_dir"], 'needle/needle.blend'))
    tracebot = load_red_clamp(tracebot, os.path.join(config["models_dir"], 'clamp/clamp_red.blend'))
    tracebot = load_white_clamp(tracebot, os.path.join(config["models_dir"], 'clamp/clamp_white.blend'))
    tracebot = load_red_cap(tracebot, os.path.join(config["models_dir"], 'red_cap/red_cap.blend'))
    tracebot = load_yellow_cap(tracebot, os.path.join(config["models_dir"], 'yellow_cap/yellow_cap.blend'))
    tracebot = load_canister(tracebot, os.path.join(config["models_dir"], 'canister/canister.blend'))
    tracebot = load_small_bottle(tracebot, os.path.join(config["models_dir"], 'small_bottle/small_bottle.blend'))
    tracebot = load_large_bottle(tracebot, os.path.join(config["models_dir"], 'large_bottle/large_bottle.blend'))
    tracebot = load_medium_bottle(tracebot, os.path.join(config["models_dir"], 'medium_bottle/medium_bottle.blend'))


    # print(cap.get_location())
    bop_datasets = {}
    print(config["distractions"]["bop_datasets"])
    if config["distractions"]["bop_datasets"] != None:
        for bop_dataset in config["distractions"]["bop_datasets"]:
            dataset = load_bop_dataset_as_distractor(
                config["distractions"]["bop_datasets_path"], 
                bop_dataset, 
                config["distractions"]["max_size"])
            bop_datasets[bop_dataset] = dataset

    # exit()

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
    max_bounces = 10
    bproc.renderer.set_light_bounces(
        glossy_bounces=max_bounces, 
        max_bounces=max_bounces, 
        transmission_bounces=max_bounces, 
        transparent_max_bounces=max_bounces, 
        volume_bounces=max_bounces)

    bproc.camera.set_intrinsics_from_K_matrix(np.reshape(config["cam"]["K"], (3, 3)), 
                                                config["cam"]["width"], 
                                                config["cam"]["height"])
    
    for i in range(config["num_scenes"]):

        # Sample bop objects for a scene
        sampled_target_objs = list(np.random.choice(list(tracebot.keys()), size=len(tracebot.keys()), replace=False))
        sampled_distractor_bop_objs = []
        for bop_dataset in bop_datasets.values():
            dist_per_datatset = min(config["distractions"]["num_distractions"], len(bop_dataset))
            sampled_distractor_bop_objs += list(np.random.choice(bop_dataset, size=dist_per_datatset, replace=False))
        print(sampled_target_objs)
        tracebot_full_body = [tracebot[obj]['whole'] for obj in sampled_target_objs]

        # Randomize materials and set physics
        for obj in (sampled_distractor_bop_objs + tracebot_full_body):
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


        # tracebot_full_body = [tracebot[obj]['whole'] for obj in sampled_target_objs]

        # Sample object poses and check collisions 
        bproc.object.sample_poses(objects_to_sample = sampled_distractor_bop_objs + tracebot_full_body,
                                sample_pose_func = sample_pose_func, 
                                max_tries = 1000)
                
        # Physics Positioning
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                        max_simulation_time=10,
                                        check_object_interval=1,
                                        substeps_per_frame = 20,
                                        solver_iters=25,
                                        origin_mode="CENTER_OF_MASS")
   

        # BVH tree used for camera obstacle checks
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_distractor_bop_objs+tracebot_full_body)


        parts = []
        for obj in sampled_target_objs:
            whole_obj = tracebot[obj]['whole']
            pose_tmat = whole_obj.get_local2world_mat()
            whole_obj.disable_rigidbody()
            whole_obj.hide(True)
            for part in tracebot[obj]['annos']:
                part.set_local2world_mat(pose_tmat)
                part.hide(True) 
            for part in tracebot[obj]['parts']:
                part.set_local2world_mat(pose_tmat)
                part.enable_rigidbody(False, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
                part.hide(False) 
                parts.append(part)


        cam_poses = 0
        
        while cam_poses < config["img_per_scene"]:
            # Sample location
            location = bproc.sampler.shell(center = [0, 0, 0],
                                    radius_min = config["cam"]["radius_min"],
                                    radius_max = config["cam"]["radius_max"],
                                    elevation_min = config["cam"]["elevation_min"],
                                    elevation_max = config["cam"]["elevation_max"])
            # Determine point of interest in scene as the object closest to the mean of a subset of objects
            poi = bproc.object.compute_poi(np.random.choice(tracebot_full_body, size=max(1, len(tracebot_full_body)-1), replace=False)) #
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

        tracebot_anno = []
        for obj in sampled_target_objs:
            tracebot[obj]['annos'] != None
            for o in tracebot[obj]['annos']:
                tracebot_anno.append(o)

        # Write data in bop format
        bproc.writer.write_bop(os.path.join(config["output_dir"], 'bop_data'),
                            target_objects = tracebot_anno,
                            dataset = dataset_name,
                            depth_scale = 0.1,
                            depths = data["depth"],
                            colors = data["colors"], 
                            color_file_format = "JPEG",
                            ignore_dist_thres = 10)

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


    
    

    