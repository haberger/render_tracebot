import blenderproc as bproc
import argparse
import os
import numpy as np
import yaml
import sys
import imageio
import random
import shutil

# TODO fix bump and physics simulation CENTER OF MASS currently using original Blenderproc

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

def load_bop_dataset_as_distractor(bop_datasets_path, dataset, max_size): #TODO distractor or no distractor
    if dataset in ["ycbv", "lm", "tyol", "hb", "icbin", "itodd", "tud1"]:
        bop_dataset_path = os.path.join(bop_datasets_path, dataset)
        bop_dataset = bproc.loader.load_bop_objs(
            bop_dataset_path = bop_dataset_path, 
            object_model_unit= 'mm')
    elif dataset in ["tless"]:
        bop_dataset_path = os.path.join(bop_datasets_path, dataset)
        bop_dataset = bproc.loader.load_bop_objs(
            bop_dataset_path = bop_dataset_path, 
            model_type = 'cad', 
            object_model_unit= 'mm')
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


def load_needle_from_blend(tracebot_objs, path):
    objs = bproc.loader.load_blend(path)
    needle = {}
    needle['parts'] = []
    needle['annos'] = []
    for obj in objs:
        name = obj.get_name()
        if name[0:5] == 'Empty':
            continue
        elif name == 'needle':
            needle['whole'] = obj
            obj.hide(True)
            obj.disable_rigidbody()
        else:
            obj.hide(True)
            obj.disable_rigidbody()
            if name == 'needle_without':
                needle['annos'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            elif name == 'needle_cap':
                needle['annos'].append(obj)
                needle['parts'].append(obj)
                obj.set_cp("category_id", 4)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            else:
                needle['parts'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], 'needle', obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
        obj.hide(True)
        obj.disable_rigidbody()
    tracebot_objs["needle"] = needle
    return tracebot_objs

def load_needle_variation_from_blend(tracebot_objs, path, needle_name):
    objs = bproc.loader.load_blend(path)

    needle_vu = {}
    needle_vu['parts'] = []
    needle_vu['annos'] = []
    for obj in objs:
        name = obj.get_name()
        
        if name[0:5] == 'Empty':
            continue
        elif name == needle_name:
            obj.hide(True)
            obj.disable_rigidbody()
            needle_vu['whole'] = obj
        else:
            obj.hide(True)
            obj.disable_rigidbody()
            if name == f'needle_without_{needle_name[-2:]}':
                needle_vu['annos'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], needle_name, obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            elif name == f'needle_cap_{needle_name[-2:]}':
                needle_vu['annos'].append(obj)
                needle_vu['parts'].append(obj)
                obj.set_cp("category_id", 4)
                model_path = os.path.join(config['models_dir'], needle_name, obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)
                continue
            else:
                needle_vu['parts'].append(obj)
                obj.set_cp("category_id", 3)
                model_path = os.path.join(config['models_dir'], needle_name, obj.get_name() + '.ply')
                obj.set_cp('model_path', model_path)

    tracebot_objs[needle_name] = needle_vu
    return tracebot_objs

def load_tracebot_object_from_blend(tracebot_objs, path, obj_name_blend, category_id, obj_name_ply=None, obj_name_tracebot=None):
    objs = bproc.loader.load_blend(path)
    if obj_name_ply == None:
        obj_name_ply = obj_name_blend
    if obj_name_tracebot == None:
        obj_name_tracebot = obj_name_blend
    object_dict = {'parts': [], 'annos': []}
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        elif name == obj_name_blend:
            object_dict['whole'] = obj
            object_dict['annos'].append(obj)
        else:
            object_dict['parts'].append(obj)
        obj.set_cp("category_id", category_id)
        model_path = os.path.join(config['models_dir'], obj_name_ply, obj.get_name() + '.ply')
        obj.set_cp('model_path', model_path)

    tracebot_objs[obj_name_tracebot] = object_dict

    return tracebot_objs    

def load_objects_from_blend(path):
    objs = bproc.loader.load_blend(path)
    for obj in objs:
        name = obj.get_name()
        if name == 'Empty':
            continue
        else:
            obj.set_cp("category_id", 99)
            obj.hide(True) 
    return objs

def regularize_bottle_glass(obj, roughness=0.0, ior=1.5, mix_factor=0.35):
    print(obj.get_name())
    materials = obj.get_materials()
    # abs(np.random.normal(0, 0.01))
    materials[0].nodes["Glass BSDF"].inputs["Roughness"].default_value = np.abs(np.random.normal(roughness, 0.01)) 
    # np.random.normal(1.5, 0.01)
    materials[0].nodes["Glass BSDF"].inputs["IOR"].default_value = np.random.normal(ior, 0.01) 
    # np.random.normal(0.3, 0.015)
    materials[0].nodes["Mix"].inputs["Fac"].default_value = np.random.normal(mix_factor, 0.03)
    return obj

def render(config):
    bproc.init()
    # mesh_dir = os.path.join(dirname, config["models_dir"])

    dataset_name = config["dataset_name"]

    target_path = os.path.join(config['output_dir'], 'bop_data', dataset_name, 'models')
    if not os.path.isdir(target_path):
        shutil.copytree(config['models_dir'], target_path)

    tracebot = {}
    tracebot = load_needle_from_blend(tracebot, os.path.join(config["models_dir"], 'needle/needle.blend'))
    tracebot = load_needle_variation_from_blend(tracebot, os.path.join(config["models_dir"], 'needle_vd/needle_vd.blend'), 'needle_vd')
    tracebot = load_needle_variation_from_blend(tracebot, os.path.join(config["models_dir"], 'needle_vu/needle_vu.blend'), 'needle_vu')
    tracebot = load_needle_variation_from_blend(tracebot, os.path.join(config["models_dir"], 'needle_vl/needle_vl.blend'), 'needle_vl')
    tracebot = load_needle_variation_from_blend(tracebot, os.path.join(config["models_dir"], 'needle_vr/needle_vr.blend'), 'needle_vr')

    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'large_bottle/large_bottle.blend'), 'large_bottle', 7)
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'medium_bottle/medium_bottle.blend'), 'medium_bottle', 1)
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'small_bottle/small_bottle.blend'), 'small_bottle', 2)
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'canister/canister.blend'), 'canister', 6)
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'canister/canister.blend'), 'canister.001', 6, 'canister', 'canister1')
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'yellow_cap/yellow_cap.blend'), 'yellow_cap', 8)
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'yellow_cap/yellow_cap.blend'), 'yellow_cap.001', 8, 'yellow_cap', 'yellow_cap1')
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'red_cap/red_cap.blend'), 'red_cap', 5)
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'red_cap/red_cap.blend'), 'red_cap.001', 5, 'red_cap', 'red_cap1')
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'clamp/clamp_red.blend'), 'clamp_r', 10, 'clamp', 'red_clamp')
    tracebot = load_tracebot_object_from_blend(tracebot, os.path.join(config["models_dir"], 'clamp/clamp_white.blend'), 'clamp_w', 9, 'clamp', 'white_clamp')

    glass_objs = load_objects_from_blend(os.path.join(config["distractions"]["glass_distractor_path"]))
    ar_tags = load_objects_from_blend(os.path.join(config["distractions"]["artag_distractor_path"]))

    for key in tracebot.keys():
        for obj in tracebot[key]["parts"]:
            obj.hide(True) 
        tracebot[key]["whole"].hide(True) 
        for obj in tracebot[key]["annos"]:
            obj.hide(True) 

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
    def sample_pose_physics(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
        max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    def sample_pose_upright(obj: bproc.types.MeshObject):
        obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                    min_height=1, max_height=4, face_sample_range=[0.35, 0.65]))
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))        


    # activate depth rendering without antialiasing and set amount of samples for color rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(50)

    bproc.renderer.set_light_bounces(
        glossy_bounces=32, 
        max_bounces=32, 
        transmission_bounces=32, 
        transparent_max_bounces=50, 
        volume_bounces=32)    

    bproc.camera.set_intrinsics_from_K_matrix(np.reshape(config["cam"]["K"], (3, 3)), 
                                                config["cam"]["width"], 
                                                config["cam"]["height"])
    
    #save initial color values of bottle caps into dictionary
    reg_objs = ['large_bottle', 'medium_bottle', 'small_bottle']
    regularization_memory = {}
    for obj in tracebot.keys():
        if obj in reg_objs:
            for part in tracebot[obj]['parts']:
                if part.get_name()[-10:] == 'bottle_cap':
                    #get material
                    mat = part.get_materials()[0]
                    #set color of material to a random color
                    param_dict = {}
                    param_dict['Base Color'] = mat.get_principled_shader_value("Base Color")[0:4]
                    regularization_memory[part.get_name()] = param_dict
    
    for i in range(config["num_scenes"]):
        np.random.seed(i)
        random.seed(i)

        #Sample Bop Distractors
        bop_objs = []
        for bop_dataset in bop_datasets.values():
            bop_objs += bop_dataset
        dist_per_datatset = min(config["distractions"]["num_bop_distractions"], len(bop_dataset))
        sampled_distractor_bop_objs = list(np.random.choice(bop_objs, size=dist_per_datatset, replace=False))

        # Sample Glass Distractors
        glass_per_scene = min(config["distractions"]["num_glass_distractions"], len(glass_objs))
        sampled_distractor_glass_objs = list(np.random.choice(glass_objs, size=glass_per_scene, replace=False))
        for glass_obj in sampled_distractor_glass_objs:
            glass_obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            glass_obj.hide(False)
    
        needles = [
            'needle',
            'needle_vd', 
            'needle_vu', 
            'needle_vr', 
            'needle_vl'
        ]
        # Sample bop objects for a scene
        sampled_needle = []
        sampled_needle = list(np.random.choice(needles, size=1, replace=False))

        singles = [
            'large_bottle', 
            'white_clamp', 
            'red_clamp', 
            'medium_bottle', 
            'small_bottle', 
            'yellow_cap', 
            'red_cap',
            'canister'
        ]

        sampled_singles = list(np.random.choice(singles, size=7, replace=False))

        duplicates = [
            'canister1',
            'yellow_cap1', 
            'red_cap1',
        ]
        
        sampled_dublicates = list(np.random.choice(duplicates, size=2, replace=False))


        sampled_target_objs = sampled_singles + sampled_dublicates + sampled_needle

        tracebot_full_body = [tracebot[obj]['whole'] for obj in sampled_target_objs if obj in tracebot.keys()]

        # Randomize materials and set physics
        for obj in (sampled_distractor_bop_objs + tracebot_full_body):
            obj = set_material_properties(obj, cc_textures)
        
        # Sample two light sources
        light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
        light_plane.replace_materials(light_plane_material)
        light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
        location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.8,
                                elevation_min = 5, elevation_max = 89)
        light_point.set_location(location)

        # sample CC Texture and assign to room planes
        random_cc_texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)
            mat = plane.get_materials()[0]      
            mat.set_principled_shader_value("Alpha", 1.0)

        # tracebot_full_body = [tracebot[obj]['whole'] for obj in sampled_target_objs]

        upright_objects = ['small_bottle', 'large_bottle', 'medium_bottle', 'canister']
        upright = []
        physics = []
        drop_parts = []
        
        for obj in tracebot_full_body:
            if obj.get_name() in upright_objects:
                num = random.random()
                if num > 0.3:
                    upright.append(obj)
                else:
                    physics.append(obj)
            else:
                physics.append(obj)

        upright_names = [x.get_name() for x in upright]

        num = random.random()
        if num > 0.7:
            for obj in tracebot[sampled_needle[0]]['parts']:
                if obj.get_name()[0:10] == 'needle_cap':
                    obj.hide(False)
                    obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
                    drop_parts.append(obj)

        # sample 2 artags
        artag_per_scene = min(config["distractions"]["num_ar_distractions"], len(ar_tags))
        sampled_distractor_artag_objs = list(np.random.choice(ar_tags, size=artag_per_scene, replace=False))
        for artag_obj in sampled_distractor_artag_objs:
            artag_obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            artag_obj.hide(False)
        upright += sampled_distractor_artag_objs
        #shuffle upright objects
        random.shuffle(upright)

        bproc.object.sample_poses_on_surface(objects_to_sample=upright,
                                                surface=room_planes[0],
                                                sample_pose_func=sample_pose_upright,
                                                min_distance=0.01,
                                                max_distance=0.2)


        # Sample object poses and check collisions 
        bproc.object.sample_poses(objects_to_sample = sampled_distractor_bop_objs + sampled_distractor_glass_objs + physics + drop_parts,
                                sample_pose_func = sample_pose_physics, 
                                max_tries = 1000)
                
        # Physics Positioning
        bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                        max_simulation_time=10,
                                        check_object_interval=1,
                                        substeps_per_frame = 20,
                                        solver_iters=25)
   

        # BVH tree used for camera obstacle checks
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_distractor_bop_objs + sampled_distractor_glass_objs + tracebot_full_body)


        parts = []


        #sample fluidlevel for bottles:
        fluid_level = np.random.choice(["fluid1","fluid2","fluid3"])
        liquid_counter = 0
        for obj in sampled_target_objs:
            if obj not in tracebot.keys():
                continue
            whole_obj = tracebot[obj]['whole']
            pose_tmat = whole_obj.get_local2world_mat()
            whole_obj.disable_rigidbody()
            whole_obj.hide(True)
            for part in tracebot[obj]['annos']:
                if part not in drop_parts:
                    part.set_local2world_mat(pose_tmat)
                    part.hide(True) 
            for part in tracebot[obj]['parts']:
                if part not in drop_parts:
                    part.set_local2world_mat(pose_tmat)

                    if part.get_name()[-7:] == 'cap_hat':
                        #with 15% no cap
                        num = random.random()
                        if num > 0.85:
                            continue
                    if part.get_name()[-10:] == 'bottle_cap':
                        #with 10% chance cap has different color
                        num = random.random()
                        mat = part.get_materials()[0]
                        mat.set_principled_shader_value("Base Color", regularization_memory[part.get_name()]['Base Color'])
                        if num > 0.9:
                            mat.set_principled_shader_value("Base Color", np.random.uniform([0.1, 0.1, 0.1, 1.0], [1.0, 1.0, 1.0, 1.0]))
                    if obj in ['large_bottle', 'medium_bottle', 'small_bottle']:
                        print(obj)
                        if len(part.get_name().split("_")) > 2:
                            if part.get_name().split("_")[2] == 'bottle':
                                part = regularize_bottle_glass(part)

                            part_type = part.get_name().split("_")[2]
                            is_liquid = part_type[0:5] == "fluid"
                            if is_liquid:
                                part.hide(True)
                                if obj in upright_names and fluid_level == part_type:
                                    num = random.random()
                                    if num > 0.8:
                                        part.hide(False)
                                        #sample color of glass shader
                                        mat = part.get_materials()
                                        #mat[0].nodes["Glass BSDF"].inputs["Color"].default_value = np.abs(np.random.normal(roughness, 0.01)) 
                                        mat[0].nodes["Glass BSDF"].inputs["Color"].default_value = np.random.uniform([0.1, 0.1, 0.1, 1.0], [1.0, 1.0, 1.0, 1.0])
                                        parts.append(part)
                                continue

                    part.enable_rigidbody(False, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
                    part.hide(False) 
                    parts.append(part)

        #iterate over all parts and check if liquids are hidden
        cam_poses = 0
        for obj in tracebot.keys():
            whole_obj = tracebot[obj]['whole']
            whole_obj.disable_rigidbody()
            whole_obj.hide(True)        

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
            if obj not in tracebot.keys():
                continue
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

        for obj in (parts + sampled_distractor_bop_objs + drop_parts + sampled_distractor_glass_objs + sampled_distractor_artag_objs + physics + upright):      
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


    
    

    