import blenderproc as bproc
# from curses.panel import bottom_panel
# from logging import raiseExceptions
# from sre_parse import CATEGORIES
# from unicodedata import category
import argparse
import os
import numpy as np
import yaml
import shutil

# TODO fix bump and physics simulTION CENTER OF MASS
# TODO ALPHA
# TODO if bop Dataset dosesn nopt have info generate info

def render(config):

    bproc.init()

    dataset = "ycbv"
    bop_datasets_path = config["bop_dir"]
    bop_dataset_path = os.path.join(bop_datasets_path, dataset)
    bop_objects = bproc.loader.load_bop_objs(
        bop_dataset_path = bop_dataset_path, 
        mm2m = True)

    for obj in bop_objects:
        obj.hide(True)

    # light = bproc.types.Light()
    # light.set_type("POINT")
    # light.set_location([-1, 0, 0.5])
    # light.set_energy(50)

    room_size = 1

    light_plane_top = bproc.object.create_primitive('PLANE', scale=[room_size * 1.5, room_size * 1.5, 1], location=[0, 0, room_size*5])
    light_plane_bot = bproc.object.create_primitive('PLANE', scale=[room_size * 1.5, room_size * 1.5, 1], location=[0, 0, -room_size*5])
    light_plane1 = bproc.object.create_primitive('PLANE', scale=[room_size* 1.5, room_size* 1.5, 1], location=[0, -room_size*5, room_size*5], rotation=[-1.570796, 0, 0])
    light_plane2 = bproc.object.create_primitive('PLANE', scale=[room_size* 1.5, room_size* 1.5, 1], location=[0, room_size*5, room_size*5], rotation=[1.570796, 0, 0])
    light_plane3 = bproc.object.create_primitive('PLANE', scale=[room_size* 1.5, room_size* 1.5, 1], location=[-room_size*5, 0, room_size*5], rotation=[0, 1.570796, 0])
    light_plane1.set_name('light_plane1')
    light_plane2.set_name('light_plane2')
    light_plane3.set_name('light_plane3')
    light_plane_top.set_name('light_plane_top')
    light_plane_bot.set_name('light_plane_bot')
    light_planes = [
        light_plane_bot, 
        light_plane_top,
        light_plane1,
        light_plane2,
        light_plane3]

    light_plane_material = bproc.material.create('light_material')
    light_plane_material.make_emissive(emission_strength=5, 
                                    emission_color=[0.7, 0.7, 0.7, 1.0]) 
    for plane in light_planes:
        plane.replace_materials(light_plane_material) 

    # activate depth rendering without antialiasing and set amount of samples for color rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_world_background([0,0,0], strength=0.0)
    bproc.renderer.set_max_amount_of_samples(25)
    # max_bounces = 10
    # bproc.renderer.set_light_bounces(
    #     glossy_bounces=max_bounces, 
    #     max_bounces=max_bounces, 
    #     transmission_bounces=max_bounces, 
    #     transparent_max_bounces=max_bounces, 
    #     volume_bounces=max_bounces)
    
    bproc.camera.set_intrinsics_from_K_matrix(np.reshape(config["cam"]["K"], (3, 3)), 
                                                config["cam"]["width"], 
                                                config["cam"]["height"])
    cam2world_matrix = np.array([
        [0, 0,  -1, -0.5],
        [-1, 0, 0, 0],
        [0,  1,  0, 0],
        [0, 0, 0, 1]
    ])

    bproc.camera.add_camera_pose(cam2world_matrix)
    
    for obj in bop_objects:
        obj.hide(False)
        dataset_name = dataset + str(obj.get_cp("category_id"))

        stepsize_grad=config["stepsize"] #in degree
        stepsize_rad = stepsize_grad*(np.pi/180)

        alpha = np.linspace(0, 2*np.pi, int((2*np.pi)/stepsize_rad))
        beta = np.linspace(0, np.pi, int((np.pi)/stepsize_rad))

        # render the whole pipeline
        for b in beta[:-1]:
            # print(b)
            for a in alpha:
                obj.set_rotation_euler([a,b, 0])
                data = bproc.renderer.render()
        
            # Write data in bop format
            
                bproc.writer.write_bop(os.path.join(config["output_dir"], 'bop_data'),
                                    target_objects = [obj],
                                    dataset = dataset_name,
                                    depth_scale = 0.1,
                                    depths = data["depth"],
                                    colors = data["colors"], 
                                    color_file_format = "JPEG",
                                    ignore_dist_thres = 10)
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


    
    

    