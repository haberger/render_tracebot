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

    objs = bproc.loader.load_blend(config["blend_dir"])

    dataset_name = config["dataset_name"]

    for i, obj in enumerate(objs):
        obj.set_cp("category_id", "obj_00000"+str(i))

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([-1, 0, 0.5])
    light.set_energy(50)

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

    cam2world_matrix = np.array([
        [0, 0,  -1, -0.2],
        [-1, 0, 0, 0],
        [0,  1,  0, 0],
        [0, 0, 0, 1]
    ])

    bproc.camera.add_camera_pose(cam2world_matrix)

    stepsize_grad=10 #in degree
    stepsize_rad = stepsize_grad*(np.pi/180)

    alpha = np.linspace(0, 2*np.pi, int((2*np.pi)/stepsize_rad))
    beta = np.linspace(0, np.pi, int((np.pi)/stepsize_rad))

    # render the whole pipeline
    for b in beta[:-1]:
        # print(b)
        for a in alpha[:-1]:
            for obj in objs:
                obj.set_rotation_euler([a,b, 0])
            data = bproc.renderer.render()
    
        # Write data in bop format
        
            bproc.writer.write_bop(os.path.join(config["output_dir"], 'bop_data'),
                                target_objects = objs,
                                dataset = dataset_name,
                                depth_scale = 0.1,
                                depths = data["depth"],
                                colors = data["colors"], 
                                color_file_format = "JPEG",
                                ignore_dist_thres = 10)
    

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


    
    

    