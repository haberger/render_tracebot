import blenderproc as bproc
import argparse
import numpy as np
import bpy

bproc.init()

# load the objects into the scene
# obj = bproc.loader.load_obj(args.object)[0]
objs = bproc.loader.load_blend(
    "/home/david/tracebot-management/TraceBotMeshesRealistic/needle/needle.blend",
    data_blocks=['objects', 'cameras']
    )
# Use vertex color for texturing
# for obj in objs:
#     print(obj.get_local2world_mat())
print(objs[-1])
print()
print(bpy.data.objects[-1])

print(objs[-1].angle_x)
obj = objs[0]
for mat in obj.get_materials():
    mat.map_vertex_color()
# Set pose of object via local-to-world transformation matrix
#obj.set_local2world_mat(
#    [[0.331458, -0.9415833, 0.05963787, -0.04474526765165741],
#    [-0.6064861, -0.2610635, -0.7510136, 0.08970402424862098],
#    [0.7227108, 0.2127592, -0.6575879, 0.6823395750305427],
#    [0, 0, 0, 1.0]]
#)
# Scale 3D model from mm to m
obj.set_scale([0.001, 0.001, 0.001])
# Set category id which will be used in the BopWriter
obj.set_cp("category_id", 1)

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([-1, 0, 0.5])
light.set_energy(50)

# Set intrinsics via K matrix
bproc.camera.set_intrinsics_from_K_matrix(
    [[537.4799, 0.0, 318.8965],
     [0.0, 536.1447, 238.3781],
     [0.0, 0.0, 1.0]], 640, 480
)
#Set camera pose via cam-to-world transformation matrix

#location=[0.3466,-0.177, 0.0359]
# Determine point of interest in scene as the object closest to the mean of a subset of objects
#poi = bproc.object.compute_poi([obj]) #
# Compute rotation based on vector going from location towards poi
#rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(0, 0))
# Add homog cam pose based on location an rotation
#cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

cam2world_matrix = np.array([
    [0.4560809, -0.0750993,  0.8867639, 0.3466],
    [0.8898147,  0.0550927, -0.4529842, -0.177],
    [-0.0148354,  0.9956530,  0.0919511, 0.03593],
    [0, 0, 0, 1]
])

cam2world_matrix = np.array([
    [0, 0,  -1, -0.2],
    [-1, 0, 0, 0],
    [0,  1,  0, 0],
    [0, 0, 0, 1]
])

# # Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
#cam2world_matrix = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world_matrix, ["X", "-Y", "-Z"])
bproc.camera.add_camera_pose(cam2world_matrix)

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
max_bounces = 10
bproc.renderer.set_light_bounces(glossy_bounces=max_bounces, max_bounces=max_bounces, transmission_bounces=max_bounces, transparent_max_bounces=max_bounces, volume_bounces=max_bounces)

# render the whole pipeline
data = bproc.renderer.render()

# Write object poses, color and depth in bop format
bproc.writer.write_bop("outpout", [obj], data["depth"], data["colors"], m2mm=True, append_to_existing_output=True)