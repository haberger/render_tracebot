import blenderproc as bproc
import argparse
import numpy as np
import bpy
import time

bproc.init()

# load the objects into the scene
# obj = bproc.loader.load_obj(args.object)[0]
objs = bproc.loader.load_blend(
    "/home/david/tracebot-management/TraceBotMeshesRealistic/needle/needle.blend",
    data_blocks=['objects']
    )

obj = objs[0]
for mat in obj.get_materials():
    mat.map_vertex_color()
# Set pose of object via local-to-world transformation matrix

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([-1, 0, 0.5])
light.set_energy(50)

# Set intrinsics via K matrix

cam2world_matrix = np.array([
    [0, 0,  -1, -0.2],
    [-1, 0, 0, 0],
    [0,  1,  0, 0],
    [0, 0, 0, 1]
])

bproc.camera.add_camera_pose(cam2world_matrix)

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
max_bounces = 10
bproc.renderer.set_light_bounces(glossy_bounces=max_bounces, max_bounces=max_bounces, transmission_bounces=max_bounces, transparent_max_bounces=max_bounces, volume_bounces=max_bounces)

# render the whole pipeline

stepsize_grad=10 #in degree
stepsize_rad = stepsize_grad*(np.pi/180)

alpha = np.linspace(0, 2*np.pi, int((2*np.pi)/stepsize_rad))
beta = np.linspace(0, np.pi, int((np.pi)/stepsize_rad))

for i, obj in enumerate(objs):
    obj.set_cp("category_id", "obj_00000"+str(i))

t0 = time.time()


for b in beta[:-1]:
    for a in alpha[:-1]:
        for obj in objs:
            obj.set_rotation_euler([a,b, 0])
        data = bproc.renderer.render()

        # Write object poses, color and depth in bop format
        bproc.writer.write_bop("outpout", [obj], data["depth"], data["colors"], m2mm=True, append_to_existing_output=True)

print(time.time()-t0)