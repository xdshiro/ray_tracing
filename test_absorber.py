"""
Testing the absorption material on 1 sphere with a cone beam
"""

import logging

logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
import numpy as np
import matplotlib.pyplot as plt
import pvtrace as pv


world = pv.Node(
    name="world (air)",
    geometry=pv.Sphere(
        radius=10.0,
        material=pv.Material(refractive_index=1.0)
    )
)

# sphere = Node(
#     name="sphere (abs)",
#     geometry=Sphere(
#         radius=1.0,
#         material=Material(
#             refractive_index=1.5,
#             components=[
#
#                 Absorber(coefficient=12.1)
#             ]
#         ),
#     ),
#     parent=world
# )
# sphere = Node(
#     name="sphere (glass)",
#     geometry=Sphere(
#         radius=0.8,
#         material=Material(
#             refractive_index=1.5,
#
#
#         ),
#     ),
#     parent=world
# )
# ray = Ray(
#     position=(-1.0, 0.0, 0.9),
#     direction=(1.0, 0.0, 0.0),
#     wavelength=555.0
# )
cylinder1 = pv.Node(
    name="cyl1 (glass)",
    geometry=pv.Cylinder(
        radius=0.8,
        length=5.5,
        material=pv.Material(
            refractive_index=1.5,
            components=[
                pv.Absorber(coefficient=1.1),
                pv.Scatterer(coefficient=1.1)
            ]
        ),
    ),
    parent=world
)
cylinder1.rotate(np.pi / 2, [0, 1, 0])
import functools

light = pv.Node(
    name="Light (555nm)",
    light=pv.Light(direction=functools.partial(pv.cone, np.pi / 32)),
    parent=world
)
light.translate((0.0, 0.0, 3))
light.rotate(np.pi, [1, 0, 0])
scene = pv.Scene(world)
np.random.seed(0)
vis = pv.MeshcatRenderer(wireframe=True, open_browser=True)
vis.render(scene)
for ray in scene.emit(2500):
    steps = pv.photon_tracer.follow(scene, ray)
    path, decisions = zip(*steps)
    vis.add_ray_path(path)
vis.render(scene)
# vis.vis.jupyter_cell()
