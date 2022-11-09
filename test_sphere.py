"""
s Testing the absorption material on 1 sphere with a cone beam
"""

import logging
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
import numpy as np
import matplotlib.pyplot as plt
from pvtrace import *

world = Node(
    name="world (air)",
    geometry=Sphere(
        radius=10.0,
        material=Material(refractive_index=1.0,

                          )
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
sphere = Node(
    name="sphere (glass)",
    geometry=Sphere(
        radius=0.8,
        material=Material(
            refractive_index=1.0,
            components=[
                Scatterer(coefficient=4.1)
            ]

        ),
    ),
    parent=world
)
sphere2 = Node(
    name="sphere (glass)",
    geometry=Sphere(
        radius=0.81,
        material=Material(
            refractive_index=1.0,

        ),
    ),
    parent=world
)
ray = Ray(
    position=(-1.0, 0.0, 0.9),
    direction=(1.0, 0.0, 0.0),
    wavelength=555.0
)
import functools

light = Node(
    name="Light (555nm)",
    light=Light(direction=functools.partial(cone, np.pi / 32)),
    parent=world
)
light.translate((0.0, 0.0, 3))
light.rotate(np.pi, [1, 0, 0])
scene = Scene(world)
np.random.seed(0)
# vis = MeshcatRenderer(wireframe=True, open_browser=True)
# vis.render(scene)
positions = []
for ray in scene.emit(4):
    steps = photon_tracer.follow(scene, ray)
    path, decisions = zip(*steps)
    positions_ray = []
    for ray in path:
        positions_ray.append(ray.position)
    positions.append(positions_ray)
    # for position in path[0].position
    # vis.add_ray_path(path)

# vis.render(scene)
import time
# time.sleep(10)
# vis.vis.jupyter_cell()
