import time
import sys
import functools
import numpy as np
from pvtrace import *
import logging
logging.getLogger('trimesh').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

world = Node(
    name="world (air)",
    geometry=Sphere(
        radius=50.0,
        material=Material(refractive_index=1.0),
    )
)

box = Node(
    name="sphere (glass)",
    geometry=Box(
        (10.0, 10.0, 1.0),
        material=Material(refractive_index=1.5),
    ),
    parent=world
)

light = Node(
    name="Light (555nm)",
    light=Light(),
    parent=world
)
light.rotate(np.radians(60), (1.0, 0.0, 0.0))

start_t = time.time()
scene = Scene(world)
vis = MeshcatRenderer(wireframe=True, open_browser=True)

for ray in scene.emit(100):
    steps = photon_tracer.follow(scene, ray)
    path, decisions = zip(*steps)
    vis.add_ray_path(path)
vis.render(scene)
import time
time.sleep(10)
print(f"Took {time.time() - start_t}s to trace 100 rays.")