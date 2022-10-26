import logging
# Some packages used by pvtrace are a little noisy
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
from ipywidgets import interact
from pvtrace import *
import numpy as np

def interact_ray(scene, vis):
    ray_ids = []

    def move_ray(x=0.0, y=0.0, z=0.0, theta=0.0, phi=0.0, nanometers=555.0):
        # Clear old objects
        [vis.remove_object(_id) for _id in ray_ids]

        # Create new array with position from the interact UI
        ray = Ray(
            position=(x, y, z),
            direction=(
                np.sin(np.radians(theta)) * np.cos(np.radians(phi)),
                np.sin(np.radians(theta)) * np.sin(np.radians(phi)),
                np.cos(np.radians(theta))
            ),
            wavelength=nanometers
        )

        # Re-create the scene with the new ray but reuse the renderers
        steps = photon_tracer.follow(scene, ray, maxsteps=10)
        path, events = zip(*steps)
        vis.render(scene)

        # Remove old rays; add new rays
        ray_ids.clear()
        ray_ids.extend(vis.add_ray_path(path))

    return interact(
        move_ray,
        x=(-0.6, 0.6, 0.01),
        y=(-0.6, 0.6, 0.01),
        z=(-0.6, 0.6, 0.01),
        theta=(0, 180, 1),
        phi=(0, 360, 1),
        nanometers=(300, 800, 1)
    )


def pv_sphere_test():
    world = Node(
        name="world (air)",
        geometry=Sphere(
            radius=10.0,
            material=Material(refractive_index=1.0),
        )
    )
    sphere = Node(
        name="sphere (glass)",
        geometry=Sphere(
            radius=1.0,
            material=Material(refractive_index=1.5),
        ),
        parent=world
    )
    scene = Scene(world)

    renderer = MeshcatRenderer(wireframe=True)
    renderer.render(scene)
    renderer.vis.jupyter_cell()

    _ = interact_ray(scene, renderer)


if __name__ == '__main__':
    pv_sphere_test()

