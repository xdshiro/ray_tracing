"""
This module simulates the integration sphere using pvtrace package.
Inside the sphere we can place different objects and structures.
Absorption is included.
"""

# from ipywidgets import interact
# from pvtrace import *
import pvtrace as pv
import numpy as np
import functools
import logging
import time

# Some packages used by pvtrace are a little noisy
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True

m1 = pv.Material(
    refractive_index=1.5,
)
m2 = pv.Material(
    refractive_index=1.5,
    components=[
        pv.Absorber(coefficient=10)  # cm-1
    ]
)
bubble = m2
material_int_sphere = pv.Material(
    refractive_index=3,
    components=[
        pv.Absorber(coefficient=10)  # cm-1
    ]
)


def interact_ray(scene, vis):
    """
    Function computes rays
    :param scene: world + all objects + all light_sources
    :param vis: rendered scene, it's needed to automatically plot the light on top of it
    :return: None
    """
    ray_ids = []
    # Clear old objects
    [vis.remove_object(_id) for _id in ray_ids]
    # Re-create the scene with the new ray but reuse the renderers
    renderer = pv.MeshcatRenderer(wireframe=True, open_browser=True)
    renderer.render(scene)
    for ray in scene.emit(5):
        steps = pv.photon_tracer.follow(scene, ray, maxsteps=10)
        path, events = zip(*steps)
        vis.render(scene)
        renderer.add_ray_path(path)


def structure_bubbles(parent):
    """
    3D rays of small spheres
    :param parent: world
    :return: None
    """

    a = 0.4
    x_ar = np.linspace(-5 * a, 5 * a, 11)
    y_ar = [-3 * a, -2 * a, -a, 0, a, 2 * a, 3 * a]
    z_ar = [-3 * a, -2 * a, -a, 0, a, 2 * a, 3 * a]
    for x in x_ar:
        for y in y_ar:
            for z in z_ar:
                sphere2 = pv.Node(
                    name=f"sphere: {x}, {y}, {z} (glass)",
                    geometry=pv.Sphere(
                        radius=a / 2,
                        material=bubble,
                    ),
                    parent=parent

                )
                sphere2.translate((x, y, z))


def pv_integrating_sphere(structure=structure_bubbles):
    """
    This function simulate the integrating sphere and the structure() inside it.
    Light source is described here as well.
    :structure: objects you want to modulate inside the integrating sphere
    :return: None
    """
    world = pv.Node(
        name="world (air)",
        geometry=pv.Sphere(
            radius=10.0,
            material=pv.Material(refractive_index=1.0),
        )
    )
    # sphere1 = pv.Node(
    #     name="sphere1 (glass)",
    #     geometry=pv.Sphere(
    #         radius=8.0,
    #         material=material_int_sphere,
    #     ),
    #     parent=world
    # )
    # sphere1.translate((0.0, 0.0, 0.0))

    # structure(parent=world)
    a = 0.4
    x_ar = np.linspace(-5 * a, 5 * a, 11)
    y_ar = [-3 * a, -2 * a, -a, 0, a, 2 * a, 3 * a]
    z_ar = [-3 * a, -2 * a, -a, 0, a, 2 * a, 3 * a]
    for x in x_ar:
        for y in y_ar:
            for z in z_ar:
                sphere2 = pv.Node(
                    name=f"sphere: {x}, {y}, {z} (glass)",
                    geometry=pv.Sphere(
                        radius=a / 2,
                        material=bubble,
                    ),
                    parent=world

                )
                sphere2.translate((x, y, z))
                break
            break
        break
    # sphere = pv.Node(
    #     name="sphere (glass)",
    #     geometry=pv.Sphere(
    #         radius=1.0,
    #         material=pv.Material(
    #             refractive_index=1.5,
    #             components=[
    #
    #                 pv.Absorber(coefficient=2.1)
    #             ]
    #         ),
    #     ),
    #     parent=world
    # )

    light = pv.Node(
        name="Light (555nm)",
        light=pv.Light(direction=functools.partial(pv.cone, np.pi / 32)),
        parent=world
    )
    light.translate((0.0, 0.0, 7.95))
    light.rotate(np.pi, [1, 0, 0])
    scene = pv.Scene(world)
    renderer = pv.MeshcatRenderer(wireframe=True, open_browser=True)
    renderer.render(scene)
    for ray in scene.emit(2500):
        steps = pv.photon_tracer.follow(scene, ray, maxsteps=10)
        path, events = zip(*steps)
        renderer.add_ray_path(path)
    renderer.render(scene)
    # renderer.vis.jupyter_cell()


if __name__ == '__main__':
    pv_integrating_sphere()
