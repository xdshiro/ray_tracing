"""
This module simulates the integration sphere using pvtrace package.
Inside the sphere we can place different objects and structures.
Absorption is included.
"""
import logging

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
# from ipywidgets import interact
# from pvtrace import *
import pvtrace as pv
import numpy as np
import functools

import time
import matplotlib.pyplot as plt
import cross_sections as cs  # mine
import my_functions.functions_general as fg
import my_functions.singularities as sing
import my_functions.plotings as pl
import my_functions.beams_and_pulses as bp

# Some packages used by pvtrace are a little noisy


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
    DOESN'T WORK
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
    sphere1 = pv.Node(
        name="sphere1 (glass)",
        geometry=pv.Sphere(
            radius=8.0,
            material=material_int_sphere,
        ),
        parent=world
    )
    sphere1.translate((0.0, 0.0, 0.0))

    structure(parent=world)
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
    for ray in scene.emit(250):
        steps = pv.photon_tracer.follow(scene, ray, maxsteps=10)
        path, events = zip(*steps)
        renderer.add_ray_path(path)
        time.sleep(1)
    renderer.render(scene)
    # renderer.vis.jupyter_cell()


def spherical_to_cart(theta, phi, r=1):
    """
    transformation of the spherical coordinates to cartesiane coordinates
    :return: x y z
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    cart = np.column_stack((x, y, z))
    if cart.size == 3:
        return cart[0, :]
    return cart


def cylindrical_to_cart(r, phi, z=0):
    """
    transformation of the spherical coordinates to cartesiane coordinates
    :return: x y z
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = z
    cart = np.column_stack((x, y, z))
    if cart.size == 3:
        return cart[0, :]
    return cart


def cone(theta_max):
    """ Samples directions within a cone of solid angle defined by `theta_max`.

        Notes
        -----
        Derived as follows using sympy::

            from sympy import *
            theta, theta_max, p = symbols('theta theta_max p')
            f = cos(theta) * sin(theta)
            cdf = integrate(f, (theta, 0, theta))
            pdf = cdf / cdf.subs({theta: theta_max})
            inv_pdf = solve(Eq(pdf, p), theta)[-1]
    """
    if np.isclose(theta_max, 0.0) or theta_max > np.pi / 2:
        raise ValueError("Expected 0 < theta_max <= pi/2")
    p1, p2 = np.random.uniform(0, 1, 2)
    theta = np.arcsin(np.sqrt(p1) * np.sin(theta_max))
    phi = 2 * np.pi * p2
    coords = spherical_to_cart(theta, phi)
    return coords


def collimated_beam(r):
    """ Samples directions within a cone of solid angle defined by `theta_max`.

        Notes
        -----
        Derived as follows using sympy::

            from sympy import *
            theta, theta_max, p = symbols('theta theta_max p')
            f = cos(theta) * sin(theta)
            cdf = integrate(f, (theta, 0, theta))
            pdf = cdf / cdf.subs({theta: theta_max})
            inv_pdf = solve(Eq(pdf, p), theta)[-1]
    """
    if np.isclose(r, 0.0):
        raise ValueError("Expected 0 < r")
    r_random = np.random.normal(0, r, 1)
    p2 = np.random.uniform(0, 1, 1)
    phi = 2 * np.pi * p2
    coords = cylindrical_to_cart(r_random, phi, 0)

    return coords


def light_beam(parent):
    r = 0.05
    light = pv.Node(
        name="Light (555nm)",
        light=pv.Light(position=functools.partial(collimated_beam, r),
                       wavelength=lambda: 555
                       ),
        parent=parent,
    )
    light.translate([0, 0, 8])
    light.rotate(np.pi, [1, 0, 0])


def structure_box(parent):
    # box = pv.Node(
    #     name="box_1",
    #     geometry=pv.Box(
    #         (2.0, 1.0, 2),
    #         material=pv.Material(
    #             refractive_index=1.08,
    #             components=[
    #                 pv.Absorber(coefficient=0.098),
    #                 pv.Scatterer(coefficient=5.409)
    #             ]
    #
    #         ),
    #     ),
    #     parent=parent
    # )
    # box = pv.Node(
    #     name="box_1",
    #     geometry=pv.Box(
    #         (2.0, 1.0, 2),
    #         material=pv.Material(
    #             refractive_index=1.08,
    #             components=[
    #                 pv.Absorber(coefficient=0.1703),
    #                 pv.Scatterer(coefficient=2.181)
    #             ]
    #
    #         ),
    #     ),
    #     parent=parent
    # )
    # box = pv.Node(
    #     name="box_1",
    #     geometry=pv.Box(
    #         (2.0, 1.0, 2),
    #         material=pv.Material(
    #             refractive_index=1.08,
    #             components=[
    #                 pv.Absorber(coefficient=0.1666),
    #                 pv.Scatterer(coefficient=0.9063)
    #             ]
    #
    #         ),
    #     ),
    #     parent=parent
    # )

    from pvtrace import SurfaceDelegate
    class PartialTopSurfaceMirror(pv.FresnelSurfaceDelegate):
        """ A section of the top surface is covered with a perfect mirrrors.

            All other surface obey the Fresnel equations.
        """
        # print(super(PartialTopSurfaceMirror, self).reflected_direction(surface, ray, geometry, container,
        #                                                                adjacent))

        def reflected_direction(self, surface, ray, geometry, container, adjacent):
            TOP_SURFACE = (0, 0, 1)
            BOT_SURFACE = (0, 0, -1)
            LEFT_SURFACE = (-1, 0, 0)
            RIGHT_SURFACE = (1, 0, 0)
            FRONT_SURFACE = (0, -1, 0)
            BACK_SURFACE = (0, 1, 0)
            theta = np.pi/3
            phi = np.pi/5
            r=1
            x0, y0, z0 = ray.position[0], ray.position[1], ray.position[2]
            normal = geometry.normal(ray.position)
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            if np.allclose(normal, TOP_SURFACE):
                return 0, 0, -1
            elif np.allclose(normal, BOT_SURFACE):
                return x, y, z
            elif np.allclose(normal, LEFT_SURFACE):
                return 1, 0, 0
            elif np.allclose(normal, RIGHT_SURFACE):
                return -1, 0, 0
            elif np.allclose(normal, FRONT_SURFACE):
                return 0, 1, 0
            elif np.allclose(normal, BACK_SURFACE):
                return 0, -1, 0
            return normal
            # return x, y, z


        def reflectivity(self, surface, ray, geometry, container, adjacent):
            """ Return the reflectivity of the part of the surface hit by the ray.

                Parameters
                ----------
                surface: Surface
                    The surface object belonging to the material.
                ray: Ray
                    The ray hitting the surface in the local coordinate system of the `geometry` object.
                geometry: Geometry
                    The object being hit (e.g. Sphere, Box, Cylinder, Mesh etc.)
                container: Node
                    The node containing the ray.
                adjacent: Node
                    The node that will contain the ray if the ray is transmitted.
            """
            # Get the surface normal to determine which surface has been hit.
            normal = geometry.normal(ray.position)
            # print(self, surface, ray, geometry, container, adjacent)
            # exit()
            # Normal are outward facing

            x, y, z = ray.position[0], ray.position[1], ray.position[2]

            if np.isclose(z, 1) and np.abs(x) < 0.3 and np.abs(y) < 0.3:
                return super(PartialTopSurfaceMirror, self).reflectivity(surface, ray, geometry, container,
                                                                         adjacent)
            else:
                # print(x, y, z)
                return 0.9
            # If a ray hits the top surface where x > 0 and y > 0 reflection
            # set the reflectivity to 1.
            if np.allclose(normal, BOT_SURFACE):
                x, y = ray.position[0], ray.position[1]
                if x > -0.5 and y > -0.5:
                    return 1
            print(super(PartialTopSurfaceMirror, self).reflected_direction(surface, ray, geometry, container,
                                                                     adjacent))
            # direction = self.phase_function()
            # ray = replace(ray, direction=direction, source=self.name)
            # return ray
            # direction = self.delegate.reflected_direction(
            #     self, ray, geometry, container, adjacent
            # )
            # Otherwise return the Frensel reflection probability.
            return super(PartialTopSurfaceMirror, self).reflectivity(surface, ray, geometry, container,
                                                                     adjacent)  # opt-out of handling custom reflection

    # box2 = pv.Node(
    #     name="box_2",
    #     geometry=pv.Box(
    #         (2.0 * 1.001, 1.0 * 1.001, 2 * 1.001),
    #         material=pv.Material(
    #             refractive_index=1.08,
    #             components=[
    #                 pv.Absorber(coefficient=1.),
    #                 pv.Scatterer(coefficient=5000.001)
    #             ],
    #             surface=pv.Surface(delegate=PartialTopSurfaceMirror())

            # ),
        # ),
        # parent=parent
    # )
    box = pv.Node(
        name="box_1",
        geometry=pv.Box(
            (2.0, 1.0, 2),
            # radius=3,
            material=pv.Material(
                refractive_index=1.05,
                # surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
                components=[
                    pv.Absorber(coefficient=0.1),
                    pv.Scatterer(coefficient=0.00001)
                ],

            ),
        ),
        parent=parent
    )
    # box3 = pv.Node(
    #     name="box_3",
    #     geometry=pv.Box(
    #         (0.5, 0.5, 0.02),
    #         material=pv.Material(
    #             refractive_index=1.08,
    #             components=[
    #                 pv.Absorber(coefficient=0.001),
    #                 pv.Scatterer(coefficient=0.001)
    #             ],
    # surface=pv.Surface(delegate=PartialTopSurfaceMirror())
    #
    # ),
    # ),
    # parent=parent
    # )
    # box3.translate((0, 0, 1))

    #
    # box3 = pv.Node(
    #     name="box_3",
    #     geometry=pv.Box(
    #         (0.5, 0.5, 0.02),
    #         material=pv.Material(
    #             refractive_index=1.08,
    #             components=[
    #                 pv.Absorber(coefficient=0.001),
    #                 pv.Scatterer(coefficient=0.001)
    #             ],
                # surface=pv.Surface(delegate=PartialTopSurfaceMirror())
            #
            # ),
        # ),
        # parent=parent
    # )
    # box3.translate((0, 0, 1))


def pv_scene_real(structure=structure_box, light=light_beam):
    """

    :param structure:
    :param light:
    :return:
    """
    world = pv.Node(
        name="world (air)",
        geometry=pv.Sphere(
            radius=10.0,
            material=pv.Material(refractive_index=1.0),
        )
    )
    structure(parent=world)
    light(parent=world)
    scene = pv.Scene(world)
    return scene


def main_create_scene_test():
    world = pv.Node(
        name="world (air)",
        geometry=pv.Sphere(
            radius=7.0,
            material=pv.Material(refractive_index=1.0,

                                 )
        )
    )
    # pv.Node(
    #     name="sphere1 (glass)",
    #     geometry=pv.Sphere(
    #         radius=0.8,
    #         material=pv.Material(
    #             refractive_index=1.0,
    #
    #         ),
    #     ),
    #     parent=world
    # )
    # sphere2 = pv.Node(
    #     name="sphere2 (glass)",
    #     geometry=pv.Sphere(
    #         radius=0.35,
    #         material=pv.Material(
    #             refractive_index=1.0,
    #             components=[
    #                 pv.Scatterer(coefficient=3.1)
    #             ]
    #         ),
    #     ),
    #     parent=world
    # )
    # sphere2.translate((0, 0.3, 0.3))
    # sphere3 = pv.Node(
    #     name="sphere3 (glass)",
    #     geometry=pv.Sphere(
    #         radius=0.3,
    #         material=pv.Material(
    #             refractive_index=1.0,
    #             components=[
    #                 pv.Scatterer(coefficient=3.1)
    #             ]
    #         ),
    #     ),
    #     parent=world
    # )
    # sphere3.translate((0, -0.2, -0.2))
    # a = 0.4
    # x_ar = np.linspace(-5 * a, 5 * a, 11)
    # # y_ar = [-3 * a, -2 * a, -a, 0, a, 2 * a, 3 * a]
    # # y_ar = [-2 * a, -a, 0, a, 2 * a, 3 * a]
    # y_ar = [-2 * a, -a, 0, a]
    # # z_ar = [-3 * a, -2 * a, -a, 0, a, 2 * a, 3 * a]
    # z_ar = [-a, 0, a]
    # for x in x_ar:
    #     for y in y_ar:
    #         for z in z_ar:
    #             sphere2 = pv.Node(
    #                 name=f"sphere: {x}, {y}, {z} (glass)",
    #                 geometry=pv.Sphere(
    #                     radius=a / 2,
    #                     material=pv.Material(
    #                         refractive_index=1.4,
    #                         components=[
    #                             pv.Scatterer(coefficient=2.1)
    #                         ]
    #                     ),
    #                 ),
    #                 parent=world
    #
    #             )
    #             sphere2.translate((x, y, z))
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

    # cylinder1.translate((0.0, -0.3, 0.2))
    import functools
    light = pv.Node(
        name="Light (555nm)",
        light=pv.Light(direction=functools.partial(pv.cone, np.pi / 32)),
        parent=world
    )
    light.translate((0.0, 0.0, 7))
    light.rotate(np.pi, [1, 0, 0])
    scene = pv.Scene(world)
    return scene


def field_from_crossings_2D(crossings_x, crossings_y, x_res, y_res, x_max_min=(-1, 1), y_max_min=(-1, 1)):
    grid_xy = fg.create_mesh_XY(xMinMax=x_max_min, yMinMax=y_max_min, xRes=x_res, yRes=y_res)
    xAr_, yAr_ = fg.arrays_from_mesh(grid_xy)
    scale_coeff_x = 1 / (xAr_[1] - xAr_[0])
    scale_coeff_y = 1 / (yAr_[1] - yAr_[0])
    crossings_scaled_x = crossings_x * scale_coeff_x
    crossings_scaled_y = crossings_y * scale_coeff_y
    crossings_scaled_x_round = np.rint(crossings_scaled_x).astype(int)  # !!!!!!!!!!!!!!!!
    crossings_scaled_y_round = np.rint(crossings_scaled_y).astype(int)
    # crossings_scaled = np.multiply(crossings_scaled_round, 1 / scale_coeff_xyz)
    field = np.zeros((x_res, y_res))
    for dot_scaled in zip(crossings_scaled_x_round, crossings_scaled_y_round):
        dot = np.multiply(dot_scaled, (1 / scale_coeff_x, 1 / scale_coeff_y))
        if (x_max_min[0] <= dot[0] <= x_max_min[1]) and (y_max_min[0] <= dot[1] <= y_max_min[1]):
            field[dot_scaled[0] + x_res // 2, dot_scaled[1] + y_res // 2] += 1
    return field
    # try:
    #     field[dot[0], dot[1], dot[2]] += 1
    # except IndexError:
    #     pass

def field_from_crossings_2D(crossings_x, crossings_y, x_res, y_res, x_max_min=(-1, 1), y_max_min=(-1, 1)):
    grid_xy = fg.create_mesh_XY(xMinMax=x_max_min, yMinMax=y_max_min, xRes=x_res, yRes=y_res)
    xAr_, yAr_ = fg.arrays_from_mesh(grid_xy)
    scale_coeff_x = 1 / (xAr_[1] - xAr_[0])
    scale_coeff_y = 1 / (yAr_[1] - yAr_[0])
    crossings_scaled_x = crossings_x * scale_coeff_x
    crossings_scaled_y = crossings_y * scale_coeff_y
    crossings_scaled_x_round = np.rint(crossings_scaled_x).astype(int)  # !!!!!!!!!!!!!!!!
    crossings_scaled_y_round = np.rint(crossings_scaled_y).astype(int)
    # crossings_scaled = np.multiply(crossings_scaled_round, 1 / scale_coeff_xyz)
    field = np.zeros((x_res, y_res))
    for dot_scaled in zip(crossings_scaled_x_round, crossings_scaled_y_round):
        dot = np.multiply(dot_scaled, (1 / scale_coeff_x, 1 / scale_coeff_y))
        if (x_max_min[0] <= dot[0] <= x_max_min[1]) and (y_max_min[0] <= dot[1] <= y_max_min[1]):
            field[dot_scaled[0] + x_res // 2, dot_scaled[1] + y_res // 2] += 1
    return field


def lines_dots(positions):
    """
    Line : (x-x1)/(x2-x1) = (y-y1)/(y2-y1) = (z-z1)/(z2-z1)
    Parametric :
        x = x1 + (x2 - x1) * a
        y = y1 + (y2 - y1) * a
        z = z1 + (z2 - z1) * a
    :param positions:
    :return:
    """
    for dots in positions:
        dot1 = dots[0]
        for dot2 in dots[1:]:
            pass
            dot1 = dot2
    exit()
    """
    Function finds the crossing of the line (in between dot1 and dot2) and the plane.
    Line : (x-x1)/(x2-x1) = (y-y1)/(y2-y1) = (z-z1)/(z2-z1)

    Plane: Ax + By + Cz + D = 0
    |x-x1  y-y1  z-z1 |
    |x2-x1 y2-y1 z2-z1| = 0
    |x3-x1 y3-y1 z3-z1|
    Example:
    (-5, -5, 0), (-5, 5, 0), (5, 5, 0)
    z = 0
    (https://ru.onlinemschool.com/math/assistance/cartesian_coordinate/plane/)

    Crossing: https://matworld.ru/analytic-geometry/tochka-peresechenija-prjamoj-i-ploskosti.php
    (look at equations, ignore the text)
    :param dot1: (x1, y1, z1)
    :param dot2: (x2, y2, z2)
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0
    :param check_segment: if True, checking if the don is on the segment in between dot1 and dot2
    :return: dot: (x, y ,z) or None if there is no crossing
    """
    x1, y1, z1 = dot1
    x2, y2, z2 = dot2
    m1 = x2 - x1
    p1 = y2 - y1
    l1 = z2 - z1
    # z = D plane
    A, B, C, D = plane
    matrix = [[p1, -m1, 0], [0, l1, -p1], [A, B, C]]
    vector = [[p1 * x1 - m1 * y1], [l1 * y1 - p1 * z1], [-D]]
    try:
        dot_cross = np.linalg.solve(matrix, vector)
        if not np.allclose(np.dot(matrix, dot_cross), vector):
            print(f'WRONG CROSSINGS')
        dot_ans = np.array([dot_cross[0, 0], dot_cross[1, 0], dot_cross[2, 0]])
    except np.linalg.LinAlgError:
        dot_cross = None

    if check_segment:
        check = dot_on_segment(dot_cross, dot1, dot2)
        if check:
            return dot_ans
        else:
            return None

    return dot_ans

if __name__ == '__main__':
    # pv_integrating_sphere()
    # scene = main_create_scene_test()
    scene = pv_scene_real()
    positions = cs.scene_render_and_positions(scene, rays_number=10, show_3d=False)
    lines_dots(positions)
    exit()
    # positions = cs.scene_render_and_positions(scene, rays_number=50, show_3d=True)
    # time.sleep(10)
    # exit()
    x_res, y_res = 201, 201
    xM = -2, 2
    yM = -2, 2
    # grid_xyz = fg.create_mesh_XYZ(xMax=3, yMax=3, zMax=2, xRes=x_res, yRes=y_res, zRes=z_res, xMin=-3, yMin=-3, zMin=-2)
    # xAr_, yAr_, zAr_ = fg.arrays_from_mesh(grid_xyz)
    # scale_coeff_xyz = np.array([1 / (xAr_[1] - xAr_[0]), 1 / (yAr_[1] - yAr_[0]), 1 / (zAr_[1] - zAr_[0])])
    flux_xy = False
    cross_sections = True
    if flux_xy:
        full_field = []
        for z in np.linspace(-2, +2, 50):
            plane = (0, 0, 1, z)
            crossings = cs.crossings_plane_rays(positions, plane)
            field = field_from_crossings_2D(
                crossings[:, 0], crossings[:, 1],
                x_res=x_res, y_res=y_res, x_max_min=xM, y_max_min=yM
            )
            full_field.append(field)
        full_field = np.array(full_field)
        plt.imshow(full_field[:, :, y_res//2], cmap='nipy_spectral', interpolation='bilinear',
                   extent=[xM[0], xM[1], yM[0], yM[1]])
        plt.tight_layout()
        plt.show()
    if cross_sections:
        scatter = False
        intensity = True
        for plane_xyz in ['xy', 'zx', 'yz']:

            if plane_xyz == 'xy':
                plane = (0, 0, 1, 0)
            elif plane_xyz == 'zx':
                plane = (0, 1, 0, 0)
            elif plane_xyz == 'yz':
                plane = (1, 0, 0, 0)
            else:
                plane = (0, 0, 0, 0)

            crossings = cs.crossings_plane_rays(positions, plane)
            # crossings_scaled = np.multiply(crossings, scale_coeff_xyz)
            # crossings_scaled_round = np.rint(crossings_scaled).astype(int)  ## centers
            # crossings_scaled = np.multiply(crossings_scaled_round, 1 / scale_coeff_xyz)
            # crossings = crossings_scaled
            # field = np.zeros((x_res, y_res, z_res))
            # for dot in crossings_scaled_round:
            #     dot += [x_res // 2, y_res // 2, z_res // 2]
            #     if dot[0] >= 0 and dot[1] >= 0:
            #         try:
            #             field[dot[0], dot[1], dot[2]] += 1
            #         except IndexError:
            #             pass
            # print(field)
            # print(positions_scaled_round)
            # exit()
            if plane_xyz == 'xy':
                if intensity:
                    field = field_from_crossings_2D(
                        crossings[:, 0], crossings[:, 1],
                        x_res=x_res, y_res=y_res, x_max_min=xM, y_max_min=yM
                    )
                    plt.imshow(field[:, :].T, cmap='nipy_spectral', interpolation='bilinear',
                               extent=[xM[0], xM[1], yM[0], yM[1]])
                    plt.tight_layout()
                    plt.show()
                if scatter:
                    plt.scatter(crossings[:, 0], crossings[:, 1])
            elif plane_xyz == 'zx':
                if intensity:
                    field = field_from_crossings_2D(crossings[:, 2], crossings[:, 0],
                                                    x_res=x_res, y_res=y_res, x_max_min=xM, y_max_min=yM)
                    plt.imshow(field[:, :].T, cmap='nipy_spectral', interpolation='bilinear',
                               extent=[xM[0], xM[1], yM[0], yM[1]])
                    plt.tight_layout()
                    plt.show()
                if scatter:
                    plt.scatter(crossings[:, 2], crossings[:, 0])
            elif plane_xyz == 'yz':
                if intensity:
                    field = field_from_crossings_2D(crossings[:, 1], crossings[:, 2],
                                                    x_res=x_res, y_res=y_res, x_max_min=xM, y_max_min=yM)
                    plt.imshow(field[:, ::-1].T, cmap='nipy_spectral', interpolation='bilinear',
                               extent=[xM[0], xM[1], yM[0], yM[1]])
                    plt.tight_layout()
                    plt.show()
                if scatter:
                    plt.scatter(crossings[:, 1], crossings[:, 2])

            cs.plot_scene_2D(scene, plane)
            if scatter:
                ax = plt.subplot()
                ax.set_aspect(1)

                if plane_xyz == 'xy':
                    plt.title(f'XY, Z={-plane[3]}')
                    plt.xlabel('x')
                    plt.ylabel('y')
                elif plane_xyz == 'zx':
                    plt.title(f'ZX, y={-plane[3]}')
                    plt.xlabel('z')
                    plt.ylabel('x')
                elif plane_xyz == 'yz':
                    plt.title(f'YZ, X={-plane[3]}')
                    plt.xlabel('y')
                    plt.ylabel('z')
                ax = plt.subplot()
                ax.set_aspect(1)
                plt.xlim(-3, 3)
                plt.ylim(-2, 2)
                plt.tight_layout()
                plt.show()

