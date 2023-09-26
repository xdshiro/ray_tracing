"""
This module plots the cross-sections of the pvtrace ray-tracing simulations.
We can place different objects and structures.
For now spheres and cylinders are implemented.
XY, XZ, ZY implementer.

# XY
plt.scatter(crossings[:, 0], crossings[:, 1])
plt.xlabel('x')
plt.ylabel('y')
# YZ
plt.scatter(crossings[:, 1], crossings[:, 2])
plt.xlabel('x')
plt.ylabel('z')
# ZX
plt.scatter(crossings[:, 2], crossings[:, 0])
plt.xlabel('z')
plt.ylabel('x')
"""

# Some packages used by pvtrace are a little noisy
import logging

import numpy
import numpy as np
import matplotlib.pyplot as plt
import pvtrace as pv
from numpy.random import Generator, PCG64, SeedSequence
rng = Generator(PCG64(SeedSequence(103)))

logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)


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
def scene_render_and_positions(scene, rays_number=50, random_seed=0, open_browser=True, show_3d=True):
    """
    The main purpose of this function is to propagate rays and return all there trajectories, as
    arrays of all dots they have "visited".
    Function also renders 3D picture if show_3d=True.
    :param scene: pv.Scene(world) with all Nodes (objects and light sources)
    :param random_seed: random seed for the ray-tracing
    :param open_browser: show in the browser
    :param show_3d: 3d rendering
    :return: positions of all the rays
    """
    positions = []
    np.random.seed(random_seed)
    if show_3d:
        vis = pv.MeshcatRenderer(wireframe=True, open_browser=True)
        vis.render(scene)
        # vis = pv.MeshcatRenderer(wireframe=True, open_browser=open_browser)
    for rays in scene.emit(rays_number):
        steps = pv.photon_tracer.follow(scene, rays)
        path, decisions = zip(*steps)
        positions_ray = []
        for ray in path:
            positions_ray.append(ray.position)
        positions.append(np.array(positions_ray))
        if show_3d:
            vis.add_ray_path(path)

    if show_3d:
        # pass
        # vis = pv.MeshcatRenderer(wireframe=True)
        vis.render(scene)
    return positions


def dot_on_segment(dot, dot1, dot2):
    """
    Function checks if the dot is on the line in between dot1 and dot2. dot must be on
    the line.
    :param dot: (x, y, z) the do must be on the line (it's from the crossing_plane_line())
    :param dot1: (x1, y1, z1)
    :param dot2: (x2, y2, z2)
    :return: True if the line is on the segment, otherwise False
    """
    if dot is None:
        return False
    for x, x1, x2 in zip(dot, dot1, dot2):
        if x2 >= x1:
            if x1 <= x <= x2:
                pass
            else:
                return False
        else:
            if x2 <= x <= x1:
                pass
            else:
                return False
    return True


def crossing_plane_line(dot1, dot2, plane=(0, 0, 1, 0), check_segment=True):
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


def crossings_plane_rays(positions, plane):
    """
    Function return all crossings of rays with the plane.
    :param positions: Array [[[x1_r1, x2_r1, x3_r1], [x1_r1, x2_r1, x3_r1],...],
                            [[x1_r2, x2_r2, x3_r2], [x1_r2, x2_r2, x3_r2],...],...]
                            r1 - ray1, r2 - ray2.
                            Arrays from the photon_tracer.follow(scene, ray)
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0
    :return: [[x1, x2, x3], [x1, x2, x3],...] - all crossings
    """
    crossings = []
    for dots in positions:
        dot1 = dots[0]
        for dot2 in dots[1:]:
            dot_cross = crossing_plane_line(dot1, dot2, plane=plane)
            # print(dot_cross)
            if dot_cross is not None:
                crossings.append(dot_cross)
            dot1 = dot2
    return np.array(crossings)


def homogeneous_transform_inverse(pose):
    pose_inverse = np.zeros(np.shape(pose))
    R_AB = pose[:3, :3]
    d_AB = pose[:3, 3]
    pose_inverse[:3, :3] = R_AB.T
    pose_inverse[:3, 3] = -1 * R_AB.T@d_AB
    pose_inverse[3, 3] = 1
    return pose_inverse


def crossing_plane_cylinder(radius, plane, length, pose, circle_res=100):
    """
    This function return the circle of the plane with sphere crossing.
    For 3 different cross-section we are just renaming the axis.
    Parametric equation of the sphere is used. (check wiki: Sphere)

    ##################################################################################
    Function is not finished. Its performance is bad, in numerically finds the closest
    to the plane dots from the surface on mesh, not a real math solution.
    ##################################################################################

    3rd axis is pointing towards us. for XZ cross-section y is pointing to us,
    that means that y axis is X, and x axis is Z.

    This function uses homogeneous transformation matrices:
    http://www.ccs.neu.edu/home/rplatt/cs5335_fall2017/slides/homogeneous_transforms.pdf

    :param location: (x0, y0, z0) center of the sphere
    :param radius: radius of the sphere
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0
    :param circle_res: the number of dots for the circle
    :return: x_array, y_array, if not crossing: None, None
    """
    z_res = 300
    # number_edge_circles = 20
    phi = np.linspace(0, 2 * np.pi, circle_res)
    x = radius * np.cos(phi)
    y = radius * np.sin(phi)
    z = np.linspace(-length / 2, length / 2, z_res)
    x_surf = np.array(list(x) * z_res)
    y_surf = np.array(list(y) * z_res)
    z_surf = np.array([z.T] * circle_res).T.reshape((z_res * circle_res))
    ones = np.ones(circle_res * z_res)
    p_ones = np.array([x_surf, y_surf, z_surf, ones])
    moved = pose @ p_ones
    x = moved[0]
    y = moved[1]
    z = moved[2]

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(x_surf, y_surf, z_surf)
    # ax.plot(x, y, z)
    # plt.show()
    # exit()
    # plt.plot(x, z)
    # ax = plt.subplot()
    # ax.set_aspect(1)
    # plt.show() ми фри пас
    xyz = np.array([x, y, z])
    distances_to_plane = xyz.T@plane[:3] + plane[3]
    indexes_sorted = numpy.argsort(np.abs(distances_to_plane))
    indexes = indexes_sorted[0:circle_res + z_res * 3]
    # indexes = indexes_sorted[0:circle_res + z_res * 40]
    # plt.plot(z[indexes])
    # plt.show()
    # exit()
    # plt.scatter(z[indexes], x[indexes])
    # ax = plt.subplot()
    # ax.set_aspect(1)
    # plt.show()
    # exit()
    if plane[:3] == (0, 0, 1):  # yz-cross-section
        return x[indexes], y[indexes]
    elif plane[:3] == (1, 0, 0):  # xz-cross-section
        return y[indexes], z[indexes]
    elif plane[:3] == (0, 1, 0):  # xy-cross-section
        return z[indexes], x[indexes]
    else:
        print(f'This cross-section is not implemented. Use XY, XZ, or YZ for now')
        return None, None

    # print(x_surf)
    # print(np.shape(x_surf))
    # z = np.linspace(0, 0, circle_res)
    # X, Y, Z = np.meshgrid(x, y ,z)

    # p_ones = np.array([x, y, z, ones])
    # pose_inverse = homogeneous_transform_inverse(pose)
    # print(pose_inverse)
    # print(pose)
    # moved = pose@p_ones
    # # moved = pose@p_ones
    # x = moved[0]
    # y = moved[1]
    # z = moved[2]
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot(x, y, z)
    # plt.show()
    # plt.plot(x, z)
    # ax = plt.subplot()
    # ax.set_aspect(1)
    # plt.show()
    #
    # exit()
    # x = p_ones[0]
    # y = p_ones[1]
    # return x, y


def crossing_plane_sphere(location, radius, plane, circle_res=100):
    """
    This function return the circle of the plane with sphere crossing.
    For 3 different cross-section we are just renaming the axis.
    Parametric equation of the sphere is used. (check wiki: Sphere)

    3rd axis is pointing towards us. for XZ cross-section y is pointing to us,
    that means that y axis is X, and x axis is Z.

    :param location: (x0, y0, z0) center of the sphere
    :param radius: radius of the sphere
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0
    :param circle_res: the number of dots for the circle
    :return: x_array, y_array, if not crossing: None, None
    """
    if plane[:3] == (1, 0, 0):  # yz-cross-section
        z0, x0, y0 = location
    elif plane[:3] == (0, 1, 0):  # xz-cross-section
        y0, z0, x0 = location
    elif plane[:3] == (0, 0, 1):  # xy-cross-section
        x0, y0, z0 = location
    else:
        print(f'This cross-section is not implemented. Use XY, XZ, or YZ for now')
        return None, None
    z = -plane[3]
    if abs((z - z0) / radius) > 1:
        return None, None
    theta = np.arccos((z - z0) / radius)
    phi = np.linspace(0, 2 * np.pi, circle_res)
    x = x0 + radius * np.sin(theta) * np.cos(phi)
    y = y0 + radius * np.sin(theta) * np.sin(phi)
    return x, y


def plot_scene_2D(scene, plane):
    """
    Plotting the cross-section of the scene.
    :param scene: pv scene
    :param plane: (A, B, C, D) where Ax + By + Cz + D = 0 determines the cross-section
    :return: None (but it plot lines, so you need to plt.show() after)
    """
    for node in scene.root.leaves:
        if isinstance(node.geometry, pv.Sphere):
            location = node.location
            radius = node.geometry.radius
            x, y = crossing_plane_sphere(location, radius, plane)
            if x is not None:
                plt.plot(x, y, color='r')
        if isinstance(node.geometry, pv.Cylinder):
            location = node.location
            pose = node.pose
            radius = node.geometry.radius
            length = node.geometry.length
            x, y = crossing_plane_cylinder(radius, plane, length, pose, circle_res=100)
            if x is not None:
                plt.scatter(x, y, color='g')

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
def main_create_scene_test():
    world = pv.Node(
        name="world (air)",
        geometry=pv.Sphere(
            radius=10,
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
    #
    # sphere3.translate((0, -0.2, -0.2))
    class PartialTopSurfaceMirror(pv.FresnelSurfaceDelegate):
        """ A section of the top surface is covered with a perfect mirrrors.

            All other surface obey the Fresnel equations.
        """

        # print(super(PartialTopSurfaceMirror, self).reflected_direction(surface, ray, geometry, container,
        #                                                                adjacent))
        # Xn0 = np.array((0, 0, -1))
        # # Xp0 = perpendicular_vector_3D(Xn0)
        # # Xc0 = np.cross(Xp0, Xn0)
        # # print(Xn0, Xp0, Xc0)
        # # exit()
        # Xp0 = np.array((1, 0, 0))
        # Xc0 = np.array((0, 1, 0))
        # Xn = geometry.normal(ray.position)
        # Xp = perpendicular_vector_3D(Xn)
        # Xc = np.cross(Xp, Xn)
        # dir0, up0, cross0 = Xp0, Xn0, Xc0
        # dir, up, cross = Xp, Xn, Xc
        # transform = [
        #     [np.dot(Xn, Xn0), np.dot(Xp, Xn0), np.dot(Xc, Xn0)],
        #     [np.dot(Xn, Xp0), np.dot(Xp, Xp0), np.dot(Xc, Xp0)],
        #     [np.dot(Xn, Xc0), np.dot(Xp, Xc0), np.dot(Xc, Xc0)]
        # ]
        # R1 = np.array([Xp0, Xn0, Xc0])
        # R2 = np.array([Xp, Xn, Xc])
        # transform2 = np.matmul(R1, R2.T)
        def reflected_direction(self, surface, ray, geometry, container, adjacent):
            """
            Implementation of the scattering surface. We can control the scattering angles in scattered_angles.
            Vector transformation works only for a box rn, need to implement the general case.
            :return: direction of the scattered beam
            """

            # basis surface (bottom, directed upwards)
            # xn0, yn0, zy0 = 0, 0, -1

            def scattered_angles(absorption=0.9):
                """
                Angles onf the scattered from the surface beams. (horizontal plane. Other planes
                are implemented via linear algebra transformations)
                :return: phi, theta of the scattered beam.
                """
                roll_the_dice = np.random.uniform(0, 1, 1)
                if roll_the_dice > absorption:
                    return 0, np.pi
                else:
                    p1, p2 = rng.uniform(0, 1, 2)
                    # theta = np.arcsin(np.sqrt(p1) * np.sin(theta_max))
                    phi = 2 * np.pi * p1
                    # phi = np.pi / 5
                    theta = 0.96 * 0.5 * np.pi * p2

                    return phi, theta

            def perpendicular_vector_3D(X):
                """
                add the real case, rn only for BOXes
                :param X: vector 3D.
                :return: Any perpendicular vector to X
                """
                x, y, z = X
                if x == 0 and y == 0:
                    return 1, 0, 0
                elif x == 0 and z == 0:
                    return 0, 0, 1
                elif y == 0 and z == 0:
                    return 0, 1, 0

            # current surface normal
            # xn, yn, zn = geometry.normal(ray.position)
            # xp, yp, zp = perpendicular_vector_3D(xn, yn, zn)
            # xc, yc, zc = np.cross([xn, yn, zn], [xp, yp, zp])
            u = np.array((0, 0, -1))
            v = perpendicular_vector_3D(u)
            uv = np.cross(u, v)
            # print(Xn0, Xp0, Xc0)
            # exit()
            x = geometry.normal(ray.position)
            y = perpendicular_vector_3D(x)
            # xy = np.array((x[0]*y[2]-x[2]*y[0],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]))
            print(ray.position, x, y)
            xy = np.cross(x, y)
            # print(u, v, uv)
            # print(x, y, xy)
            # transform3 = np.array([
            #     [np.dot(x, u), np.dot(x, v), np.dot(x, uv)],
            #     [np.dot(y, u), np.dot(y, v), np.dot(y, uv)],
            #     [np.dot(xy, u), np.dot(xy, v), np.dot(xy, uv)]
            # ])
            R1 = np.array([u, v, uv])
            R2 = np.array([x, y, xy])
            transform3 = np.matmul(R1.T, R2)
            # print(transform3)
            # exit()
            phi_new, theta_new = scattered_angles(absorption=1)  # 0.9 ABSORBTION OF THE SURFACE
            # print(np.matmul(x, transform3), u)
            # print(np.matmul(u, transform3), x)
            # print(np.matmul(u, transform3.T), x)
            # print(np.matmul(x, transform3.T), u)
            # print(np.matmul(x.T, transform3), u)
            # print(np.matmul(transform3, x), u)
            # print(np.matmul(transform3, u), x)
            # print(np.matmul(transform3.T, x), u)
            # print(np.matmul(transform3.T, u), x)
            # print(np.matmul(transform3, x.T), u)
            # print('a')
            # print(np.matmul(v, transform3), y)
            # print(np.matmul(uv, transform3), xy)
            x = np.sin(theta_new) * np.cos(phi_new)
            y = np.sin(theta_new) * np.sin(phi_new)
            z = np.cos(theta_new)
            # z = np.sin(theta_new) * np.cos(phi_new)
            # y = np.sin(theta_new) * np.sin(phi_new)
            # x = np.cos(theta_new)

            Xnew0 = np.array((x, y, z))

            Xnew = tuple(np.matmul(Xnew0, transform3))
            return Xnew

            # _, phi0, theta0 = fg.spherical_coordinates(xn0, yn0, zy0)
            # _, phin, thetan = fg.spherical_coordinates(xn, yn, zn)
            #
            # phi_delta, theta_delta = phin - phi0, thetan - theta0
            #
            # phi, theta = scattered_angles()
            # phi_new, theta_new = phi + phi_delta, theta + theta_delta
            #
            # x = np.sin(theta_new) * np.cos(phi_new)
            # y = np.sin(theta_new) * np.sin(phi_new)
            # z = np.cos(theta_new)
            # print(x, y, z)
            # coords = tuple(spherical_to_cart(theta_new, phi_new))

            # return x, y, z
            # normal = geometry.normal(ray.position)
            # if np.allclose(normal, TOP_SURFACE):
            #     return 0, 0, -1
            # elif np.allclose(normal, BOT_SURFACE):
            #     return x, y, z
            # elif np.allclose(normal, LEFT_SURFACE):
            #     return 1, 0, 0
            # elif np.allclose(normal, RIGHT_SURFACE):
            #     return -1, 0, 0
            # elif np.allclose(normal, FRONT_SURFACE):
            #     return 0, 1, 0
            # elif np.allclose(normal, BACK_SURFACE):
            #     return 0, -1, 0
            # return normal
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
            # normal = geometry.normal(ray.position)
            # print(self, surface, ray, geometry, container, adjacent)
            # exit()
            # Normal are outward facing

            x, y, z = ray.position[0], ray.position[1], ray.position[2]
            if np.isclose(z, 2.5) and np.abs(x) < 0.3 and np.abs(y) < 0.3:
                return super(PartialTopSurfaceMirror, self).reflectivity(surface, ray, geometry, container,
                                                                         adjacent)
            else:
                # ray.alive = False
                # print(x, y, z)
                # ray = replace(ray, direction=direction, wavelength=wavelength, source=self.name)

                # exit()
                return 1.0
            # If a ray hits the top surface where x > 0 and y > 0 reflection
            # set the reflectivity to 1.
            # if np.allclose(normal, BOT_SURFACE):
            #     x, y = ray.position[0], ray.position[1]
            #     if x > -0.5 and y > -0.5:
            #         return 1
    cylinder1 = pv.Node(
        name="cyl1 (glass)",
        geometry=pv.Cylinder(
            radius=1,
            length=2,
            material=pv.Material(
                refractive_index=1.05,
                components=[
                                pv.Absorber(coefficient=1.1703),
                                pv.Scatterer(coefficient=1.181)
                            ]

            ),
        ),
        parent=world
    )
    # box = pv.Node(
    #     name="cyl1 (glass)",
    #     geometry=pv.Box(
    #         (1.0, 1.0, 2),
    #         material=pv.Material(
    #             refractive_index=1.05,
    #             surface=pv.Surface(delegate=PartialTopSurfaceMirror()),
    #             components=[
    #                 pv.Absorber(coefficient=1.1),
    #                 pv.Scatterer(coefficient=0.181)
    #             ]
    #
    #         ),
    #     ),
    #     parent=world
    # )
    # cylinder1.rotate(np.pi/2, [0, 1, 0])

    # cylinder1.translate((0.5, -0.3, 0.2))
    # cylinder1.rotate(np.pi/5, [1, 0, 0])
    import functools
    light = pv.Node(
        name="Light (555nm)",
        # light=pv.Light(direction=functools.partial(pv.cone, np.pi / 32)),
        light=pv.Light(position=functools.partial(collimated_beam, 0.1),
                       wavelength=lambda: 555
                       ),
        parent=world
    )
    light.translate((0.0, 0.0, 3))
    light.rotate(np.pi, [1, 0, 0])
    scene = pv.Scene(world)
    return scene


if __name__ == '__main__':

    scene = main_create_scene_test()

    # positions = scene_render_and_positions(scene, rays_number=50, show_3d=True)
    positions = scene_render_and_positions(scene, rays_number=500, show_3d=True)
    # ax = plt.figure().add_subplot(projection='3d')
    # for position in positions:
    #     ax.plot(position[:, 0], position[:, 1], position[:, 2])
    # plt.show()
    # exit()
    plane = (0, 1, 0, 0)
    crossings = crossings_plane_rays(positions, plane)
    # plt.scatter(crossings[:, 0], crossings[:, 1])
    # plt.scatter(crossings[:, 1], crossings[:, 2])
    # plt.scatter(crossings[:, 2], crossings[:, 0])
    plot_scene_2D(scene, plane)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    ax = plt.subplot()
    ax.set_aspect(1)
    plt.tight_layout()
    plt.title(f'XY, Z={-plane[3]}')
    # XY
    plt.xlabel('x')
    plt.ylabel('y')
    # YZ
    # plt.xlabel('x')
    # plt.ylabel('z')
    # ZX
    # plt.xlabel('z')
    # plt.ylabel('x')
    plt.show()
