import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pickle
import collections


def plane_intensity(positions, plane_vec=(3, 2, 1), plane_dot=(1, 2, 3), x_res=21, y_res=21,
                    x_max_min=(-1, 1), y_max_min=(-1, 1), positive=True, negative=True):
    """

    :param positions:
    :param plane_vec: a, b, c in ax+by+cz+d=0
    :param plane_dot: x0,y0,z0 in d = -(ax0+by0+cz0)
    :param x_res:
    :param y_res:
    :param x_max_min:
    :param y_max_min:
    :return:
    """

    def is_plane_between_points(dot1, dot2, plane):
        distance1 = np.dot(dot1, plane[:3]) + plane[3]
        distance2 = np.dot(dot2, plane[:3]) + plane[3]
        if positive and negative:
            return (distance1 > 0 > distance2) or (distance1 < 0 < distance2)
        elif negative:
            return (distance1 > 0 > distance2)
        elif positive:
            return (distance1 < 0 < distance2)
        return False

    def intersection_point_with_plane(dot1, dot2, plane):
        # print(dot1, dot2, plane)
        line_direction = dot2 - dot1
        to_plane = np.dot(line_direction, plane[0:3])
        if to_plane:
            t = (-(plane[0] * dot1[0] + plane[1] * dot1[1] + plane[2] * dot1[2] + plane[3])
                 / to_plane)
            if not is_plane_between_points(dot1, dot2, plane):
                return None
            intersection_point = dot1 + t * line_direction
            return intersection_point
        else:
            return None

    plane = np.array(list(plane_vec) + [-np.dot(plane_vec, plane_dot)])
    dots_all_rays = []
    for dots in positions:
        # print(dots)
        dot1 = dots[0]
        for dot2 in dots[1:]:
            intersect = intersection_point_with_plane(dot1, dot2, plane)
            if intersect is not None:
                dots_all_rays.append(intersect)
            dot1 = dot2
    # return np.array(dots_all_rays)
    dots_all_rays = np.array(dots_all_rays)
    delta_x = (x_max_min[1] - x_max_min[0]) / (x_res - 1)
    delta_y = (y_max_min[1] - y_max_min[0]) / (y_res - 1)
    scale_coeff_x = 1 / delta_x
    scale_coeff_y = 1 / delta_y
    dots_scaled = dots_all_rays * [scale_coeff_x, scale_coeff_y, 1]
    # print(dots_scaled)
    dots_round = np.rint(dots_scaled).astype(int)
    dots_tuples = [tuple(point) for point in dots_round]
    #  / [scale_coeff_x, scale_coeff_y, 1]
    dots_ind_collection = collections.Counter(dots_tuples)
    x_ind = np.arange(int(x_max_min[0] / delta_x), int(x_max_min[1] / delta_x))
    y_ind = np.arange(int(y_max_min[0] / delta_y), int(y_max_min[1] / delta_y))
    intensity = np.zeros((x_res, y_res))
    print(dots_ind_collection)
    z = np.rint(plane_dot[2]).astype(int)
    for ind_i, i in enumerate(x_ind):
        for ind_j, j in enumerate(y_ind):
            # print((i, j, int(plane_dot[2])))
            if (i, j, z) in dict(dots_ind_collection):
                # print((i, j, z))
                intensity[ind_i, ind_j] = dots_ind_collection[(i, j, z)]
    # mesh = np.meshgrid(np.arange(x_res), np.arange(y_res), indexing='ij')
    # print(intensity)
    return dots_all_rays, intensity

    #         dot1 = dot2
    # dots = np.concatenate(np.array(dots_all_rays, dtype=object), axis=0)


if __name__ == '__main__':
    cmap = 'nipy_spectral'
    cmap = 'gist_stern'
    cmap = 'plasma'
    cmap = 'gnuplot'
    cmap = 'turbo'
    Z7 = False
    if Z7:
        planes = False
        if planes:
            # with open('Z7_positions_150.pkl', 'rb') as file:
            with open('..\\holders__5_2__3_8\\Z7_positions_300000.pkl', 'rb') as file:
                positions = pickle.load(file)

            dots, intensity = plane_intensity(positions, plane_vec=(0, 0, -1), plane_dot=(0, 0, 3.8001),
                                              # dots, intensity = plane_intensity(positions, plane_vec=(0, 0, -1), plane_dot=(0, 0, -0.5001),
                                              x_res=51, y_res=51,
                                              x_max_min=(-2.6, 2.6), y_max_min=(-2.6, 2.6),
                                              negative=True,
                                              positive=False,
                                              # negative=False,
                                              # positive=True,
                                              )
            plt.imshow(intensity.T, cmap='hot')  # , interpolation='bilinear')
            print(np.sum(intensity))
            plt.show()
        intensity_dots = True
        if intensity_dots:
            # dots_3d = np.load('Z7_221_150.npy')
            dots_3d = np.load('..\\holders__5_2__3_8\\Z7_221_300000.npy')
            reso = 221
            dots_3d = gaussian_filter(dots_3d, sigma=2)
            dots_3d = np.sqrt(dots_3d)
            max_int = dots_3d.max()
            plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(dots_3d[:, :, -1].T, cmap=cmap, interpolation='spline36',
                       vmin=0, vmax=max_int)
            plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
            plt.show()
            plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(dots_3d[:, :, reso // 2].T, cmap=cmap, interpolation='spline36',
                       vmin=0, vmax=max_int)
            plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
            plt.show()
            plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(dots_3d[:, :, 0].T, cmap=cmap, interpolation='spline36',
                       vmin=0, vmax=max_int)
            plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
            plt.show()
            # exit()
            plt.figure(figsize=(5, 5), dpi=100)
            plt.imshow(dots_3d[:, 4 * reso // 8, ::-1].T, cmap=cmap, interpolation='spline36',
                       vmin=0, vmax=max_int)
            plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
            plt.show()
            exit()
    RZ11 = True
    if RZ11:
        planes = False
        total = 200000
        # total = 15000
        r = 7.5
        focus = 7.6
        r_foc = 1.3
        dist = 11.4
        name = '15n14even'
        # max_int = 15
        if planes:
            # with open('Z7_positions_150.pkl', 'rb') as file:
            # with open('..\\holders__5_2__3_8\\R'
            #           f'Z11_positions_{total}_r{r}_{name}.pkl', 'rb') as file:
            with open('..\\holders__5_2__3_8\\RZ11_2_dist_7.1_foc_4.6_rfoc_0.0_221_3000_r2.5_15n14even.pkl', 'rb') as file:

                positions = pickle.load(file)

            # dots, intensity = plane_intensity(positions, plane_vec=(0, 0, -1), plane_dot=(0, 0, 3.8001),
            dots, intensity = plane_intensity(positions, plane_vec=(0, 0, -1), plane_dot=(0, 0, -0.5001),
                                              x_res=51, y_res=51,
                                              x_max_min=(-2.6, 2.6), y_max_min=(-2.6, 2.6),
                                              # negative=True,
                                              # positive=False,
                                              negative=False,  # transmission
                                              positive=True,
                                              )
            plt.imshow(intensity.T, cmap='hot')  # , interpolation='bilinear')
            print(f'Transmission: {np.sum(intensity) / total}')
            plt.show()
            dots, intensity2 = plane_intensity(positions, plane_vec=(0, 0, -1), plane_dot=(0, 0, 3.8001),
                                               x_res=51, y_res=51,
                                               x_max_min=(-2.6, 2.6), y_max_min=(-2.6, 2.6),
                                               negative=True,  # reflection
                                               positive=False,
                                               # negative=False,  # transmission
                                               # positive=True,
                                               )
            print(f'Reflection: {np.sum(intensity2) / total}')
        intensity_dots = True
        if intensity_dots:
            # dots_3d = np.load('Z7_221_150.npy')
            # dots_3d = np.load(f'..\\holders__5_2__3_8\\RZ13_dist_{dist}_foc_{focus}_rfoc_{r_foc}_221_{total}_r{r}_{name}.npy')
            # dots_3d = np.load(f'..\\holders__5_2__3_8\\Z7_2_dist_7.1_foc_4.6_rfoc_0.0_221_2000000_r2.5_15n14even.npy')
            # dots_3d = np.load(f'..\\holders__5_2__3_8\\abs_2_dist_7.1_foc_4.6_rfoc_0.0_221_4000000_r2.5_15n14even.npy')
            # dots_3d = np.load(f'..\\holders__5_2__3_8\\RZ11_2_dist_7.1_foc_4.6_rfoc_0.0_221_4000000_r2.5_15n14even.npy')
            # dots_3d = np.load(f'..\\holders__5_2__3_8\\ RZ13_2_dist_7.1_foc_4.6_rfoc_0.0_221_4000000_r2.5_15n14even.npy')
            # Load and process the 3D intensity data
            dots_3d = np.load(r'..\holders__5_2__3_8\ RZ13_2_dist_7.1_foc_4.6_rfoc_0.0_221_40000_r2.5_15n14even.npy')
            reso = 221
            dots_3d = gaussian_filter(dots_3d, sigma=1)

            # Define real-world dimensions
            x_min, x_max = -2.2, 2.2  # mm (original full x range)
            y_min, y_max = -2.2, 2.2  # mm (original full y range)
            z_min, z_max = 0, 3.0  # mm (original full z range)

            # Target crop dimensions
            target_x_min, target_x_max = -2.1, 2.1  # mm
            target_y_min, target_y_max = -2.1, 2.1  # mm
            target_z_min, target_z_max = 0, 2.5  # mm

            # Calculate step sizes for each dimension
            x_step = (x_max - x_min) / reso
            y_step = (y_max - y_min) / reso
            z_step = (z_max - z_min) / dots_3d.shape[0]

            # Convert real-world target coordinates to array indices
            x_start_idx = int((target_x_min - x_min) / x_step)
            x_end_idx = int((target_x_max - x_min) / x_step)
            y_start_idx = int((target_y_min - y_min) / y_step)
            y_end_idx = int((target_y_max - y_min) / y_step)
            z_start_idx = int((target_z_min - z_min) / z_step)
            z_end_idx = int((target_z_max - z_min) / z_step)

            # Crop the 3D array
            dots_3d_cropped = dots_3d[z_start_idx:z_end_idx, y_start_idx:y_end_idx, x_start_idx:x_end_idx]
            dots_3d_cropped_swapped = np.transpose(dots_3d_cropped, (2, 1, 0))
            dots_2D = dots_3d_cropped_swapped[:, dots_3d_cropped.shape[2] // 2, :]
            max_int = dots_3d.max()
            # Verify the shape of the cropped data
            print("Original shape:", dots_3d.shape)
            print("Cropped shape:", dots_3d_cropped.shape)

            # Plot the XZ cross-section
            plt.figure(figsize=(7, 5), dpi=200)
            im = plt.imshow(
                dots_2D,
                cmap=cmap,
                interpolation='spline36',
                vmin=0,
                vmax=max_int,
                extent=(x_min, x_max, z_min, z_max)
            )

            # Improved colorbar with larger font for min and max labels
            cbar = plt.colorbar(im, fraction=0.031, pad=0.01)
            cbar.set_ticks([0, max_int])  # Show only min and max values
            # cbar.set_ticklabels(['Min', 'Max'])  # Label as "Min" and "Max"
            cbar.ax.tick_params(labelsize=14)  # Increase font size for Min/Max labels
            # cbar.set_label('Intensity', fontsize=14)

            # Enhanced title and labels
            # plt.title('XZ Cross-Section of Intensity Distribution at Y = 0 mm', fontsize=16)
            plt.xlabel('X (mm)', fontsize=16)
            plt.ylabel('Z (mm)', fontsize=16)

            # Increase font size for axis tick numbers
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.yticks([0, 1, 2], fontsize=14)
            plt.xticks([-2, -1, 0, 1, 2], fontsize=14)
            # Set axis limits, aspect ratio, and reverse z-axis
            plt.xlim(-2.1, 2.1)
            plt.ylim(0, 2.75)
            # plt.ylim(0, 0.1)
            plt.gca().set_aspect('equal')
            plt.gca().invert_yaxis()
            plt.tight_layout(pad=0.2)
            plt.show()

            # Create a circular mask for the cylinder
            x = np.linspace(target_x_min, target_x_max, dots_3d_cropped_swapped.shape[0])
            y = np.linspace(target_y_min, target_y_max, dots_3d_cropped_swapped.shape[1])
            xx, yy = np.meshgrid(x, y)
            radius = (target_x_max - target_x_min) / 2  # Radius of the cylinder
            circular_mask = (xx ** 2 + yy ** 2) <= radius ** 2

            # Apply the mask to each z-slice and calculate energy
            z_sums = []
            for z_slice in range(dots_3d_cropped_swapped.shape[2]):
                masked_slice = dots_3d_cropped_swapped[:, :, z_slice] * circular_mask
                z_sums.append(np.sum(masked_slice))

            z_sums = np.array(z_sums)

            # Normalize to get cumulative energy percentage
            cumulative_energy = np.cumsum(z_sums) / np.sum(z_sums) * 100

            # Define the z-axis range (real-world dimensions)
            z_range = np.linspace(target_z_min, target_z_max, dots_3d_cropped_swapped.shape[2])

            # Plot cumulative energy distribution along the z-axis
            plt.figure(figsize=(8, 6), dpi=150)
            plt.plot(z_range, cumulative_energy, label='Cumulative Energy (%)', color='blue')
            plt.axhline(100, color='gray', linestyle='--', linewidth=0.8, label='Maximum Energy')
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)

            # Add labels and title
            plt.xlabel('Z (mm)', fontsize=14)
            plt.ylabel('Cumulative Energy (%)', fontsize=14)
            plt.title('Cumulative Energy Distribution Along Z-axis (Cylinder)', fontsize=16)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            exit()
