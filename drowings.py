import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



if __name__ == '__main__':
    cmap = 'nipy_spectral'
    cmap = 'gist_stern'
    cmap = 'plasma'
    cmap = 'gnuplot'
    dots_3d = np.load('dots3D_test.npy')
    dots_3d = gaussian_filter(dots_3d, sigma=2)
    max_int = dots_3d.max()
    plt.imshow(dots_3d[:, :, 191 // 2].T, cmap=cmap, interpolation='spline36',
               vmin=0, vmax=max_int)
    plt.show()
    plt.imshow(dots_3d[:, :, 0].T, cmap=cmap, interpolation='spline36',
               vmin=0, vmax=max_int)
    plt.show()
    plt.imshow(dots_3d[:, :, -1].T, cmap=cmap, interpolation='spline36',
               vmin=0, vmax=max_int)
    plt.show()
    # exit()
    plt.imshow(dots_3d[:, 191 // 2, ::-1].T, cmap=cmap, interpolation='spline36',
               vmin=0, vmax=max_int)
    plt.show()
    plt.imshow(dots_3d[:, 5 * 191 // 8, ::-1].T, cmap=cmap, interpolation='spline36',
               vmin=0, vmax=max_int)
    plt.show()
    plt.imshow(dots_3d[:, 6 * 191 // 8, ::-1].T, cmap=cmap, interpolation='spline36',
               vmin=0, vmax=max_int)
    plt.show()
    plt.imshow(dots_3d[:, 7 * 191 // 8, ::-1].T, cmap=cmap, interpolation='spline36',
               vmin=0, vmax=max_int)
    plt.show()

    exit()
    plt.imshow(gaussian_filter(dots_3d[:, 191 // 2, :].T, sigma=0), cmap='hot', interpolation='spline36')
    plt.show()
    plt.imshow(gaussian_filter(dots_3d[:, 191 // 2, :].T, sigma=1), cmap='hot', interpolation='spline36')
    plt.show()
    dots_3d = gaussian_filter(dots_3d, sigma=1)
    plt.imshow(dots_3d[:, 191 // 2, ::-1].T, cmap='hot', interpolation='spline36')
    plt.show()

