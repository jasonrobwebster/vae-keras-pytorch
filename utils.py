import numpy as np
import matplotlib.pyplot as plt

def make_grid(images, pad=3, n=7):
    """Make a square array of images from the mnist dataset.

    Params
    ------

    images: array (N, H, W)
        Images to plot

    pad: int
        Number of pixels to pad around the image

    n: int
        Number of images to plot for the nxn grid

    Returns
    -------

    grid: array
        The calculated grid of images
    """
    
    images = images[:n**2]
    _, w, h = images.shape

    x = np.zeros([(w+pad)*n + pad, (h+pad)*n + pad])

    for i in range(n):
        for j in range(n):
            x[i*w + (i+1)*pad : (i+1)*w + (i+1)*pad, j*h + (j+1)*pad : (j+1)*h + (j+1)*pad] = images[i*n+j]

    return x

def show_grid(images, pad=3, n=7):
    grid = make_grid(images, pad, n)
    plt.imshow(grid, cmap='Greys_r')
    ax = plt.gca()
    ax.set_axis_off()
    plt.show()