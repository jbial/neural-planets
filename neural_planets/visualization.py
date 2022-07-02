import os
import numpy as np
import matplotlib.pyplot as plt

from mayavi import mlab
from neural_planets.utils import planet_encoding

# ---------------- MAYAVI SPECIFICS ----------------
mlab.options.offscreen = True
# --------------------------------------------------


def visualize_projection(height_field, config):
    """Visualize the projection of the sphere surface onto the (phi, theta) plane
    """
    plt.figure(figsize=config.fig_size)
    plt.xlabel("$\phi$")
    plt.ylabel(r"$\theta$")
    plt.xticks(np.linspace(0, height_field.shape[1]-1, 5), [0, '$\pi$/4', '$\pi/2$', '$3\pi/4$', '$\pi$'])
    plt.yticks(np.linspace(0, height_field.shape[0]-1, 5), ['2$\pi$', '$3\pi/2$', '$\pi$', '$\pi/2$', 0])
    plt.imshow(height_field, cmap=config.colormap)
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    enc = planet_encoding(config)
    os.makedirs('images', exist_ok=True)
    plt.savefig(f"images/{enc}_PROJ.{config.save_format}", dpi=config.dpi)


def visualize_sphere(x, y, z, height_field, config):
    """Visualize height map on a sphere using the mayavi renderer
    """
    mlab.clf()
    fig = mlab.figure(size=tuple(config.img_size), bgcolor=tuple(config.bg_color))
    mlab.mesh(x, y, z, scalars=height_field, colormap=config.colormap, figure=fig)

    enc = planet_encoding(config)
    os.makedirs('images', exist_ok=True)
    mlab.savefig(f"images/{enc}_RENDER.{config.save_format}", size=config.img_size, figure=fig)