import os
import torch
import numpy as np
import moviepy.editor as mpy

from mayavi import mlab
from neural_planets.utils import planet_encoding

# ---------------- MAYAVI SPECIFICS ----------------
mlab.options.offscreen = True
# --------------------------------------------------


def generate_animation(frame_fn, config):
    """Create GIF animation from arbitrary frame function
    """
    mlab.clf()
    fig = mlab.figure(size=tuple(config.img_size), bgcolor=tuple(config.bg_color))
    step_fn = frame_fn(fig)
    animation = mpy.VideoClip(step_fn, duration=config.duration)

    os.makedirs("videos", exist_ok=True)
    enc = planet_encoding(config)
    animation.write_gif(
        f"videos/{enc}_{config.animation_type.upper()}.gif", 
        fps=config.frames_per_second
    )


def rotate_frame(fig, xyz, height_field, config):
    """Generate frame with planet rotated
    """
    _ = mlab.mesh(
        *[height_field * c for c in xyz],
        scalars=height_field, 
        colormap=config.colormap, 
        figure=fig
    )

    def make_frame(t):
        mlab.view(azimuth=360*t/config.duration, distance=config.view_distance)
        return mlab.screenshot(antialiased=True)

    return make_frame


def latent_interp_frame(fig, xyz, latent1, latent2, planet, config):
    """Generate frame with planet interpolated along a line in latent space
    """
    grid = torch.from_numpy(np.stack(xyz, axis=-1)).float()
    hfield = planet(grid, latent1, scales=config.noise_levels).numpy()
    s = mlab.mesh(
        *[hfield*c for c in xyz], 
        scalars=hfield, 
        colormap=config.colormap, 
        figure=fig
    )

    def make_frame(t):
        step = t / config.duration
        mix = np.sin(np.pi * step)
        latent = (1 - mix) * latent1 + mix * latent2
        hfield = planet(grid, latent, scales=config.noise_levels).numpy()
        s.mlab_source.set(x=hfield*xyz[0])
        s.mlab_source.set(y=hfield*xyz[1])
        s.mlab_source.set(z=hfield*xyz[2])
        s.mlab_source.set(scalars=hfield)
        mlab.view(azimuth=360*step, distance=config.view_distance) 
        return mlab.screenshot(antialiased=True)

    return make_frame


def planet_interp_frame(fig, xyz, planet1, planet2, config):
    """Generate frame with planet interpolated with another planet
    """
    grid = torch.from_numpy(np.stack(xyz, axis=-1)).float()
    hfield1 = planet1(grid, scales=config.noise_levels).numpy()
    hfield2 = planet2(grid, scales=config.other.noise_levels).numpy()
    s = mlab.mesh(
        *[hfield1*c for c in xyz], 
        scalars=hfield1, 
        colormap=config.colormap, 
        figure=fig
    )

    def make_frame(t):
        step = t / config.duration
        mix = np.sin(np.pi * step)
        hfield = hfield1 * (1 - mix) + hfield2 * mix
        s.mlab_source.set(x=hfield*xyz[0])
        s.mlab_source.set(y=hfield*xyz[1])
        s.mlab_source.set(z=hfield*xyz[2])
        s.mlab_source.set(scalars=hfield)
        mlab.view(azimuth=360*step, distance=config.view_distance) # camera angle
        return mlab.screenshot(antialiased=True)

    return make_frame
