import os
import torch
import numpy as np
import moviepy.editor as mpy

from mayavi import mlab


def generate_animation(config, *args):
    """Generate specific type of planet animation
    """
    frame_fn = {
        "rotation": lambda fig: rotate_frame(fig, *args, config),
        "latent_interpolation": lambda fig: latent_interp_frame(fig, *args, config),
        "planet_interpolation": lambda fig: planet_interp_frame(fig, *args, config)
    }[config.animation_type]
    _generate_animation(frame_fn, config)


def _generate_animation(frame_fn, config):
    """Create GIF animation from arbitrary frame function
    """
    mlab.clf()
    fig = mlab.figure(size=tuple(config.img_size), bgcolor=config.bg_color)
    step_fn = frame_fn(fig)
    os.makedirs('/'.join(config.animation_path.split('/')[:-1]), exist_ok=True)
    animation = mpy.VideoClip(step_fn, duration=config.duration)
    animation.write_gif(config.animation_path, fps=config.frames_per_second)


def rotate_frame(fig, phi, theta, height_field, config):
    """Generate frame with planet rotated
    """
    _ = mlab.mesh(
        height_field * np.sin(phi) * np.cos(theta),
        height_field * np.sin(phi) * np.sin(theta),
        height_field * np.cos(phi),
        scalars=height_field, 
        colormap=config.color_map, 
        figure=fig
    )

    def make_frame(t):
        mlab.view(azimuth= 360*t/config.duration, distance=config.view_distance) # camera angle
        return mlab.screenshot(antialiased=True)

    return make_frame


def latent_interp_frame(fig, phi, theta, latent1, latent2, planet, config):
    """Generate frame with planet interpolated along a line in latent space
    """
    x_ = np.sin(phi)*np.cos(theta)
    y_ = np.sin(phi)*np.sin(theta)
    z_ = np.cos(phi)
    grid = torch.from_numpy(np.stack([x_, y_, z_], axis=-1)).float()
    hfield = planet(grid, latent1)
    s = mlab.mesh(
        *[hfield*c for c in [x_, y_, z_]], 
        scalars=hfield, 
        colormap=config.colormap, 
        figure=fig
    )

    def make_frame(t):
        step = t / config.duration
        mix = np.sin(np.pi * step)
        hfield = planet(grid, (1 - mix) * latent1 + mix * latent2)
        s.mlab_source.set(x=hfield*x_)
        s.mlab_source.set(y=hfield*y_)
        s.mlab_source.set(z=hfield*z_)
        s.mlab_source.set(scalars=hfield)
        mlab.view(azimuth=360*step, distance=config.view_distance)  # camera angle
        return mlab.screenshot(antialiased=True)

    return make_frame


def planet_interp_frame(fig, phi, theta, planet1, planet2, config):
    """Generate frame with planet interpolated with another planet
    """
    x_ = np.sin(phi)*np.cos(theta)
    y_ = np.sin(phi)*np.sin(theta)
    z_ = np.cos(phi)
    grid = torch.from_numpy(np.stack([x_, y_, z_], axis=-1)).float()
    hfield = planet1(grid)
    s = mlab.mesh(
        *[hfield*c for c in [x_, y_, z_]], 
        scalars=hfield, 
        colormap=config.colormap, 
        figure=fig
    )

    def make_frame(t):
        step = t / config.duration
        mix = np.sin(np.pi * step)
        hfield = planet1(grid) * (1 - mix) + planet2(grid) * mix
        s.mlab_source.set(x=hfield*x_)
        s.mlab_source.set(y=hfield*y_)
        s.mlab_source.set(z=hfield*z_)
        s.mlab_source.set(scalars=hfield)
        mlab.view(azimuth=360*step, distance=config.view_distance) # camera angle
        return mlab.screenshot(antialiased=True)

    return make_frame
