"""Generate an animation of a planet
"""
import hydra
import torch
import random
import numpy as np

from neural_planets.model import PlanetMLP
from neural_planets.utils import spherical_to_cartesian_coords
from neural_planets.animation import (
    generate_animation,
    rotate_frame,
    latent_interp_frame,
    planet_interp_frame
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rotation_frame_fn(phi, theta, config):
    """Partially evaluates the rotation frame function
    """
    xyz = spherical_to_cartesian_coords(phi, theta)
    grid = torch.from_numpy(np.stack(xyz, axis=-1)).float()
    height_field = PlanetMLP(**config.model)(grid, scales=config.noise_levels)

    return lambda fig: rotate_frame(fig, xyz, height_field, config)


def latent_interpolation_frame_fn(phi, theta, config):
    """Partially evaluates the latent interpolation frame function
    """
    xyz = spherical_to_cartesian_coords(phi, theta)
    grid = torch.from_numpy(np.stack(xyz, axis=-1)).float()
    z1 = torch.randn(1, config.model.zdim).repeat(*grid.shape[:2], 1)
    z2 = torch.randn(1, config.model.zdim).repeat(*grid.shape[:2], 1)
    planet = PlanetMLP(**config.model)

    return lambda fig: latent_interp_frame(fig, xyz, z1, z2, planet, config)


def planet_interpolation_frame_fn(phi, theta, config):
    """Partially evaluates the planet interpolation frame function
    """
    xyz = spherical_to_cartesian_coords(phi, theta)
    planet1 = PlanetMLP(**config.model)
    planet2 = PlanetMLP(**config.other.model)

    return lambda fig: planet_interp_frame(fig, xyz, planet1, planet2, config)
    

@hydra.main(config_path='../config', config_name='config')
def main(config):

    # set random seed for reproducibility
    set_seed(config.random_seed)

    # define the domain
    theta = np.linspace(0, 2*np.pi, config.map_resolution)   
    phi = np.linspace(0, np.pi, config.map_resolution)       
    phi, theta = np.meshgrid(phi, theta)  

    # animate
    animation_fn = {
        "rotation": rotation_frame_fn,
        "latent_interpolation": latent_interpolation_frame_fn,
        "planet_interpolation": planet_interpolation_frame_fn
    }[config.animation_type](phi, theta, config)
    generate_animation(animation_fn, config)


if __name__ == '__main__':
    main()