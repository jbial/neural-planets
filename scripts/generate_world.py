import hydra
import torch
import random
import numpy as np

from neural_planets.model import PlanetMLP
from neural_planets.utils import spherical_to_cartesian_coords
from neural_planets.visualization import visualize_projection, visualize_sphere


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(config_path='../config', config_name='config')
def main(config):

    # set random seed for reproducibility
    set_seed(config.random_seed)

    # define the domain
    theta = np.linspace(0, 2*np.pi, config.map_resolution)   
    phi = np.linspace(0, np.pi, config.map_resolution)       
    phi, theta = np.meshgrid(phi, theta)  

    # create the planet model
    planet = PlanetMLP(**config.model)

    # generate the planet topography
    coords = spherical_to_cartesian_coords(phi, theta)
    grid = torch.from_numpy(np.stack(coords, axis=-1)).float()
    height_field = planet(grid, scales=config.noise_levels)

    # visualize
    visualize_projection(height_field, config)
    visualize_sphere(*[height_field * c for c in coords], height_field, config)


if __name__ == '__main__':
    main()