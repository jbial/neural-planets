from platform import architecture
import numpy as np


def spherical_to_cartesian_coords(phi, theta):
    """Converts spherical coordinates (of unit 2-sphere) to cartesian coords
    """
    return np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)


def planet_encoding(config):
    """Generate unqiue string encoding for planet parameters (for I/O)
    """
    hparams = config.model
    model_enc = (
        f"r-{hparams.radius}_z-{hparams.zdim}_i-{hparams.init_limit}_"
        f"a-{'-'.join(f'{a}-{l}' for a, l in zip(hparams.activations, hparams.layers))}-"
        f"{hparams.final_activation}_nsd-{hparams.noise_scale}-{hparams.noise_decay}_"
        f"ls-{config.noise_levels}_md-{hparams.min_delta}_rs-{config.random_seed}_"
        f"{'' if hparams.ffeats < 0 else f'ff-{hparams.ffeats}-{hparams.ffscale}'}"
    )

    return f"{config.prefix}_{model_enc}_cm-{config.colormap}"
