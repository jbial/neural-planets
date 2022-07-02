# Neural Planets
----

Neural planets is a repository for rendering planet-like spheres with topography generated from random neural networks. The design-space is intentionally crafted
to be large. With a mixture of 12 continuous and discrete design parameters (not including neural network weight parameters which can be arbitrarily large), the variety of planets is endless.

### Usage
Usage of this software requires Python 3.6 or higher, pip, and access to a command line. To setup, run `pip install -r requirements.txt` to install the necessary python modules, and then run `pip install -e .`.

#### 0. Parameters
This software utilizes [hydra-core](https://hydra.cc/docs/intro/) for handling parameter settings. For a reference of every mutable parameter and it's description, refer to the YAML files in the `config` directory.

#### 1. Generating planets
Use the `scripts/generate_world.py` file to generate a rendering of a planet's 2D topographic projection and it's rendering onto a sphere. For example:
```
python scripts/generate_world.py \
prefix=unique-identifying-prefix \
random_seed=30 \
noise_levels=8 \
model.radius=10 \
model.layers=\[64,64,64\] \
model.activations=\[sigmoid,sin,tanh\] \
model.zdim=16 \ 
model.final_activation=tanh \
model.init_limit=2 \
model.min_delta=0.5 \
model.ffeats=-1 \
model.ffscale=1 \
model.noise_decay=0.35 \
model.noise_scale=2 \
colormap='gist_earth' \
img_size=\[500,500\]
```

### 2. Generating animations
TODO

### 3. World interpolation
TODO

### TODO
- Documentation
- More animations
- Flow maps on the surface
