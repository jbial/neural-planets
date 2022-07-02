import torch
import torch.nn as nn


class LinearBlock(nn.Module):

    def __init__(self, din, dout, act='relu', lim=2):
        """
        Args:
            din (int): input dimension for the layer
            dout (int): output dimension for the layer
            act (str): string of pytorch activation function
            lim (float): upper and lower limit of initialization
                for the linear layer weights and biases
        """
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.act = self._get_activation_fn(act)
        
        # important for quality of results
        nn.init.uniform_(self.lin.weight, -lim, lim)
            
    def _get_activation_fn(self, name):
        """Gets the activation function callable
        
        Args:
            name (str): name of the pytorch function
            
        Returns:
            callable: activation function
        """
        try:
            return eval(f"torch.{name}")
        except:
            return eval(f"nn.functional.{name}")
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x (torch.Tensor): layer input of shape (*, din)
            
        Returns:
            torch.Tensor: layer output of shape (*, dout)
        """
        return self.act(self.lin(x))
    

class PlanetMLP(nn.Module):
    
    def __init__(
        self, 
        in_dim, 
        layers,
        radius=10,
        activations=None, 
        zdim=16, 
        ffeats=-1,
        ffscale=1,
        final_activation='sigmoid',
        init_limit=5,
        noise_decay=0.25,
        noise_scale=1.5,
        min_delta=0.5,
        spiky=True
    ):
        """
        Args:
            in_dim (int): input dimension of data
            layers (Iterable[int]): list of the number of nodes per hidden layer
            radius (float): radius of the planet
            activations (Iterable[str]): list of activations for each hidden layer
            zdim (int): latent dimension
            ffeats (int): number of fourier features, must be greater than 0 to use
            ffscale (float): scale of fourier feature mapping (higher -> high frequencies)
            scale (float): scale of latent features (zoom)
            final_activation (Union[str, callable]): Final activation to apply to NN output
            init_limit (float): Upper and lower limit of init values for layer weights
        """
        super().__init__()
        self.layers = layers
        self.zdim = zdim
        self.init_limit = init_limit
        self.ffeats = ffeats
        self.in_dim = in_dim
        self.radius = radius
        self.noise_decay = noise_decay
        self.noise_scale = noise_scale
        self.min_delta = min_delta
        self.spiky = spiky

        # check params
        assert init_limit > 0, "init_limit must be > 0"
        assert noise_decay < 1.0, "noise_decay parameter must be < 1"
        assert (activations is None) or (len(activations) == len(layers)), (
            f"# of activation functions {activations} != # of layers {layers}"
        )
        
        # fourier features initialization
        if ffeats > 0:
            self.B = torch.randn(size=(self.in_dim, ffeats)) * ffscale
            self.in_dim = 2 * ffeats
        
        # define hidden layer dimensions
        self.hidden = list(zip(layers, layers[1:]))
        
        # activation functions
        self.acts = activations or ['tanh'] * len(layers)
        self.init_act = self._get_activation_fn(self.acts[0])
        self.final_act = (
            final_activation if hasattr(final_activation, '__call__')
            else (  # if it's not a str or callable, default to identity
                self._get_activation_fn(final_activation) if isinstance(final_activation, str)
                else lambda x: x
            )
        )
            
        # initialize layers
        self.xyzlayer = nn.Linear(self.in_dim, layers[0], bias=False)
        self.latent_layer = nn.Linear(zdim, layers[0])
        self.mlp = self._create_layers()
        
        print(f"\n{'-'*30} [ARCHITECTURE] {'-'*30}\n{self}{'-'*75}")
        
    def _get_activation_fn(self, name):
        """Gets the activation function callable
        
        Args:
            name (str): name of the pytorch function
            
        Returns:
            callable: activation function
        """
        try:
            return eval(f"torch.{name}")
        except:
            return eval(f"nn.functional.{name}")
        
    def _create_layers(self):
        """Initialize the hidden layers of NN
        
        Returns:
            nn.Sequential: NN module for the hidden layers
        """
        layers = [
            LinearBlock(i, o, a, lim=self.init_limit) 
            for (i, o), a in zip(self.hidden, self.acts[1:])
        ]   
        layers.append(nn.Linear(self.layers[-1], 1))
        return nn.Sequential(*layers)

    def _fourier_feature_transform(self, x):
        """Randomly lifts inputs into a fourier basis
        
        Refer to - https://arxiv.org/abs/2006.10739
        
        Args:
            x (torch.Tensor): input, must be of shape (*, in_dims)
            
        Returns:
            torch.Tensor: lifted input features of shape (*, # of fourier features)
        """
        x_proj = 2 * torch.pi * x @ self.B 
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    
    def _latent_code(self, H, W, dist_param=1.0):
        """Generate latent code for inference
        
        Args:
            H (int): size of domain along dim 0
            W (int): size of domain along dim 1
            dist_param (float[Optional]): parameter for the latent distribution
            
        Returns:
            torch.Tensor: latent code in a grid of shape (H, W, zdim)
        """
        return torch.FloatTensor(1, self.zdim).normal_(dist_param).repeat(H, W, 1)
    
    def _postprocess(self, topo_delta):
        """Compute topographic map. This function can be modified freely.
        
        Args:
            topo_delta (torch.Tensor): output of the layers of shape (H, W, 1)
            
        Returns:
            np.ndarray: actual height map of the planet of shape (H, W)
        """
        # makes things spiky
        delta = (1 - torch.abs(topo_delta))**2

        # choose to give spiky (mountainous) vs flat (plateau-y) terrain
        # also because I couldn't think of a better thing to do, but this function
        if self.spiky:  # spiky only above the min_delta threshold
            terrain = delta * (delta > self.min_delta)
        else:  # cut off the spikiness at the min delta threshold
            terrain = self.min_delta * (delta > self.min_delta) + delta * (delta <= self.min_delta)
            
        return terrain.squeeze()

    def forward(self, x, latent=None, scales=1):
        """Generate topography for the planet
        
        Args:
            x (torch.Tensor): domain for the sphere in Euclidean coordinates of shape (H,W,3)
                where H & W is the resolution of the phi & theta space
            latent (torch.Tensor[Optional]): latent code for generation of shape (H,W,zdim)
                If none provided, a default code will be generated.
            multiplicative (bool[Optional]): multiply the first layer output with the latent
                layer output, otherwise add latent layer output (default: False)
            scale (float[Optional]): scale of latent features (zooming effect)
                
        Returns:
            torch.Tensor: topographic (height) map of shape (H,W)
        """
        if latent is None:
            latent = self._latent_code(*x.shape[:2])
            
        if self.ffeats > 0:
            x = self._fourier_feature_transform(x)

        topo_map = torch.zeros(x.shape[:2])
        for i in range(scales):
            coords = self.xyzlayer((self.noise_scale**(i+1))*x)
            coords += self.latent_layer(latent)
            coords = self.init_act(coords)
            coords = self.mlp(coords)
            deltas = self.final_act(coords)
            topo_map += ((self.noise_decay**(i+1)) * self._postprocess(deltas))
            
        return (topo_map + self.radius).detach()
            