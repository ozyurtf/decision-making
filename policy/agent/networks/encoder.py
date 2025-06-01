import torch.nn as nn
import utils

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 512

        # TODO: Define convnet with activation layers
        self.convnet = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 10, stride = 2, padding = 1),
                                     nn.ReLU(),
                                     nn.Flatten())

        # TODO: Define linear layer to map convnet output to representation dimension
        self.linear = nn.Linear(9409, self.repr_dim)

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs - 0.5
        # TODO: Forward pass using obs as input
        h = self.convnet(obs)
        h = self.linear(h)

        return h
