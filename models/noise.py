import torch
import torch.nn as nn
from torch import Tensor

class NoiseInjector(nn.Module):
    def __init__(self, std=0.25) -> None:
        super().__init__()
        self.std = std
        self.noise = None

    def get_noise(self, x):
        if self.noise is None:
            # Generates values constrained between 0.0 and 2.0 following a normal distribution 
            # with mean 1.0 and standard deviation 0.25
            self.noise = (
                torch.fmod(torch.randn_like(x, device=x.device) * self.std, 1) + 1
            ) 
        return self.noise

    def forward(self, x: Tensor) -> Tensor:
        return x * self.get_noise(x)

    def reset_noise(self):
        self.noise = None

