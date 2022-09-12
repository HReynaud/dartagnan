"""
The following class serves as a base class for all trainable models.
"""

import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x):
        raise NotImplementedError

    def get_parameters(self):
        return self.parameters()

    def with_weights(self, path):
        loaded_dict = torch.load(path)
        self.load_state_dict(loaded_dict["weights"])
        return self