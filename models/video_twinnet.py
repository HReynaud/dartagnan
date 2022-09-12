import torch
import torch.nn as nn

from .base import BaseModel
from .noise import NoiseInjector
from .video_branch import VideoBranch


class VideoTwinNet(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.noise_injector = NoiseInjector(std=0.25)
        self.branch = VideoBranch(args, self.noise_injector)
        self.branch_star = (
            self.branch
            if args.model.share_weights
            else VideoBranch(args, self.noise_injector)
        )
    
    def forward(self, X, Xs, Z):

        self.noise_injector.reset_noise()

        Y = self.branch(X, Z)
        Ys = self.branch_star(Xs, Z)

        return Y, Ys