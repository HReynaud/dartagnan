import os
import pandas as pd
import numpy as np
import tqdm
from torch.utils.data import Dataset

from utils.datasets import LazyVideoLoader


class Echo2s(Dataset):
    """
    Base dataset for DARTAGNAN.
    Uses a pre-processed version of EchoNet dynamic.
    The dataset is supposed to be generated by the appropriate script, see the tools folder.
    """

    def __init__(self, args, split="TRAIN") -> None:
        super().__init__()
        self.args = args

        self.metainfo = pd.read_csv(os.path.join(self.args.dataset.root, "metainfo.csv"))
        self.metainfo = self.metainfo.drop(
            self.metainfo[self.metainfo.Split != split.upper()].index
        )
        self.video_folder = os.path.join(os.path.join(self.args.dataset.root, "videos"))

        self.videos = LazyVideoLoader(
            self.video_folder, self.metainfo["FileName"].to_list(), suffix=".npy"
        )

    def __getitem__(self, index):
        fname = self.metainfo.iloc[index, 1]
        lvef = self.metainfo.iloc[index, 2]
        es_index = self.metainfo.iloc[index, 12], # ES
        ed_index =self.metainfo.iloc[index, 13],  # ED

        video = (self.videos[index][None, :, :, :] / 255.0).astype(np.float16)  # CxFxHxW

        if self.args.dataset.video_downscale:
            video = video[:, :, :: self.args.dataset.video_downscale, :: self.args.dataset.video_downscale]

        if self.args.dataset.video_step > 1:
            video = video[:, :: self.args.dataset.video_step, :, :]

        return fname, video, lvef, es_index, ed_index

    def __len__(self):
        return self.metainfo.shape[0]



