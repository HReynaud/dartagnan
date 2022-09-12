import os
import numpy as np

class LazyVideoLoader:
    def __init__(self, root, filenames, suffix=".npy"):
        self.root = root
        self.filenames = filenames
        self.suffix = suffix
        self.videos = [None] * len(filenames)

    def __getitem__(self, index):
        if self.videos[index] is None:
            self.videos[index] = np.load(
                os.path.join(self.root, self.filenames[index] + self.suffix)
            )
        return self.videos[index]